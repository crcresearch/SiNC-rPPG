"""Shared rPPG video dataset base (face crops + waveforms)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

import datasets.transforms as transforms


class BaseRPPGDataset(Dataset, ABC):
    """Common setup and clip sampling for PURE- and UBFC-style loaders."""

    def __init__(self, split: str, arg_obj):
        super().__init__()
        self.arg_obj = arg_obj
        self.round_robin_index = int(arg_obj.K)
        self.debug = bool(int(arg_obj.debug))
        self.split = split.lower()
        self.channels = arg_obj.channels.lower()
        self.frames_per_clip = int(arg_obj.fpc)
        self.step = int(arg_obj.step)
        self.aug = arg_obj.augmentation.lower()
        self.speed_slow = float(arg_obj.speed_slow)
        self.speed_fast = float(arg_obj.speed_fast)
        self.fps = int(arg_obj.fps)

        self.set_augmentations()
        self.load_data()
        self.pad_inputs()
        self.build_samples()

        print(self.split)
        print("Samples: ", self.samples.shape)
        print("Total frames: ", self.samples.shape[0] * self.frames_per_clip)

    @abstractmethod
    def set_augmentations(self) -> None:
        """Set ``self.aug_*`` flags from ``self.split`` and ``self.aug``."""

    @abstractmethod
    def load_data(self) -> None:
        """Populate ``self.data`` and ``self.waves`` (and masks if needed)."""

    @abstractmethod
    def __getitem__(self, idx):
        """Return one training or validation sample (signature varies by task)."""

    def pad_inputs(self) -> None:
        """Add a step-width pad to both ends so the whole video is processed."""
        if (self.split == "test") or (self.split == "all"):
            self.masks = []
            for i in range(len(self.waves)):
                self.data[i]["video"] = self.data[i]["video"][: len(self.waves[i])]
                pad = self.step - (len(self.waves[i]) % self.step) + 1
                mask = np.ones_like(self.waves[i], dtype=bool)
                if pad > 0:
                    self.waves[i] = np.hstack((self.waves[i], np.repeat(self.waves[i][-1], pad)))
                    back = self.data[i]["video"][[-1]].repeat(pad, 0)
                    self.data[i]["video"] = np.append(self.data[i]["video"], back, axis=0)
                    mask = np.hstack((mask, np.zeros(pad, dtype=bool)))
                self.masks.append(mask)

    def build_samples(self) -> None:
        start_idcs = self.get_start_idcs()
        samples = []
        for subj in range(len(self.waves)):
            starts = start_idcs[subj]
            subj_rep = np.repeat(subj, len(starts))
            sample = np.vstack((subj_rep, starts))
            samples.append(sample)
        self.samples = np.hstack(samples).T

    def get_start_idcs(self):
        start_idcs = []
        for wave in self.waves:
            slen = len(wave)
            end = slen - self.frames_per_clip
            starts = np.arange(0, end, self.step)
            start_idcs.append(starts)
        start_idcs = np.array(start_idcs, dtype=object)
        return start_idcs

    def get_subj_sizes(self):
        subjects = np.unique(self.samples[:, 0])
        ends = []
        for subj in subjects:
            end = self.samples[self.samples[:, 0] == subj, 1][-1]
            ends.append(end)
        frames_per_subj = np.array(ends) + self.frames_per_clip
        return frames_per_subj

    def apply_transformations(self, clip, subj, idcs, augment=True):
        speed = 1.0
        if augment:
            if self.aug_speed:
                entire_clip = self.data[subj]["video"]
                clip, idcs, speed = transforms.augment_speed(
                    entire_clip,
                    idcs,
                    self.frames_per_clip,
                    self.channels,
                    self.speed_slow,
                    self.speed_fast,
                )

            if self.aug_flip:
                clip = transforms.augment_horizontal_flip(clip)

            if self.aug_reverse:
                clip = transforms.augment_time_reversal(clip)

            if self.aug_illum:
                clip = transforms.augment_illumination_noise(clip)

            if self.aug_gauss:
                clip = transforms.augment_gaussian_noise(clip)

            if self.aug_resizedcrop:
                clip = transforms.random_resized_crop(clip)

        clip = np.clip(clip, 0, 255)
        clip = clip / 255
        clip = torch.from_numpy(clip).float()

        return clip, idcs, speed

    def __len__(self):
        return self.samples.shape[0]
