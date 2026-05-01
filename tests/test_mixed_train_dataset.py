"""Mixed PURE+UBFC training dataset and training DataLoader construction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from torch.utils.data import RandomSampler, WeightedRandomSampler

from datasets.mixed_train import MixedTrainDataset
from datasets.utils import build_training_dataloader


def _base_arg(tmp: Path, dataset: str, mixed_spec: list, **extra):
    d = dict(
        dataset=dataset,
        mixed_sub_datasets=mixed_spec,
        metadata_dir=str(tmp / "metadata"),
        preprocessed_dir=str(tmp / "preprocessed"),
        K=0,
        debug=0,
        channels="rgb",
        fpc=120,
        step=60,
        augmentation="",
        speed_slow=0.6,
        speed_fast=1.4,
        fps=30,
        frame_width=64,
        frame_height=64,
        batch_size=2,
        num_workers=0,
    )
    d.update(extra)
    return SimpleNamespace(**d)


def _write_pure_ubfc_npz(
    tmp: Path, *, supervised: bool, pure_name: str = "clip.npz", ubfc_name: str = "u.npz"
) -> None:
    (tmp / "metadata").mkdir(parents=True)
    pre_p = tmp / "preprocessed" / "PURE"
    pre_u = tmp / "preprocessed" / "UBFC"
    pre_p.mkdir(parents=True)
    pre_u.mkdir(parents=True)
    video = np.zeros((250, 64, 64, 3), dtype=np.uint8)
    if supervised:
        wave = (0.5 + 0.1 * np.sin(np.linspace(0.0, 12.0, 250, dtype=np.float32))).astype(np.float32)
        np.savez(pre_p / pure_name, video=video, wave=wave)
        np.savez(pre_u / ubfc_name, video=video, wave=wave)
    else:
        np.savez(pre_p / pure_name, video=video)
        np.savez(pre_u / ubfc_name, video=video)
    pd.DataFrame([{"subj_id": 5, "sess_id": 1, "path": pure_name}]).to_csv(
        tmp / "metadata" / "PURE.csv", index=False
    )
    pd.DataFrame([{"id": 5, "path": ubfc_name}]).to_csv(
        tmp / "metadata" / "UBFC.csv", index=False
    )


def test_mixed_unsupervised_weighted_concat(tmp_path: Path) -> None:
    _write_pure_ubfc_npz(tmp_path, supervised=False)
    arg = _base_arg(
        tmp_path,
        "mixed_unsupervised",
        [
            {"dataset": "pure_unsupervised", "weight": 0.7},
            {"dataset": "ubfc_unsupervised", "weight": 0.3},
        ],
    )
    ds = MixedTrainDataset("train", arg)
    assert len(ds) > 0
    assert ds.fps == 30
    assert ds.sample_weights is not None
    assert ds.sample_weights.shape[0] == len(ds)
    assert abs(float(ds.sample_weights.sum()) - 1.0) < 1e-5
    sample = ds[0]
    assert sample is not None


def test_mixed_supervised_weighted_concat(tmp_path: Path) -> None:
    _write_pure_ubfc_npz(tmp_path, supervised=True)
    arg = _base_arg(
        tmp_path,
        "mixed_supervised",
        [
            {"dataset": "pure_supervised", "weight": 0.7},
            {"dataset": "ubfc_supervised", "weight": 0.3},
        ],
    )
    ds = MixedTrainDataset("train", arg)
    assert len(ds) > 0
    assert ds.sample_weights is not None
    assert len(ds[0]) == 5


def test_build_training_dataloader_weighted_sampler(tmp_path: Path) -> None:
    _write_pure_ubfc_npz(tmp_path, supervised=False)
    arg = _base_arg(
        tmp_path,
        "mixed_unsupervised",
        [
            {"dataset": "pure_unsupervised", "weight": 0.7},
            {"dataset": "ubfc_unsupervised", "weight": 0.3},
        ],
    )
    ds = MixedTrainDataset("train", arg)
    loader = build_training_dataloader(ds, arg)
    assert isinstance(loader.sampler, WeightedRandomSampler)


def test_mixed_no_weights_uses_shuffle_dataloader(tmp_path: Path) -> None:
    _write_pure_ubfc_npz(tmp_path, supervised=False)
    arg = _base_arg(
        tmp_path,
        "mixed_unsupervised",
        [
            {"dataset": "pure_unsupervised"},
            {"dataset": "ubfc_unsupervised"},
        ],
    )
    ds = MixedTrainDataset("train", arg)
    assert ds.sample_weights is None
    loader = build_training_dataloader(ds, arg)
    assert isinstance(loader.sampler, RandomSampler)
