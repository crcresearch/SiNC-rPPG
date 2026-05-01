"""Training-time mixture of multiple registered datasets (ConcatDataset + optional weights)."""

from __future__ import annotations

import copy
import sys
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, Dataset


def _entry_to_dict(entry: Any) -> dict:
    if OmegaConf.is_config(entry):
        return dict(OmegaConf.to_container(entry, resolve=True))
    if isinstance(entry, dict):
        return dict(entry)
    return dict(vars(entry)) if hasattr(entry, "__dict__") else {}


class MixedTrainDataset(Dataset):
    """``ConcatDataset`` over registry children; optional ``sample_weights`` for ``WeightedRandomSampler``.

    Configure with Hydra ``mixed_sub_datasets`` (list of ``dataset`` + optional ``weight``).
    If every entry has a ``weight``, ``sample_weights`` is built so each child's total sampling
    mass matches normalized weights (each clip in child *i* gets ``w_i / len(child_i)``).
    If any weight is missing, ``sample_weights`` is ``None`` and the trainer uses shuffle over
    the concat (length-proportional mixing).
    """

    def __init__(self, split: str, arg_obj):
        super().__init__()
        from datasets.dataset_registry import DATASET_REGISTRY

        spec = getattr(arg_obj, "mixed_sub_datasets", None)
        if spec is None or (isinstance(spec, (list, tuple)) and len(spec) == 0):
            print("mixed_sub_datasets missing or empty for mixed_* dataset. Exiting.")
            sys.exit(-1)

        split = split.lower()
        self.arg_obj = arg_obj
        # Match BaseRPPGDataset: validation uses val_set.fps (see validate.infer_over_dataset_training).
        self.fps = int(arg_obj.fps)
        children: list[Dataset] = []
        raw_weights: list[float | None] = []
        keys: list[str] = []

        for entry in spec:
            d = _entry_to_dict(entry)
            key = str(d.get("dataset", "")).strip().lower()
            if not key:
                print("mixed_sub_datasets entry missing 'dataset'. Exiting.")
                sys.exit(-1)
            keys.append(key)
            w = d.get("weight", None)
            try:
                raw_weights.append(float(w) if w is not None else None)
            except (TypeError, ValueError):
                raw_weights.append(None)

            child_arg = copy.copy(arg_obj)
            child_arg.dataset = key
            if hasattr(child_arg, "mixed_sub_datasets"):
                delattr(child_arg, "mixed_sub_datasets")

            cls = DATASET_REGISTRY.get(key)
            children.append(cls(split, child_arg))

        self._child_keys = tuple(keys)
        self._datasets = tuple(children)
        self.concat = ConcatDataset(children)

        if any(w is None for w in raw_weights):
            self.sample_weights = None
            if any(w is not None for w in raw_weights):
                print(
                    "Warning: mixed_sub_datasets has partial weights; using length-proportional "
                    "shuffle (set weight on every entry for WeightedRandomSampler)."
                )
        else:
            w_tensor = torch.tensor(raw_weights, dtype=torch.double)
            if float(w_tensor.sum()) <= 0:
                print("mixed_sub_datasets weights must sum to a positive value. Exiting.")
                sys.exit(-1)
            w_norm = w_tensor / w_tensor.sum()
            pieces: list[torch.Tensor] = []
            for i, ds in enumerate(children):
                n = len(ds)
                if n == 0:
                    print(f"Child dataset {keys[i]} is empty. Exiting.")
                    sys.exit(-1)
                pieces.append(torch.full((n,), w_norm[i] / float(n), dtype=torch.double))
            self.sample_weights = torch.cat(pieces)

        lens = [len(c) for c in children]
        print(
            f"MixedTrainDataset ({arg_obj.dataset}): children={list(zip(keys, lens))} "
            f"total_len={len(self)} weighted={self.sample_weights is not None}"
        )
        if self.sample_weights is not None:
            idx = 0
            masses = []
            for n in lens:
                masses.append(float(self.sample_weights[idx : idx + n].sum()))
                idx += n
            print("  per-child weight mass (target shares):", dict(zip(keys, masses)))

    def __len__(self) -> int:
        return len(self.concat)

    def __getitem__(self, idx: int):
        return self.concat[idx]
