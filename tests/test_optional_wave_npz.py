"""NPZ without oximeter ``wave``: unsupervised loaders accept; supervised require ``wave``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from datasets.PURE_supervised import PURESupervised
from datasets.PURE_unsupervised import PUREUnsupervised
from datasets.UBFC_supervised import UBFCSupervised
from datasets.UBFC_unsupervised import UBFCUnsupervised


def _arg_pure(tmp: Path, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
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
    )


def _arg_ubfc(tmp: Path, dataset: str) -> SimpleNamespace:
    return _arg_pure(tmp, dataset)


def test_pure_unsupervised_npz_without_wave(tmp_path: Path) -> None:
    pre = tmp_path / "preprocessed" / "PURE"
    pre.mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    clip = pre / "clip.npz"
    video = np.zeros((250, 64, 64, 3), dtype=np.uint8)
    np.savez(clip, video=video)
    pd.DataFrame([{"subj_id": 5, "sess_id": 1, "path": "clip.npz"}]).to_csv(
        tmp_path / "metadata" / "PURE.csv", index=False
    )
    ds = PUREUnsupervised("train", _arg_pure(tmp_path, "pure_unsupervised"))
    assert len(ds) > 0
    sample = ds[0]
    assert len(sample) == 4


def test_pure_supervised_missing_wave_raises(tmp_path: Path) -> None:
    pre = tmp_path / "preprocessed" / "PURE"
    pre.mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    clip = pre / "clip.npz"
    np.savez(clip, video=np.zeros((250, 64, 64, 3), dtype=np.uint8))
    pd.DataFrame([{"subj_id": 5, "sess_id": 1, "path": "clip.npz"}]).to_csv(
        tmp_path / "metadata" / "PURE.csv", index=False
    )
    with pytest.raises(ValueError, match="missing required key 'wave'"):
        PURESupervised("train", _arg_pure(tmp_path, "pure_supervised"))


def test_ubfc_unsupervised_npz_without_wave(tmp_path: Path) -> None:
    pre = tmp_path / "preprocessed" / "UBFC"
    pre.mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    clip = pre / "u.npz"
    video = np.zeros((250, 64, 64, 3), dtype=np.uint8)
    np.savez(clip, video=video)
    pd.DataFrame([{"id": 5, "path": "u.npz"}]).to_csv(
        tmp_path / "metadata" / "UBFC.csv", index=False
    )
    ds = UBFCUnsupervised("train", _arg_ubfc(tmp_path, "ubfc_unsupervised"))
    assert len(ds) > 0
    assert len(ds[0]) == 4


def test_ubfc_supervised_missing_wave_raises(tmp_path: Path) -> None:
    pre = tmp_path / "preprocessed" / "UBFC"
    pre.mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    clip = pre / "u.npz"
    np.savez(clip, video=np.zeros((250, 64, 64, 3), dtype=np.uint8))
    pd.DataFrame([{"id": 5, "path": "u.npz"}]).to_csv(
        tmp_path / "metadata" / "UBFC.csv", index=False
    )
    with pytest.raises(ValueError, match="missing required key 'wave'"):
        UBFCSupervised("train", _arg_ubfc(tmp_path, "ubfc_supervised"))
