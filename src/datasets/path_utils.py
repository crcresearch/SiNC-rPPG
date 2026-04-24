"""Resolve metadata and preprocessed sample paths for rPPG datasets."""

from __future__ import annotations

from pathlib import Path


def preprocessed_subdir_for_dataset(dataset_key: str) -> str:
    """Map ``pure_unsupervised``-style keys to ``data/preprocessed`` folder names."""
    root = dataset_key.lower().split("_")[0]
    mapping = {
        "pure": "PURE",
        "ubfc": "UBFC",
        "ddpm": "DDPM",
        "hkbu": "HKBU",
        "celebv": "CelebV",
    }
    return mapping.get(root, root.upper())


def resolve_npz_path(raw_path: str, preprocessed_dir: Path, subdir: str) -> Path:
    """Resolve a ``path`` cell from metadata CSV to an existing ``.npz`` file."""
    p = Path(raw_path).expanduser()
    if p.is_file():
        return p.resolve()
    name = p.name
    candidate = (preprocessed_dir / subdir / name).resolve()
    if candidate.is_file():
        return candidate
    candidate2 = (preprocessed_dir / subdir / p).resolve()
    if candidate2.is_file():
        return candidate2
    return p
