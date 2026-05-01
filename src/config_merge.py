"""Merge Hydra ``DictConfig`` into a flat namespace compatible with existing code."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from repo_paths import get_repo_root


def _resolve_repo_path(value: str | Path | None, repo: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (repo / p).resolve()
    else:
        p = p.resolve()
    return p


def hydra_cfg_to_arg_namespace(cfg: DictConfig) -> SimpleNamespace:
    """Flatten ``model``, ``dataset``, ``training``, ``paths`` and top-level run keys."""
    repo = get_repo_root()
    merged: dict = {}
    # Order matters: later groups overwrite earlier keys. Dataset after training so
    # e.g. pure_supervised.yaml can set optimization_step/losses for supervised runs.
    for group in ("model", "training", "paths", "dataset"):
        if group in cfg and cfg[group] is not None:
            merged.update(OmegaConf.to_container(cfg[group], resolve=True))
    root_cfg = OmegaConf.to_container(cfg, resolve=True)
    skip = {"model", "dataset", "training", "paths", "hydra", "defaults"}
    for key, val in root_cfg.items():
        if key in skip:
            continue
        merged[key] = val

    path_keys = (
        "raw_dir",
        "preprocessed_dir",
        "metadata_dir",
        "experiments_dir",
        "predictions_dir",
        "results_dir",
    )
    for pk in path_keys:
        if pk in merged and merged[pk] is not None:
            merged[pk] = _resolve_repo_path(merged[pk], repo)

    er = merged.get("experiment_root")
    if er is not None and er != "":
        merged["experiment_root"] = _resolve_repo_path(er, repo)
    else:
        merged["experiment_root"] = None

    return SimpleNamespace(**merged)
