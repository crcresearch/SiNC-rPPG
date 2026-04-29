"""Hydra compose and path merge smoke test."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_paths_resolve_to_repo_data():
    repo = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf"), job_name="pytest"):
        cfg = compose(config_name="config")
    from config_merge import hydra_cfg_to_arg_namespace

    ns = hydra_cfg_to_arg_namespace(cfg)
    assert getattr(ns, "testing_dataset", None) == "pure_testing"
    assert ns.metadata_dir.is_absolute()
    assert ns.metadata_dir.name == "metadata"
    assert (repo / "data" / "metadata").resolve() == ns.metadata_dir
    assert (repo / "data" / "preprocessed").resolve() == ns.preprocessed_dir
