"""Hydra compose and path merge smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


def test_hydra_paths_resolve_to_repo_data():
    repo = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf"), job_name="pytest"):
        cfg = compose(config_name="config")
    from config_merge import hydra_cfg_to_arg_namespace

    ns = hydra_cfg_to_arg_namespace(cfg)
    assert getattr(ns, "testing_dataset", None) == "pure_testing"
    assert ns.optimization_step == "unsupervised"
    assert ns.losses == "bsv"
    assert ns.metadata_dir.is_absolute()
    assert ns.metadata_dir.name == "metadata"
    assert (repo / "data" / "metadata").resolve() == ns.metadata_dir
    assert (repo / "data" / "preprocessed").resolve() == ns.preprocessed_dir


def test_hydra_compose_mixed_unsupervised():
    repo = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf"), job_name="pytest"):
        cfg = compose(config_name="config", overrides=["dataset=mixed_unsupervised"])
    from config_merge import hydra_cfg_to_arg_namespace

    ns = hydra_cfg_to_arg_namespace(cfg)
    assert ns.dataset == "mixed_unsupervised"
    assert len(ns.mixed_sub_datasets) == 2
    assert ns.mixed_sub_datasets[0]["dataset"] == "pure_unsupervised"
    assert float(ns.mixed_sub_datasets[0]["weight"]) == pytest.approx(0.7)


def test_hydra_compose_pure_supervised_uses_supervised_training():
    repo = Path(__file__).resolve().parent.parent
    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf"), job_name="pytest"):
        cfg = compose(config_name="config", overrides=["dataset=pure_supervised"])
    from config_merge import hydra_cfg_to_arg_namespace

    ns = hydra_cfg_to_arg_namespace(cfg)
    assert ns.dataset == "pure_supervised"
    assert ns.optimization_step == "supervised"
    assert ns.validation_step == "supervised"
    assert ns.losses == "supervised"
    assert ns.validation_loss == "supervised"
