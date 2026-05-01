"""Hydra ``testing_dataset`` / ``testing_datasets`` resolution for run_evaluation."""

from __future__ import annotations

from types import SimpleNamespace

from engine.evaluation import _resolve_testing_datasets


def test_resolve_defaults() -> None:
    assert _resolve_testing_datasets(SimpleNamespace()) == ["pure_testing"]


def test_resolve_single() -> None:
    assert _resolve_testing_datasets(SimpleNamespace(testing_dataset="ubfc_testing")) == [
        "ubfc_testing"
    ]


def test_resolve_list_wins() -> None:
    ns = SimpleNamespace(
        testing_datasets=["pure_testing", "ubfc_testing"],
        testing_dataset="ignored",
    )
    assert _resolve_testing_datasets(ns) == ["pure_testing", "ubfc_testing"]
