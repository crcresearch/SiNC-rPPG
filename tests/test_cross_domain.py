from __future__ import annotations

from types import SimpleNamespace

from datasets.cross_domain import is_cross_domain, validation_arg_obj


def test_validation_arg_obj_same_when_null() -> None:
    base = SimpleNamespace(dataset="pure_unsupervised", fps=30, K=0)
    assert validation_arg_obj(base) is base


def test_validation_arg_obj_override() -> None:
    base = SimpleNamespace(
        dataset="pure_unsupervised", fps=30, K=0, validation_dataset="ubfc_unsupervised"
    )
    v = validation_arg_obj(base)
    assert v is not base
    assert v.dataset == "ubfc_unsupervised"
    assert v.fps == 30


def test_validation_fps_override() -> None:
    base = SimpleNamespace(
        dataset="pure_unsupervised",
        fps=30,
        K=0,
        validation_dataset="ubfc_unsupervised",
        validation_fps=90,
    )
    v = validation_arg_obj(base)
    assert v.fps == 90


def test_is_cross_domain() -> None:
    a = SimpleNamespace(dataset="pure_unsupervised", validation_dataset="ubfc_unsupervised")
    assert is_cross_domain(a) is True
    b = SimpleNamespace(dataset="pure_unsupervised", validation_dataset=None)
    assert is_cross_domain(b) is False
