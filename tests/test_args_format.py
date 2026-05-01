"""args.print_args / log_args must tolerate list-valued config (e.g. mixed_sub_datasets)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import args as args_mod


def test_format_val_list_for_print_args(capsys) -> None:
    ns = SimpleNamespace(
        dataset="mixed_unsupervised",
        mixed_sub_datasets=[
            {"dataset": "pure_unsupervised", "weight": 0.7},
            {"dataset": "ubfc_unsupervised", "weight": 0.3},
        ],
        metadata_dir=Path("/tmp/metadata"),
    )
    args_mod.print_args(ns)
    captured = capsys.readouterr().out
    assert "mixed_sub_datasets" in captured
    assert "pure_unsupervised" in captured
