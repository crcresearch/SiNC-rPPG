#!/usr/bin/env python3
"""Inspect a predictions ``.pkl`` file written by ``engine.evaluation.run_evaluation``.

The pickle is a nested dict::

    { <testing_dataset_key>: { <seed>: { <fold>: {
        "pred_waves", "pred_HRs", "gt_waves", "gt_HRs", "ME", "MAE", "RMSE", "r"
    }}}}}

Run from ``src/`` (same as ``train.py`` / ``test.py``)::

    uv run python utils/read_predictions_pickle.py ../predictions/PURE_smoke.pkl
    uv run python utils/read_predictions_pickle.py --path ../predictions/PURE_smoke.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def _describe_value(name: str, val: object, indent: str) -> None:
    if isinstance(val, np.ndarray):
        print(f"{indent}{name}: ndarray shape={tuple(val.shape)} dtype={val.dtype}")
    elif isinstance(val, (list, tuple)):
        n = len(val)
        if n == 0:
            print(f"{indent}{name}: empty {type(val).__name__}")
            return
        el0 = val[0]
        if isinstance(el0, np.ndarray):
            shapes = [tuple(x.shape) for x in val[:3] if isinstance(x, np.ndarray)]
            more = " ..." if n > 3 else ""
            print(f"{indent}{name}: {type(val).__name__}[{n}] of ndarray (first shapes: {shapes}{more})")
        else:
            print(f"{indent}{name}: {type(val).__name__}[{n}] (element type {type(el0).__name__})")
    elif isinstance(val, (float, int, np.floating, np.integer)):
        print(f"{indent}{name}: {type(val).__name__} = {val}")
    else:
        print(f"{indent}{name}: {type(val).__name__}")


def summarize_pickle(path: Path) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() not in (".pkl", ".pickle"):
        print(f"Warning: expected .pkl extension, got {path.suffix}", file=sys.stderr)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"File: {path}")
    print(f"Top-level type: {type(data).__name__}")

    if not isinstance(data, dict):
        print("Raw object (not the usual nested dict):")
        print(repr(data)[:2000])
        return

    print(f"Top-level keys ({len(data)}): {list(data.keys())}")
    for test_key, exper in data.items():
        print()
        print(f"== testing_dataset: {test_key!r} ==")
        if not isinstance(exper, dict):
            print(f"  (unexpected type {type(exper).__name__}, repr below)")
            print(repr(exper)[:1500])
            continue
        fold_keys_example = next(iter(exper.values()), {})
        if isinstance(fold_keys_example, dict):
            print(f"  seeds: {list(exper.keys())}")
        for seed, folds in exper.items():
            print(f"  seed {seed!r}: {len(folds) if isinstance(folds, dict) else type(folds).__name__} fold(s)")
            if not isinstance(folds, dict):
                continue
            for fold in sorted(folds.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k)):
                block = folds[fold]
                print(f"    fold {fold!r}:")
                if not isinstance(block, dict):
                    print(f"      (unexpected {type(block).__name__})")
                    continue
                for k in sorted(block.keys()):
                    _describe_value(k, block[k], "      ")
    print()
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Load and summarize a SiNC predictions pickle (test.py / run_evaluation output)."
    )
    p.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to the .pkl file (e.g. predictions/PURE_smoke.pkl relative to repo root).",
    )
    p.add_argument(
        "--path",
        dest="path_flag",
        default=None,
        metavar="FILE",
        help="Same as positional path (useful when the path starts with '-').",
    )
    args = p.parse_args()
    target = args.path_flag or args.path
    if not target:
        p.error("pass the pickle path as a positional argument or use --path FILE")
    summarize_pickle(Path(target))


if __name__ == "__main__":
    main()
