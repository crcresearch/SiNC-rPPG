#!/usr/bin/env python3
"""Cross-platform entry point for PURE K-fold training and evaluation.

Run from the repository root, for example:

  uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper
  uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper

Training/test subprocesses invoke Hydra (e.g. ``experiment_root=...``, ``K=...``, ``dataset=...``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _src_dir() -> Path:
    return _repo_root() / "src"


def _resolve_experiment_root(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p


def _run_train(args: argparse.Namespace) -> int:
    src = _src_dir()
    root = _resolve_experiment_root(args.experiment_root)
    dataset = args.dataset
    k_min = args.k_min
    k_max = args.k_max
    for k in range(k_min, k_max + 1):
        cmd = [
            sys.executable,
            "train.py",
            f"experiment_root={root}",
            f"K={k}",
            f"dataset={dataset}",
        ]
        print("Running:", " ".join(cmd), f"(cwd={src})", flush=True)
        r = subprocess.run(cmd, cwd=src, check=False)
        if r.returncode != 0:
            return r.returncode
    return 0


def _run_test(args: argparse.Namespace) -> int:
    src = _src_dir()
    root = _resolve_experiment_root(args.experiment_root)
    cmd = [
        sys.executable,
        "test.py",
        f"experiment_root={root}",
    ]
    print("Running:", " ".join(cmd), f"(cwd={src})", flush=True)
    return subprocess.run(cmd, cwd=src, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SiNC-rPPG training (K folds) or testing without shell scripts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train one model per K fold (default K=0..14).")
    p_train.add_argument(
        "--experiment-root",
        default="experiments/PURE_exper",
        help="Directory under repo root (or absolute) where fold outputs are stored.",
    )
    p_train.add_argument(
        "--dataset",
        default="pure_unsupervised",
        help="Training dataset flag passed to train.py (default: pure_unsupervised).",
    )
    p_train.add_argument("--k-min", type=int, default=0, help="Minimum fold index K (inclusive).")
    p_train.add_argument("--k-max", type=int, default=14, help="Maximum fold index K (inclusive).")
    p_train.set_defaults(_runner=_run_train)

    p_test = sub.add_parser("test", help="Run test.py on a completed experiment root.")
    p_test.add_argument(
        "--experiment-root",
        default="experiments/PURE_exper",
        help="Directory under repo root (or absolute) containing fold_* subdirectories.",
    )
    p_test.set_defaults(_runner=_run_test)

    ns = parser.parse_args()
    return int(ns._runner(ns))


if __name__ == "__main__":
    raise SystemExit(main())
