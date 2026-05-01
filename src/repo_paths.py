"""Repository root resolution (parent directory of ``src``)."""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """Return absolute path to the repository root (directory containing ``src``)."""
    return Path(__file__).resolve().parent.parent
