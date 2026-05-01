"""Append-only checkpoint manifest for experiment provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_jsonable(val: Any) -> Any:
    if hasattr(val, "item"):
        try:
            return float(val.item())
        except (ValueError, TypeError):
            return str(val)
    if isinstance(val, (float, int, str, bool)) or val is None:
        return val
    return str(val)


def append_checkpoint_record(manifest_path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object per line (JSONL)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_training_summary(
    summary_path: Path,
    *,
    best_epoch: int,
    best_checkpoint: str,
    best_val_total: float,
    manifest_path: Path,
    experiment_dir: str,
) -> None:
    payload = {
        "experiment_dir": experiment_dir,
        "manifest_path": str(manifest_path),
        "best_epoch": best_epoch,
        "best_checkpoint": best_checkpoint,
        "best_val_total": best_val_total,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
