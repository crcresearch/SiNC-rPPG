"""Checkpoint manifest JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path

from utils.checkpoint_manifest import append_checkpoint_record, write_training_summary


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = tmp_path / "checkpoints_manifest.jsonl"
    append_checkpoint_record(m, {"epoch": 0, "checkpoint": "a", "is_best": True})
    append_checkpoint_record(m, {"epoch": 1, "checkpoint": "b", "is_best": False})
    lines = m.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 0

    s = tmp_path / "training_summary.json"
    write_training_summary(
        s,
        best_epoch=0,
        best_checkpoint="a",
        best_val_total=1.23,
        manifest_path=m,
        experiment_dir=str(tmp_path),
    )
    data = json.loads(s.read_text(encoding="utf-8"))
    assert data["best_epoch"] == 0
    assert data["best_val_total"] == 1.23
