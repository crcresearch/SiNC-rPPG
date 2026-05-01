from __future__ import annotations

import numpy as np
import torch

from utils.metrics import mean_absolute_error, snr_db


def test_mae_numpy() -> None:
    assert mean_absolute_error(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5


def test_mae_torch() -> None:
    p = torch.tensor([0.0, 1.0])
    r = torch.tensor([0.0, 3.0])
    assert abs(mean_absolute_error(p, r) - 1.0) < 1e-6


def test_snr_db_perfect() -> None:
    x = np.sin(np.linspace(0, 4 * np.pi, 64))
    assert snr_db(x, x) > 60.0


def test_snr_db_noisy() -> None:
    r = np.ones(32)
    s = r + 0.1 * np.random.default_rng(0).standard_normal(32)
    v = snr_db(s, r)
    assert v < 30.0
