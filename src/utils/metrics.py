"""Scalar signal quality metrics for rPPG / HR evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def mean_absolute_error(pred: Any, ref: Any) -> float:
    """Mean absolute error between ``pred`` and ``ref`` (broadcasting like NumPy)."""
    p = _to_numpy(pred).reshape(-1)
    r = _to_numpy(ref).reshape(-1)
    if p.shape != r.shape:
        raise ValueError(f"pred and ref must broadcast to same shape; got {p.shape} vs {r.shape}")
    return float(np.mean(np.abs(p - r)))


def snr_db(signal: Any, reference: Any, eps: float = 1e-12) -> float:
    """Waveform SNR in dB: ``10 log10( mean(ref^2) / mean((sig - ref)^2) )``.

    Higher is better when ``signal`` is an estimate of ``reference``.
    """
    s = _to_numpy(signal).reshape(-1)
    r = _to_numpy(reference).reshape(-1)
    if s.shape != r.shape:
        raise ValueError(f"signal and reference must have same shape; got {s.shape} vs {r.shape}")
    noise = s - r
    p_sig = float(np.mean(r**2))
    p_noise = float(np.mean(noise**2))
    if p_noise < eps:
        return float("inf") if p_sig > eps else float("-inf")
    return float(10.0 * np.log10((p_sig + eps) / (p_noise + eps)))
