"""Smoke test: PhysNet forward and backward pass without real data."""

from __future__ import annotations

import pytest
import torch

from models.PhysNet import PhysNet


@pytest.fixture
def dummy_video_batch() -> torch.Tensor:
    """Random clip with shape [B, C, T, W, H] (same layout as model input)."""
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 120, 64, 64, generator=g, dtype=torch.float32)


def test_physnet_forward_backward(dummy_video_batch: torch.Tensor) -> None:
    model = PhysNet(input_channels=3, drop_p=0.5).float().train()
    x = dummy_video_batch.clone()
    x.requires_grad_(True)
    y = model(x)
    assert y.ndim == 2
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
