"""Resolve the best available PyTorch device (CUDA, MPS, XPU, CPU)."""

import torch


def select_torch_device() -> torch.device:
    """Prefer CUDA (NVIDIA / ROCm), then Apple MPS, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        is_available = getattr(xpu, "is_available", None)
        if callable(is_available) and is_available():
            return torch.device("xpu")
    return torch.device("cpu")
