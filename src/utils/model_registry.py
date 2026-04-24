"""Central registration of model type strings to model classes."""

from __future__ import annotations

from utils.registry import Registry

MODEL_REGISTRY = Registry()


def _register_models() -> None:
    from models.PhysNet import PhysNet
    from models.RPNet import RPNet

    MODEL_REGISTRY.register("physnet", PhysNet)
    MODEL_REGISTRY.register("rpnet", RPNet)


_register_models()
