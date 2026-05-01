"""Lightweight name-to-class registry."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class Registry(dict[str, type[T]]):
    """Maps lowercase string keys to callables (typically model or dataset classes)."""

    def register(self, name: str, cls: type[T]) -> type[T]:
        key = name.lower()
        if key in self:
            raise ValueError(f"Duplicate registry key: {key!r}")
        self[key] = cls
        return cls

    def get(self, name: str) -> type[T]:  # type: ignore[override]
        key = name.lower()
        if key not in self:
            keys = ", ".join(sorted(self))
            raise KeyError(f"Unknown key {key!r}. Registered keys: {keys}")
        return self[key]
