"""Backend discovery and executor factory."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc import ExecutorBackend

_REGISTRY: dict[str, type[ExecutorBackend]] = {}
_LOADED: set[str] = set()


def register_backend(cls: type[ExecutorBackend]) -> type[ExecutorBackend]:
    """Register a backend class via decorator."""
    _REGISTRY[cls.name] = cls
    return cls


def get_executor(name: str | None = None) -> ExecutorBackend:
    """
    Get an executor instance.

    Args:
        name: Backend name ('numpy' or 'torch').
              None = auto-select (torch > numpy > error).

    """
    _discover()
    if name is None:
        for preferred in ("torch", "numpy"):
            if preferred in _REGISTRY:
                return _REGISTRY[preferred]()
        msg = "No executor backend available. Install numpy or torch."
        raise RuntimeError(msg)
    if name not in _REGISTRY:
        msg = f"Backend '{name}' not available. Installed: {list(_REGISTRY.keys())}"
        raise ValueError(
            msg,
        )
    return _REGISTRY[name]()


def list_backends() -> list[str]:
    """List available backend names."""
    _discover()
    return list(_REGISTRY.keys())


def _discover() -> None:
    """Try importing backend modules to trigger registration."""
    for mod_name in (
        "transturing.backends.torch_backend",
        "transturing.backends.numpy_backend",
    ):
        if mod_name not in _LOADED:
            try:
                importlib.import_module(mod_name)
            except ImportError:
                pass
            finally:
                _LOADED.add(mod_name)
