"""Backend discovery and executor factory."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc import ExecutorBackend

_REGISTRY: dict[str, type["ExecutorBackend"]] = {}
_LOADED: set = set()


def register_backend(cls: type["ExecutorBackend"]) -> type["ExecutorBackend"]:
    """Decorator to register a backend class."""
    _REGISTRY[cls.name] = cls  # type: ignore[attr-defined]
    return cls


def get_executor(name: str | None = None) -> "ExecutorBackend":
    """Get an executor instance.

    Args:
        name: Backend name ('numpy' or 'torch').
              None = auto-select (torch > numpy > error).

    """
    _discover()
    if name is None:
        for preferred in ("torch", "numpy"):
            if preferred in _REGISTRY:
                return _REGISTRY[preferred]()
        raise RuntimeError("No executor backend available. Install numpy or torch.")
    if name not in _REGISTRY:
        raise ValueError(
            f"Backend '{name}' not available. Installed: {list(_REGISTRY.keys())}",
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
                __import__(mod_name)
            except ImportError:
                pass
            finally:
                _LOADED.add(mod_name)
