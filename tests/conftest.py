"""Pytest configuration: auto-skip missing backends."""

import importlib.util

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark tests requiring unavailable backends."""
    del config
    # Check numpy availability
    if importlib.util.find_spec("numpy") is None:
        for item in items:
            if "numpy" in item.nodeid.lower() or "np_exec" in getattr(
                item, "fixturenames", []
            ):
                item.add_marker(pytest.mark.skip(reason="NumPy not installed"))

    # Check torch availability
    if importlib.util.find_spec("torch") is None:
        for item in items:
            if "torch" in item.nodeid.lower() or "pt_exec" in getattr(
                item, "fixturenames", []
            ):
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))
