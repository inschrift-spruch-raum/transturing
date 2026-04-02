"""Pytest configuration: auto-skip missing backends."""


import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests requiring unavailable backends."""
    # Check numpy availability
    try:
        import numpy
    except ImportError:
        for item in items:
            if "numpy" in item.nodeid.lower() or "np_exec" in item.fixturenames:
                item.add_marker(pytest.mark.skip(reason="NumPy not installed"))

    # Check torch availability
    try:
        import torch
    except ImportError:
        for item in items:
            if "torch" in item.nodeid.lower() or "pt_exec" in item.fixturenames:
                item.add_marker(pytest.mark.skip(reason="PyTorch not installed"))
