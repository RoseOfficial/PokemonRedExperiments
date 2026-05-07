"""Smoke tests — verify the package and its core dependencies import."""
import pokemon_planner


def test_package_imports():
    assert pokemon_planner.__version__ == "0.0.1"


def test_torch_imports():
    import torch
    # Confirms torch installed; CUDA availability is informational, not required
    _ = torch.zeros(1)


def test_pyboy_imports():
    """PyBoy 2.x must be importable; cwd-independent."""
    import pyboy
    assert hasattr(pyboy, "PyBoy")


def test_pydantic_v2():
    import pydantic
    assert int(pydantic.VERSION.split(".")[0]) >= 2
