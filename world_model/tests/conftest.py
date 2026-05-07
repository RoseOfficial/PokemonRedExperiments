"""Shared pytest fixtures for the pokemon_planner test suite."""
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repo root (parent of world_model/)."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def rom_path(repo_root: Path) -> Path:
    """Path to PokemonRed.gb at the repo root.

    Tests using this fixture should be marked @pytest.mark.integration so
    they can be skipped when the ROM is unavailable.
    """
    p = repo_root / "PokemonRed.gb"
    if not p.exists():
        pytest.skip(f"ROM not found at {p} (legally obtain and place at repo root)")
    return p


@pytest.fixture(scope="session")
def init_state_path(repo_root: Path) -> Path:
    """Path to init.state save state at the repo root."""
    p = repo_root / "init.state"
    if not p.exists():
        pytest.skip(f"Save state not found at {p}")
    return p
