"""Integration tests for real tile-collision extraction.

Tests are integration-marked because they need PyBoy + ROM + init.state.
"""
import pytest

from pokemon_planner._tilesets import (
    COLLISION_UNKNOWN,
    COLLISION_WALKABLE,
)
from pokemon_planner.env import PokeBoy, read_state


@pytest.mark.integration
def test_tile_collision_no_longer_all_zeros(rom_path, init_state_path):
    """Real implementation should produce non-zero collision codes for some cells."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        coll = state.tile_collision
        assert len(coll) == 256
        nonzero_count = sum(1 for c in coll if c != 0)
        assert nonzero_count > 0, (
            "All collision codes are 0 — stub still active or PyBoy returned no tile data"
        )
    finally:
        pb.close()


@pytest.mark.integration
def test_tile_collision_codes_in_valid_range(rom_path, init_state_path):
    """All cells should be valid codes (0-5 or 255)."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        valid_codes = {0, 1, 2, 3, 4, 5, 255}
        for code in state.tile_collision:
            assert code in valid_codes, f"Invalid collision code: {code}"
    finally:
        pb.close()


@pytest.mark.integration
def test_tile_collision_deterministic(rom_path, init_state_path):
    """Same state, two reads -> identical collision matrix."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        s1 = read_state(pb.pyboy)
        s2 = read_state(pb.pyboy)
        assert s1.tile_collision == s2.tile_collision
    finally:
        pb.close()


@pytest.mark.integration
def test_tile_collision_center_cell_is_player_position(rom_path, init_state_path):
    """The center of the 16x16 grid (index 8*16+8 = 136) is at the player's exact position.
    The player's own tile is always walkable (you can stand there)."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        center_code = state.tile_collision[8 * 16 + 8]
        # Player's tile should never be COLLISION_BLOCKED -- they're standing on it.
        # Could be walkable, warp, or unknown (off-map edge case)
        assert center_code in {COLLISION_WALKABLE, 5, COLLISION_UNKNOWN}, (
            f"Center cell has unexpected code {center_code}"
        )
    finally:
        pb.close()


@pytest.mark.integration
def test_tile_collision_differs_across_save_states(rom_path, init_state_path, repo_root):
    """Two different save states should produce different collision matrices.

    Catches scroll-related bugs: if SCX/SCY are ignored, a state with nonzero
    scroll reads wrong BG tiles and may coincidentally match another state's
    layout.  The two states here use different map areas, so their matrices must
    differ.
    """
    other_save = repo_root / "has_pokedex_nballs.state"
    if not other_save.exists():
        pytest.skip(f"Save state not found: {other_save}")

    pb1 = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    pb2 = PokeBoy(rom_path=str(rom_path), save_state_path=str(other_save))
    try:
        s1 = read_state(pb1.pyboy)
        s2 = read_state(pb2.pyboy)
        assert s1.tile_collision != s2.tile_collision, (
            "init.state and has_pokedex_nballs.state should produce different "
            "collision matrices — if they're identical, extraction is broken"
        )
    finally:
        pb1.close()
        pb2.close()
