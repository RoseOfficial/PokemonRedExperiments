"""Integration tests for state extraction from PyBoy.

Marked as `integration` — skipped if ROM/save-state unavailable.
"""
import pytest

from pokemon_planner.env import PokeBoy, read_state
from pokemon_planner.state import GameState


@pytest.mark.integration
def test_extract_from_init_state(rom_path, init_state_path):
    """Load init.state, extract state, verify structural validity.

    init.state lands somewhere past the title screen. We don't assert
    *which* map — we assert the extracted state is internally coherent.
    """
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        assert isinstance(state, GameState)

        # Structural sanity (not value-specific — init.state may evolve)
        assert 0 <= state.party_size <= 6
        assert 0 <= state.x <= 255
        assert 0 <= state.y <= 255
        assert 0 <= state.map_id <= 255
        assert 0 <= state.money < 1_000_000   # BCD max for 3 bytes is 999999
        assert 0 <= state.badges <= 255
        assert len(state.event_flags) == 256
        assert len(state.tile_collision) == 256

        # If party is non-empty, validate slots
        for slot in state.party:
            assert 1 <= slot.level <= 100
            assert slot.hp_cur <= slot.hp_max
            assert 0 <= slot.species_id <= 255
    finally:
        pb.close()


@pytest.mark.integration
def test_state_is_immutable(rom_path, init_state_path):
    """Pydantic frozen=True means extracted states can't be mutated in place."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        with pytest.raises(Exception):  # pydantic raises ValidationError on frozen field assign
            state.x = 99  # type: ignore[misc]
    finally:
        pb.close()


@pytest.mark.integration
def test_two_consecutive_reads_are_equal(rom_path, init_state_path):
    """Reading state twice without stepping should produce equal results."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        s1 = read_state(pb.pyboy)
        s2 = read_state(pb.pyboy)
        assert s1 == s2
    finally:
        pb.close()
