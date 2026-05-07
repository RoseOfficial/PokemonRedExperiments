"""Tests for the structured RAM state schema."""
import pytest
from pydantic import ValidationError

from pokemon_planner.state import (
    BattleState,
    PartySlot,
    GameState,
)


def test_partyslot_valid():
    slot = PartySlot(
        species_id=0x01, level=5, hp_cur=20, hp_max=20, status=0,
        moves=(0x21, 0x00, 0x00, 0x00),  # Tackle + 3 empty
    )
    assert slot.species_id == 0x01
    assert slot.level == 5


def test_partyslot_rejects_oversized_level():
    with pytest.raises(ValidationError):
        PartySlot(species_id=1, level=101, hp_cur=10, hp_max=10, status=0,
                  moves=(0, 0, 0, 0))


def test_partyslot_rejects_hp_over_max():
    with pytest.raises(ValidationError):
        PartySlot(species_id=1, level=5, hp_cur=30, hp_max=20, status=0,
                  moves=(0, 0, 0, 0))


def test_battlestate_zeroed_when_not_in_battle():
    bs = BattleState(in_battle=False)
    assert bs.opp_species_id == 0
    assert bs.opp_level == 0
    assert bs.opp_hp == 0
    assert bs.turn == 0


def test_gamestate_minimal():
    gs = GameState(
        map_id=0x00,
        x=5,
        y=5,
        party=(),
        bag=(),
        badges=0,
        event_flags=bytes(256),
        money=0,
        time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256),
        menu_flags=0,
    )
    assert gs.party_size == 0
    assert gs.has_badge(0) is False


def test_gamestate_party_size_property():
    party = (
        PartySlot(species_id=1, level=5, hp_cur=20, hp_max=20, status=0,
                  moves=(0, 0, 0, 0)),
    )
    gs = GameState(
        map_id=0x00, x=5, y=5,
        party=party,
        bag=(), badges=0,
        event_flags=bytes(256),
        money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )
    assert gs.party_size == 1


def test_gamestate_has_badge():
    gs = GameState(
        map_id=0x00, x=5, y=5, party=(), bag=(),
        badges=0b00000011,  # Boulder + Cascade badges
        event_flags=bytes(256),
        money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )
    assert gs.has_badge(0) is True   # Boulder
    assert gs.has_badge(1) is True   # Cascade
    assert gs.has_badge(2) is False  # Thunder


def test_gamestate_event_flags_size_validated():
    with pytest.raises(ValidationError):
        GameState(
            map_id=0x00, x=5, y=5, party=(), bag=(), badges=0,
            event_flags=bytes(100),   # wrong length — should be 256
            money=0, time_played_frames=0,
            battle=BattleState(in_battle=False),
            tile_collision=bytes(256), menu_flags=0,
        )
