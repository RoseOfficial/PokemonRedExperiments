"""Smoke + structural tests for the per-tileset collision data."""
from pokemon_planner._tilesets import (
    CAVERN,
    CollisionTable,
    COLLISION_TABLES,
    DEFAULT_TABLE,
    FOREST,
    GATE,
    GYM,
    HOUSE,
    MART,
    OVERWORLD,
    POKECENTER,
    code_to_name,
)


PHASE_1A_TILESETS = [OVERWORLD, HOUSE, MART, POKECENTER, FOREST, CAVERN, GYM, GATE]


def test_phase_1a_tilesets_all_have_tables():
    for ts in PHASE_1A_TILESETS:
        assert ts in COLLISION_TABLES, f"Missing collision table for tileset {ts}"


def test_overworld_table_has_blocked_tiles():
    table = COLLISION_TABLES[OVERWORLD]
    assert len(table.blocked) >= 5, "OVERWORLD should have multiple blocked tiles"


def test_collision_lookup_returns_walkable_for_unknown():
    table = COLLISION_TABLES[OVERWORLD]
    # A tile ID guaranteed to be NOT in any collision set (high value beyond table range)
    assert table.lookup(0xFE) == 0   # walkable


def test_collision_lookup_returns_blocked_for_known_wall():
    table = COLLISION_TABLES[OVERWORLD]
    # Pick any blocked tile and check it returns 1
    one_blocked = next(iter(table.blocked))
    assert table.lookup(one_blocked) == 1


def test_default_table_blocks_nothing():
    """DEFAULT_TABLE applies to unknown tilesets — model learns from positional dynamics."""
    assert DEFAULT_TABLE.lookup(0x00) == 0
    assert DEFAULT_TABLE.lookup(0xFF) == 0


def test_code_to_name_returns_strings():
    assert code_to_name(0) == "walkable"
    assert code_to_name(1) == "blocked"
    assert code_to_name(255) == "unknown"


def test_collision_table_is_frozen():
    table = COLLISION_TABLES[OVERWORLD]
    # blocked is a frozenset; can't mutate it
    import pytest as _pt
    with _pt.raises((AttributeError, TypeError)):
        table.blocked.add(0xAB)  # type: ignore[attr-defined]
