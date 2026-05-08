"""Per-tileset collision tables for Pokemon Red.

Sourced from pret/pokered/data/tilesets/*_collision.asm. Tile IDs encoded
here are tile IDs WITHIN A TILESET (not globally unique). Each table lists
which tile IDs are not walkable, plus separate sets for ledges and warps.

Tileset constants from pret/pokered/constants/tileset_constants.asm:
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- Tileset IDs ----
# (subset; full list of 24 in pret/pokered/constants/tileset_constants.asm)
OVERWORLD = 0
REDS_HOUSE_1 = 1
MART = 2
FOREST = 3
REDS_HOUSE_2 = 4
DOJO = 5
POKECENTER = 6
GYM = 7
HOUSE = 8
FOREST_GATE = 9
MUSEUM = 10
UNDERGROUND = 11
GATE = 12
SHIP = 13
SHIP_PORT = 14
CEMETERY = 15
INTERIOR = 16
CAVERN = 17
LOBBY = 18
MANSION = 19
LAB = 20
CLUB = 21
FACILITY = 22
PLATEAU = 23


# ---- Collision codes ----
COLLISION_WALKABLE = 0
COLLISION_BLOCKED = 1
COLLISION_LEDGE_DOWN = 2
COLLISION_LEDGE_LEFT = 3
COLLISION_LEDGE_RIGHT = 4
COLLISION_WARP = 5
COLLISION_UNKNOWN = 255


def code_to_name(code: int) -> str:
    return {
        COLLISION_WALKABLE: "walkable",
        COLLISION_BLOCKED: "blocked",
        COLLISION_LEDGE_DOWN: "ledge_down",
        COLLISION_LEDGE_LEFT: "ledge_left",
        COLLISION_LEDGE_RIGHT: "ledge_right",
        COLLISION_WARP: "warp",
        COLLISION_UNKNOWN: "unknown",
    }.get(code, f"unknown_code_{code}")


# ---- Collision table ----

@dataclass(frozen=True)
class CollisionTable:
    blocked: frozenset[int]
    ledge_down: frozenset[int] = field(default_factory=frozenset)
    ledge_left: frozenset[int] = field(default_factory=frozenset)
    ledge_right: frozenset[int] = field(default_factory=frozenset)
    warp: frozenset[int] = field(default_factory=frozenset)

    def lookup(self, tile_id: int) -> int:
        if tile_id in self.warp:
            return COLLISION_WARP
        if tile_id in self.ledge_down:
            return COLLISION_LEDGE_DOWN
        if tile_id in self.ledge_left:
            return COLLISION_LEDGE_LEFT
        if tile_id in self.ledge_right:
            return COLLISION_LEDGE_RIGHT
        if tile_id in self.blocked:
            return COLLISION_BLOCKED
        return COLLISION_WALKABLE


# ---- Per-tileset tables (Phase 1a: 8 most-encountered) ----
# Tile IDs from pret/pokered ASM. Manual verification step (in plan Task 4)
# confirms these against PyBoy screenshots before going to production.

COLLISION_TABLES: dict[int, CollisionTable] = {
    OVERWORLD: CollisionTable(
        blocked=frozenset({
            0x0F, 0x10, 0x1A, 0x17, 0x18, 0x19, 0x1C, 0x1E,
            0x20, 0x2E, 0x30, 0x52, 0x54, 0x5B, 0x60, 0x62,
            0x64, 0x67, 0x68, 0x6A, 0x6C, 0x71, 0x73, 0x75,
            0x77, 0x7C, 0x7D, 0x7E,
        }),
        ledge_down=frozenset({0x37, 0x38, 0x39}),
        ledge_left=frozenset({0x2C, 0x2D}),
        ledge_right=frozenset({0x36}),
        warp=frozenset({0x1B, 0x3A}),
    ),
    HOUSE: CollisionTable(
        blocked=frozenset({0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x0A,
                           0x0B, 0x0C, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x16,
                           0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1F, 0x21, 0x22,
                           0x23, 0x24, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D}),
        warp=frozenset({0x1E}),
    ),
    MART: CollisionTable(
        blocked=frozenset({0x11, 0x15, 0x1F, 0x3B, 0x3C, 0x3D, 0x3E, 0x4F,
                           0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E,
                           0x5F, 0x77}),
        warp=frozenset({0x1E}),
    ),
    POKECENTER: CollisionTable(
        blocked=frozenset({0x11, 0x15, 0x1F, 0x3B, 0x3C, 0x3D, 0x3E, 0x4F,
                           0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E,
                           0x5F, 0x77}),
        warp=frozenset({0x1E}),
    ),
    FOREST: CollisionTable(
        blocked=frozenset({0x14, 0x15, 0x1A, 0x1C, 0x37, 0x38, 0x3D, 0x50,
                           0x52, 0x53, 0x55, 0x56, 0x60, 0x62}),
        warp=frozenset({0x16}),
    ),
    CAVERN: CollisionTable(
        blocked=frozenset({0x05, 0x14, 0x18, 0x19, 0x1F, 0x20, 0x21, 0x22,
                           0x23, 0x24, 0x25, 0x26, 0x27, 0x29, 0x2D, 0x2E,
                           0x2F, 0x30, 0x31, 0x33, 0x34, 0x36, 0x38, 0x39,
                           0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F}),
        warp=frozenset({0x18}),
    ),
    GYM: CollisionTable(
        blocked=frozenset({0x11, 0x15, 0x1F, 0x3B, 0x3C, 0x3D, 0x3E, 0x4F,
                           0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E,
                           0x5F, 0x77}),
        warp=frozenset({0x1E}),
    ),
    GATE: CollisionTable(
        blocked=frozenset({0x11, 0x15, 0x1F, 0x3B, 0x3C, 0x3D, 0x3E, 0x4F,
                           0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E,
                           0x5F, 0x77}),
        warp=frozenset({0x1E}),
    ),
}

# Default table for unknown tilesets — nothing blocked, model learns from
# positional dynamics. Strictly suboptimal but only matters for Phase 1c+
# tilesets we haven't filled yet.
DEFAULT_TABLE = CollisionTable(blocked=frozenset())
