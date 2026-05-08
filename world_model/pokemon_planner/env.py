"""PyBoy wrapper + state extractor.

Public API:
    PokeBoy(rom_path, save_state_path=None) — context-manager-friendly wrapper
    read_state(pyboy) -> GameState           — extract structured state from a PyBoy instance

Phase 1 reads ~660 bytes of curated WRAM. Phase 2 will graduate to full memory
snapshots; that lives in a separate module to avoid disturbing this one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyboy import PyBoy

from pokemon_planner import _ram_addresses as ram
from pokemon_planner.state import (
    BAG_SLOTS,
    BattleState,
    BagSlot,
    EVENT_FLAGS_BYTES,
    GameState,
    PartySlot,
    PARTY_MAX,
    TILE_COLLISION_BYTES,
)


# ---- Wrapper ----

class PokeBoy:
    """Thin context-manager wrapper around PyBoy 2.x.

    Mostly here to centralize boot + save-state load so callers don't repeat
    the dance. The underlying PyBoy instance is exposed as `.pyboy`.
    """

    def __init__(
        self,
        rom_path: str | Path,
        save_state_path: Optional[str | Path] = None,
        window: str = "null",   # "null" = headless; "SDL2" for visible
    ):
        rom_path = Path(rom_path).resolve()
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        self.pyboy = PyBoy(str(rom_path), window=window)
        # PyBoy 2.x boots automatically; tick once so first frame is rendered
        self.pyboy.tick()
        if save_state_path is not None:
            with open(save_state_path, "rb") as f:
                self.pyboy.load_state(f)

    def close(self) -> None:
        self.pyboy.stop()

    def __enter__(self) -> "PokeBoy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- Helpers ----

def _read_byte(pb: PyBoy, addr: int) -> int:
    return pb.memory[addr]


def _read_bytes(pb: PyBoy, addr: int, n: int) -> bytes:
    return bytes(pb.memory[addr + i] for i in range(n))


def _read_u16_be(pb: PyBoy, addr: int) -> int:
    """Big-endian 2-byte read (Pokemon Red stat layout)."""
    hi = pb.memory[addr]
    lo = pb.memory[addr + 1]
    return (hi << 8) | lo


def _read_bcd(pb: PyBoy, addr: int, n_bytes: int) -> int:
    """Read n_bytes of binary-coded decimal big-endian."""
    val = 0
    for i in range(n_bytes):
        b = pb.memory[addr + i]
        hi = (b >> 4) & 0x0F
        lo = b & 0x0F
        val = val * 100 + hi * 10 + lo
    return val


# ---- Per-section extractors ----

def _read_party(pb: PyBoy) -> tuple[PartySlot, ...]:
    n = min(_read_byte(pb, ram.PARTY_SIZE), PARTY_MAX)
    out: list[PartySlot] = []
    for i in range(n):
        base = ram.PARTY_STRUCTS_START + i * ram.PARTY_STRUCT_SIZE
        species = _read_byte(pb, base + ram.PARTY_OFFSET_SPECIES)
        if species == 0xFF:                # empty slot sentinel
            continue
        hp_cur = _read_u16_be(pb, base + ram.PARTY_OFFSET_HP_CUR)
        hp_max = _read_u16_be(pb, base + ram.PARTY_OFFSET_HP_MAX)
        # Defensive: clamp hp_cur to hp_max in case of mid-frame torn read
        if hp_max == 0:
            hp_max = 1
        if hp_cur > hp_max:
            hp_cur = hp_max
        level = max(1, min(100, _read_byte(pb, base + ram.PARTY_OFFSET_LEVEL)))
        status = _read_byte(pb, base + ram.PARTY_OFFSET_STATUS)
        moves = tuple(_read_byte(pb, base + ram.PARTY_OFFSET_MOVES + j) for j in range(4))
        out.append(PartySlot(
            species_id=species,
            level=level,
            hp_cur=hp_cur,
            hp_max=hp_max,
            status=status,
            moves=moves,  # type: ignore[arg-type]
        ))
    return tuple(out)


def _read_bag(pb: PyBoy) -> tuple[BagSlot, ...]:
    n = min(_read_byte(pb, ram.BAG_COUNT), BAG_SLOTS)
    out: list[BagSlot] = []
    for i in range(n):
        item_id = _read_byte(pb, ram.BAG_ITEMS_START + i * 2)
        qty = _read_byte(pb, ram.BAG_ITEMS_START + i * 2 + 1)
        if item_id == 0xFF:
            break
        out.append(BagSlot(item_id=item_id, qty=min(qty, 99)))
    return tuple(out)


def _read_event_flags(pb: PyBoy) -> bytes:
    """Curated 256-byte subset of the 312-byte flag region.

    Phase 1 keeps the first 256 of 312 — covers the early-mid game story
    flags. Phase 2 may include the full range or a different curated subset.
    """
    return _read_bytes(pb, ram.EVENT_FLAGS_START, EVENT_FLAGS_BYTES)


def _read_battle(pb: PyBoy) -> BattleState:
    in_battle = _read_byte(pb, ram.IS_IN_BATTLE) != 0
    if not in_battle:
        return BattleState(in_battle=False)
    return BattleState(
        in_battle=True,
        opp_species_id=_read_byte(pb, ram.BATTLE_OPP_SPECIES),
        opp_level=min(100, _read_byte(pb, ram.BATTLE_OPP_LEVEL)),
        opp_hp=_read_u16_be(pb, ram.BATTLE_OPP_HP),
        turn=min(255, _read_byte(pb, ram.BATTLE_TURN)),
    )


def _read_tile_id_at(pb: PyBoy, world_x: int, world_y: int) -> int | None:
    """Read the raw tile byte (0-255) at world tile coordinate (world_x, world_y).

    Pokemon Red stores the current map in the Game Boy background tilemap
    starting at BG position (0, 0).  The BG tilemap is 32x32 tiles; positions
    outside [0, 32) in either axis are off-map and return None.

    PyBoy's tile_identifier() returns a "global" tile index in [0, 512):
      - 256-511  ->  tiles from VRAM bank at 0x9000  (Pokemon Red tile IDs 0-127)
      - 128-255  ->  tiles from VRAM bank at 0x8800  (Pokemon Red tile IDs 128-255)
    Pokemon Red uses LCDC addressing mode 0 (0x8800-based), so converting back
    to the raw tile byte (0-255) stored in the BG map is:
        raw_tile_byte = (tile_identifier - 256) % 256
    """
    if not (0 <= world_x < 32 and 0 <= world_y < 32):
        return None
    try:
        identifier = pb.tilemap_background.tile_identifier(world_x, world_y)
    except (AttributeError, IndexError):
        return None
    # Convert PyBoy global tile identifier to raw Pokemon Red tile byte 0-255.
    return (identifier - 256) % 256


def _read_tile_collision(pb: PyBoy, player_x: int, player_y: int) -> bytes:
    """Extract a 16x16 grid of collision codes centered on the player.

    See spec Section 4.  Reads the current tileset ID from RAM, looks up the
    matching collision table from _tilesets.py, then classifies each tile.

    Returns 256 bytes, row-major: out[dy * 16 + dx] = code at world position
    (player_x + dx - 8, player_y + dy - 8), with dy/dx in [0, 16).
    Center cell (dy=8, dx=8, index=136) is the player's own tile.
    """
    from pokemon_planner import _tilesets

    tileset_id = pb.memory[ram.CURRENT_TILESET]
    table = _tilesets.COLLISION_TABLES.get(tileset_id, _tilesets.DEFAULT_TABLE)

    out = bytearray(TILE_COLLISION_BYTES)
    for dy in range(16):
        for dx in range(16):
            world_x = player_x + dx - 8
            world_y = player_y + dy - 8
            tile_id = _read_tile_id_at(pb, world_x, world_y)
            if tile_id is None:
                code = _tilesets.COLLISION_UNKNOWN
            else:
                code = table.lookup(tile_id)
            out[dy * 16 + dx] = code
    return bytes(out)


# ---- Public API ----

def read_state(pyboy: PyBoy) -> GameState:
    """Extract a structured GameState from a PyBoy instance at its current frame."""
    return GameState(
        map_id=_read_byte(pyboy, ram.MAP_ID),
        x=_read_byte(pyboy, ram.PLAYER_X),
        y=_read_byte(pyboy, ram.PLAYER_Y),
        party=_read_party(pyboy),
        bag=_read_bag(pyboy),
        badges=_read_byte(pyboy, ram.BADGES),
        event_flags=_read_event_flags(pyboy),
        money=_read_bcd(pyboy, ram.MONEY, 3),
        time_played_frames=_read_byte(pyboy, ram.TIME_PLAYED_FRAMES),
        battle=_read_battle(pyboy),
        tile_collision=_read_tile_collision(
            pyboy,
            _read_byte(pyboy, ram.PLAYER_X),
            _read_byte(pyboy, ram.PLAYER_Y),
        ),
        menu_flags=_read_byte(pyboy, ram.MENU_FLAGS),
    )
