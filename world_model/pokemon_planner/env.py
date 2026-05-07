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


def _read_tile_collision_stub(pb: PyBoy) -> bytes:
    """Placeholder — Phase 0 returns zeros.

    Real tile-collision extraction requires looking up the current tileset's
    collision table and indexing into it via the player's overworld tile map
    around (x, y). That logic is non-trivial and lives in a follow-up task
    (Phase 1a Task X). For Phase 0 the schema requires the right *length*,
    so we return 256 zeros.
    """
    return bytes(TILE_COLLISION_BYTES)


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
        tile_collision=_read_tile_collision_stub(pyboy),
        menu_flags=_read_byte(pyboy, ram.MENU_FLAGS),
    )
