"""Structured RAM state schema for Pokemon Red.

See ../../docs/architecture.md Section 4.1 for the field-by-field design.
The full state is ~660 bytes raw (Phase 1). Phase 2 will graduate to full WRAM.
"""
from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Sizes locked by the spec
EVENT_FLAGS_BYTES = 256
TILE_COLLISION_BYTES = 256       # 16x16 tiles around player
BAG_SLOTS = 20
PARTY_MAX = 6


class PartySlot(BaseModel):
    """One Pokemon in the player's party."""
    model_config = ConfigDict(frozen=True)

    species_id: int = Field(ge=0, le=255)
    level: int = Field(ge=1, le=100)
    hp_cur: int = Field(ge=0)
    hp_max: int = Field(ge=1)
    status: int = Field(ge=0, le=255)              # status condition byte
    moves: Tuple[int, int, int, int]               # 4 move IDs (0 = empty)

    @model_validator(mode="after")
    def _hp_cur_le_max(self) -> "PartySlot":
        if self.hp_cur > self.hp_max:
            raise ValueError(f"hp_cur ({self.hp_cur}) > hp_max ({self.hp_max})")
        return self

    @field_validator("moves")
    @classmethod
    def _moves_in_byte_range(cls, v: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        for m in v:
            if not 0 <= m <= 255:
                raise ValueError(f"move id {m} out of byte range")
        return v


class BagSlot(BaseModel):
    """One inventory slot."""
    model_config = ConfigDict(frozen=True)

    item_id: int = Field(ge=0, le=255)
    qty: int = Field(ge=0, le=99)


class BattleState(BaseModel):
    """Battle state. Zeroed when not in battle."""
    model_config = ConfigDict(frozen=True)

    in_battle: bool = False
    opp_species_id: int = Field(default=0, ge=0, le=255)
    opp_level: int = Field(default=0, ge=0, le=100)
    opp_hp: int = Field(default=0, ge=0)
    turn: int = Field(default=0, ge=0)


class GameState(BaseModel):
    """Full Phase-1 structured state extracted from a single PyBoy frame.

    Roughly 660 bytes raw — see spec Section 4.1 table.
    """
    model_config = ConfigDict(frozen=True)

    # Position
    map_id: int = Field(ge=0, le=255)
    x: int = Field(ge=0, le=255)
    y: int = Field(ge=0, le=255)

    # Party (variable length 0–6)
    party: Tuple[PartySlot, ...]

    # Bag (variable length 0–20)
    bag: Tuple[BagSlot, ...]

    # Progress
    badges: int = Field(ge=0, le=255)              # 8-bit mask
    event_flags: bytes                             # exactly EVENT_FLAGS_BYTES bytes
    money: int = Field(ge=0)
    time_played_frames: int = Field(ge=0)

    # Battle
    battle: BattleState

    # Local tile data
    tile_collision: bytes                          # exactly TILE_COLLISION_BYTES bytes

    # Mode / dialogue
    menu_flags: int = Field(ge=0)

    @field_validator("party")
    @classmethod
    def _party_size_ok(cls, v: Tuple[PartySlot, ...]) -> Tuple[PartySlot, ...]:
        if len(v) > PARTY_MAX:
            raise ValueError(f"party has {len(v)} slots; max is {PARTY_MAX}")
        return v

    @field_validator("bag")
    @classmethod
    def _bag_size_ok(cls, v: Tuple[BagSlot, ...]) -> Tuple[BagSlot, ...]:
        if len(v) > BAG_SLOTS:
            raise ValueError(f"bag has {len(v)} slots; max is {BAG_SLOTS}")
        return v

    @field_validator("event_flags")
    @classmethod
    def _event_flags_correct_length(cls, v: bytes) -> bytes:
        if len(v) != EVENT_FLAGS_BYTES:
            raise ValueError(f"event_flags must be {EVENT_FLAGS_BYTES} bytes, got {len(v)}")
        return v

    @field_validator("tile_collision")
    @classmethod
    def _tile_collision_correct_length(cls, v: bytes) -> bytes:
        if len(v) != TILE_COLLISION_BYTES:
            raise ValueError(f"tile_collision must be {TILE_COLLISION_BYTES} bytes, got {len(v)}")
        return v

    @property
    def party_size(self) -> int:
        return len(self.party)

    def has_badge(self, badge_index: int) -> bool:
        """badge_index 0..7 (Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth)."""
        if not 0 <= badge_index <= 7:
            raise ValueError(f"badge_index out of range: {badge_index}")
        return bool(self.badges & (1 << badge_index))
