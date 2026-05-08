"""Trajectory and TrajectoryStep dataclasses + msgpack state serialization.

A Trajectory is a sequence of TrajectoryStep records covering one episode,
with metadata about the source PPO checkpoint and seed. Trajectories are
serialized to Parquet for on-disk storage (Task 2 / replay.py).

State serialization uses msgpack for compactness — ~700 bytes per GameState
vs. ~3KB JSON. See spec Section 3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import msgpack

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


@dataclass(frozen=True)
class TrajectoryStep:
    state: GameState
    action: int
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[TrajectoryStep, ...]
    source: str            # e.g. "v2_1310720"
    episode_id: int
    seed: int

    @property
    def length(self) -> int:
        return len(self.steps)

    def mc_returns(self, gamma: float = 0.997) -> list[float]:
        """Discounted Monte-Carlo returns from each step to episode end."""
        rewards = [s.reward for s in self.steps]
        out = [0.0] * len(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            out[t] = running
        return out


# ---- State serialization ----

def serialize_state(state: GameState) -> bytes:
    """Pack a GameState into compact msgpack bytes."""
    payload = {
        "map_id": state.map_id,
        "x": state.x,
        "y": state.y,
        "party": [
            {
                "species_id": s.species_id, "level": s.level,
                "hp_cur": s.hp_cur, "hp_max": s.hp_max,
                "status": s.status, "moves": list(s.moves),
            }
            for s in state.party
        ],
        "bag": [{"item_id": b.item_id, "qty": b.qty} for b in state.bag],
        "badges": state.badges,
        "event_flags": state.event_flags,
        "money": state.money,
        "time_played_frames": state.time_played_frames,
        "battle": {
            "in_battle": state.battle.in_battle,
            "opp_species_id": state.battle.opp_species_id,
            "opp_level": state.battle.opp_level,
            "opp_hp": state.battle.opp_hp,
            "turn": state.battle.turn,
        },
        "tile_collision": state.tile_collision,
        "menu_flags": state.menu_flags,
    }
    return msgpack.packb(payload, use_bin_type=True)


def deserialize_state(blob: bytes) -> GameState:
    """Unpack a msgpack blob into a GameState."""
    p = msgpack.unpackb(blob, raw=False)
    party = tuple(
        PartySlot(
            species_id=s["species_id"], level=s["level"],
            hp_cur=s["hp_cur"], hp_max=s["hp_max"],
            status=s["status"], moves=tuple(s["moves"]),
        )
        for s in p["party"]
    )
    bag = tuple(BagSlot(item_id=b["item_id"], qty=b["qty"]) for b in p["bag"])
    battle = BattleState(
        in_battle=p["battle"]["in_battle"],
        opp_species_id=p["battle"]["opp_species_id"],
        opp_level=p["battle"]["opp_level"],
        opp_hp=p["battle"]["opp_hp"],
        turn=p["battle"]["turn"],
    )
    return GameState(
        map_id=p["map_id"], x=p["x"], y=p["y"],
        party=party, bag=bag,
        badges=p["badges"],
        event_flags=bytes(p["event_flags"]),
        money=p["money"],
        time_played_frames=p["time_played_frames"],
        battle=battle,
        tile_collision=bytes(p["tile_collision"]),
        menu_flags=p["menu_flags"],
    )
