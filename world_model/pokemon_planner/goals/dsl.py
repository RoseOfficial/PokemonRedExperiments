"""Goal DSL — Goal class hierarchy + predicate evaluation.

See spec Section 2. Phase 0 implements: 6 atoms (catch, beat, reach, have_item,
level, evolve) and 4 combinators (then, and_, or_, forall). Phase 1c adds
named shortcuts (catch_all, beat_champion).

Predicates evaluate against a GameState. Atoms accept lookups (species_id_lookup,
map_id_lookup, etc.) so the predicate doesn't need direct KB access — that
binding happens at the planner layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pokemon_planner.state import GameState


# ---- Types ----

class Goal:
    """Base class for all goals. Predicate evaluation is per-subclass."""

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Goal):
    predicate_type: str          # "catch" | "beat" | "reach" | "have_item" | "level" | "evolve"
    entity: str = ""             # primary entity name (species, trainer, region, item)
    params: Mapping[str, Any] = field(default_factory=dict)

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return _evaluate_atom(self, state, lookups)

    def __repr__(self) -> str:
        if self.params:
            return f"{self.predicate_type}({self.entity}, {dict(self.params)})"
        return f"{self.predicate_type}({self.entity})"


@dataclass(frozen=True)
class Then(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        # All must hold (sequencing matters at decomposition, not at predicate-eval)
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "then(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class And(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "and_(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Or(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return any(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "or_(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Forall(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "forall([" + ", ".join(repr(c) for c in self.children) + "])"


# ---- Atom evaluation ----

def _evaluate_atom(atom: Atom, state: GameState, lookups: Mapping[str, Any]) -> bool:
    pt = atom.predicate_type
    if pt == "catch":
        sid = _lookup(lookups, "species_id_lookup", atom.entity)
        return any(slot.species_id == sid for slot in state.party)
    if pt == "reach":
        mid = _lookup(lookups, "map_id_lookup", atom.entity)
        return state.map_id == mid
    if pt == "have_item":
        iid = _lookup(lookups, "item_id_lookup", atom.entity)
        qty_needed = atom.params.get("qty", 1)
        for slot in state.bag:
            if slot.item_id == iid and slot.qty >= qty_needed:
                return True
        return False
    if pt == "level":
        sid = _lookup(lookups, "species_id_lookup", atom.entity)
        threshold = atom.params["threshold"]
        return any(s.species_id == sid and s.level >= threshold for s in state.party)
    if pt == "beat":
        # Phase 0: 'beat' resolves via an event-flag bit lookup if provided.
        # If no lookup is available the predicate returns False (planner must
        # supply it). This will be wired up properly in Phase 1c.
        flag_lookup = lookups.get("trainer_flag_lookup")
        if flag_lookup is None or atom.entity not in flag_lookup:
            return False
        flag_bit_index = flag_lookup[atom.entity]
        byte_index, bit_index = divmod(flag_bit_index, 8)
        if byte_index >= len(state.event_flags):
            return False
        return bool(state.event_flags[byte_index] & (1 << bit_index))
    if pt == "evolve":
        sid_to = _lookup(lookups, "species_id_lookup", atom.params["to"])
        return any(slot.species_id == sid_to for slot in state.party)
    raise ValueError(f"Unknown predicate type: {pt}")


def _lookup(lookups: Mapping[str, Any], key: str, name: str) -> int:
    table = lookups.get(key)
    if table is None:
        raise KeyError(f"required lookup '{key}' not provided for entity '{name}'")
    if name not in table:
        raise KeyError(f"'{name}' not found in {key}")
    return table[name]
