"""Public constructors for goal atoms and combinators.

Re-exported from pokemon_planner.goals so users write `from pokemon_planner
import goals as g; g.catch(g.ODDISH)`.
"""
from __future__ import annotations

from typing import Callable, Iterable

from pokemon_planner.goals.dsl import Atom, And, Forall, Goal, Or, Then


# ---- Atoms ----

def catch(species: str) -> Atom:
    return Atom(predicate_type="catch", entity=species)


def beat(trainer: str) -> Atom:
    return Atom(predicate_type="beat", entity=trainer)


def reach(region: str) -> Atom:
    return Atom(predicate_type="reach", entity=region)


def have_item(item: str, qty: int = 1) -> Atom:
    return Atom(predicate_type="have_item", entity=item, params={"qty": qty})


def level(species: str, threshold: int) -> Atom:
    return Atom(
        predicate_type="level", entity=species, params={"threshold": threshold}
    )


def evolve(from_species: str, to_species: str) -> Atom:
    return Atom(
        predicate_type="evolve",
        entity=from_species,
        params={"from": from_species, "to": to_species},
    )


# ---- Combinators ----

def then(*children: Goal) -> Then:
    return Then(children=tuple(children))


def and_(*children: Goal) -> And:
    return And(children=tuple(children))


def or_(*children: Goal) -> Or:
    return Or(children=tuple(children))


def forall(items: Iterable[str], fn: Callable[[str], Goal]) -> Forall:
    return Forall(children=tuple(fn(x) for x in items))
