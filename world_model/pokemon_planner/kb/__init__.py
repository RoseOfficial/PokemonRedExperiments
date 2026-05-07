"""Knowledge base — curated game data, not learned.

Phase 0 ships with 30 species + stub schemas for trainers/items/regions.
Subsequent phases expand the data; the loader contract stays stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel, ConfigDict


KB_DIR = Path(__file__).parent


class Species(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str                              # uppercase identifier, e.g. "ODDISH"
    species_id: int                        # canonical Pokemon Red byte ID
    encounter_method: str                  # "wild" | "gift" | "trade" | "event"
    encounter_zones: tuple[str, ...] = ()  # region names (matches regions.yaml entries)
    required_badges: tuple[str, ...] = ()
    requires_item: tuple[str, ...] = ()
    evolution_from: str | None = None
    evolution_method: str | None = None    # "level:36" | "stone:LEAF" | "trade" | None
    tags: tuple[str, ...] = ()             # free-form: "starter", "red_exclusive", "legendary", etc.


class Trainer(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    location: str
    party: tuple[str, ...] = ()


class Item(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    item_id: int
    sources: tuple[str, ...] = ()


class Region(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    map_id: int
    blockers: tuple[str, ...] = ()         # e.g. "snorlax", "cut_tree", "surf"


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    species: Mapping[str, Species]
    trainers: Mapping[str, Trainer]
    items: Mapping[str, Item]
    regions: Mapping[str, Region]

    def get_species(self, name: str) -> Species:
        if name not in self.species:
            raise KeyError(f"Unknown species: {name}")
        return self.species[name]


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f)
    return data or []


def load_kb(kb_dir: Path | None = None) -> KnowledgeBase:
    """Load all KB YAML files and construct a frozen KnowledgeBase."""
    d = kb_dir if kb_dir is not None else KB_DIR
    species = {x["name"]: Species(**x) for x in _load_yaml(d / "species.yaml")}
    trainers = {x["name"]: Trainer(**x) for x in _load_yaml(d / "trainers.yaml")}
    items = {x["name"]: Item(**x) for x in _load_yaml(d / "items.yaml")}
    regions = {x["name"]: Region(**x) for x in _load_yaml(d / "regions.yaml")}
    return KnowledgeBase(
        species=species, trainers=trainers, items=items, regions=regions
    )
