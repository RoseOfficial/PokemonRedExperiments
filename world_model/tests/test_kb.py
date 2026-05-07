"""Tests for the knowledge base loader."""
import pytest

from pokemon_planner.kb import KnowledgeBase, load_kb


def test_load_kb():
    kb = load_kb()
    assert isinstance(kb, KnowledgeBase)


def test_species_count_at_least_30():
    kb = load_kb()
    assert len(kb.species) >= 30


def test_oddish_in_species():
    kb = load_kb()
    oddish = kb.get_species("ODDISH")
    assert oddish.species_id == 0x47  # canonical Pokemon Red species ID for Oddish
    assert oddish.name == "ODDISH"


def test_charmander_starter_metadata():
    kb = load_kb()
    char = kb.get_species("CHARMANDER")
    assert char.species_id == 0xB0
    assert "starter" in char.tags
    assert char.encounter_method == "gift"


def test_ekans_red_exclusive():
    """Ekans is Red-exclusive (Sandshrew is the Blue counterpart)."""
    kb = load_kb()
    ekans = kb.get_species("EKANS")
    assert "red_exclusive" in ekans.tags


def test_unknown_species_raises():
    kb = load_kb()
    with pytest.raises(KeyError):
        kb.get_species("MISSINGNO")


def test_kb_is_frozen_after_load():
    """The loaded KB shouldn't be mutable from the outside."""
    kb = load_kb()
    with pytest.raises(Exception):
        kb.species["BULBASAUR"].name = "Tomato"  # type: ignore[misc]
