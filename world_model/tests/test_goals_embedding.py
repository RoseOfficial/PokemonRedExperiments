"""Tests for goal embedding."""
import pytest
import torch

from pokemon_planner.goals import catch, beat, reach, have_item, level, evolve
from pokemon_planner.goals.embedding import (
    DummyGoalEmbedding,
    GoalEmbedder,
    PREDICATE_TO_INDEX,
)


def test_predicate_index_covers_all_six():
    expected = {"catch", "beat", "reach", "have_item", "level", "evolve"}
    assert set(PREDICATE_TO_INDEX.keys()) == expected


def test_goal_embedder_construction():
    embedder = GoalEmbedder(num_predicates=6, num_entities=256, embed_dim=384)
    assert isinstance(embedder, torch.nn.Module)


def test_goal_embedder_atom_returns_correct_shape():
    embedder = GoalEmbedder(num_predicates=6, num_entities=256, embed_dim=384)
    g = catch("ODDISH")
    entity_lookup = {"ODDISH": 71}  # arbitrary index
    out = embedder(g, entity_lookup)
    assert out.shape == (384,)


def test_goal_embedder_batch_returns_correct_shape():
    embedder = GoalEmbedder(num_predicates=6, num_entities=256, embed_dim=384)
    goals = [catch("ODDISH"), reach("PEWTER_CITY"), beat("BROCK")]
    entity_lookup = {"ODDISH": 71, "PEWTER_CITY": 2, "BROCK": 0}
    out = embedder.batch(goals, entity_lookup)
    assert out.shape == (3, 384)


def test_goal_embedder_unknown_entity_uses_unknown_index():
    embedder = GoalEmbedder(num_predicates=6, num_entities=256, embed_dim=384)
    g = catch("MISSINGNO")  # not in lookup
    out = embedder(g, entity_lookup={})  # unknown — should use default
    assert out.shape == (384,)
    assert torch.isfinite(out).all()


def test_dummy_goal_embedding_is_learnable():
    """Phase 1a uses a single learned dummy goal vector — gradient must flow."""
    dummy = DummyGoalEmbedding(embed_dim=384)
    out = dummy()
    assert out.shape == (384,)
    assert out.requires_grad


def test_dummy_goal_embedding_batch():
    dummy = DummyGoalEmbedding(embed_dim=384)
    out = dummy.batch(batch_size=8)
    assert out.shape == (8, 384)
