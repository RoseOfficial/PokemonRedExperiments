"""Goal embedding — compile goal Atoms to torch tensors for f-head conditioning.

Phase 1a uses DummyGoalEmbedding (single learned vector, no real conditioning).
Phase 1b's MCTS will sample real goals and use GoalEmbedder against the KB.

The architecture is wired up to support both — the world-model's f-head accepts
a (B, 384) goal embedding tensor regardless of source.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from pokemon_planner.goals.dsl import Atom


# Index map for predicate types. Order matches the 6 predicate constructors
# in pokemon_planner.goals.atoms.
PREDICATE_TO_INDEX: dict[str, int] = {
    "catch": 0,
    "beat": 1,
    "reach": 2,
    "have_item": 3,
    "level": 4,
    "evolve": 5,
}

UNKNOWN_ENTITY_INDEX = 0   # reserved for entities not in lookup


class GoalEmbedder(nn.Module):
    """Embeds an Atom into a fixed-size vector via predicate + entity embeddings.

    Concrete: catch(ODDISH) → concat(predicate_emb["catch"], entity_emb[ODDISH_idx])

    See spec Section 5.5.
    """

    def __init__(self, num_predicates: int = 6, num_entities: int = 256,
                 embed_dim: int = 384):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even (split between predicate + entity); got {embed_dim}"
            )
        half = embed_dim // 2
        self.predicate_emb = nn.Embedding(num_predicates, half)
        self.entity_emb = nn.Embedding(num_entities, half)
        self.embed_dim = embed_dim

    def forward(self, goal: Atom, entity_lookup: Mapping[str, int]) -> Tensor:
        """Embed a single Atom. Returns (embed_dim,)."""
        pred_idx = PREDICATE_TO_INDEX[goal.predicate_type]   # KeyError on unknown is desired
        ent_idx = entity_lookup.get(goal.entity, UNKNOWN_ENTITY_INDEX)
        device = self.predicate_emb.weight.device
        pred_e = self.predicate_emb(torch.tensor(pred_idx, device=device))
        ent_e = self.entity_emb(torch.tensor(ent_idx, device=device))
        return torch.cat([pred_e, ent_e], dim=-1)

    def batch(self, goals: list[Atom], entity_lookup: Mapping[str, int]) -> Tensor:
        """Embed a list of Atoms. Returns (B, embed_dim)."""
        return torch.stack([self.forward(g, entity_lookup) for g in goals], dim=0)


class DummyGoalEmbedding(nn.Module):
    """Phase 1a placeholder — single learned 384d vector used for all training samples.

    The PPO bootstrap data has no goal labels (PPO optimized v2's shaped reward,
    not arbitrary goals). For Phase 1a we condition the f-head on a single
    learned vector — the head learns "imitate PPO under this dummy goal."
    Phase 1b retrains f from scratch with real goal-conditioned MCTS targets.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.vec = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.embed_dim = embed_dim

    def forward(self) -> Tensor:
        """Returns the dummy embedding, shape (embed_dim,)."""
        return self.vec

    def batch(self, batch_size: int) -> Tensor:
        """Returns the dummy embedding repeated for a batch, shape (batch_size, embed_dim)."""
        return self.vec.unsqueeze(0).expand(batch_size, -1).contiguous()
