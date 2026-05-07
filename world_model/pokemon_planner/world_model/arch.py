"""World-model architecture stubs (Phase 0).

Three networks from spec Section 4.2:
- RepresentationNet (h): obs -> latent state
- DynamicsNet (g): (latent, action) -> (next_latent, obs_pred, reward)
- PredictionNet (f): (latent, goal_emb) -> (policy_logits, value)

Phase 0 uses simple MLPs over a flat obs vector. Phase 1a will replace the
representation function with a transformer-based encoder over typed fields,
and the dynamics function will gain explicit chance-node outputs for
stochastic dynamics. Interfaces are stable across the upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class WorldModelConfig:
    obs_dim: int                  # flat obs vector length (Phase 0); typed fields in Phase 1a
    action_dim: int               # number of discrete actions
    latent_dim: int               # latent state dimensionality
    goal_emb_dim: int             # goal embedding dimensionality
    hidden_dim: int               # MLP hidden width
    num_predicate_types: int = 6  # catch/beat/reach/have_item/level/evolve
    num_entities: int = 256       # max distinct entity IDs (species, trainer, etc.)


def _mlp(*dims: int, dropout: float = 0.0) -> nn.Module:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RepresentationNet(nn.Module):
    """h(obs) -> latent state s."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = _mlp(config.obs_dim, config.hidden_dim, config.latent_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)


class DynamicsNet(nn.Module):
    """g(s, a) -> (s', obs_pred, r_hat)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.action_emb = nn.Embedding(config.action_dim, config.hidden_dim)
        self.trunk = _mlp(
            config.latent_dim + config.hidden_dim, config.hidden_dim, config.hidden_dim
        )
        self.s_next_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.obs_head = nn.Linear(config.hidden_dim, config.obs_dim)
        self.r_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a_emb = self.action_emb(a)
        h = self.trunk(torch.cat([s, a_emb], dim=-1))
        s_next = self.s_next_head(h)
        obs_pred = self.obs_head(h)
        r_pred = self.r_head(h).squeeze(-1)
        return s_next, obs_pred, r_pred


class PredictionNet(nn.Module):
    """f(s, goal_emb) -> (policy_logits, value)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.trunk = _mlp(
            config.latent_dim + config.goal_emb_dim, config.hidden_dim, config.hidden_dim
        )
        self.policy_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, s: Tensor, goal_emb: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(torch.cat([s, goal_emb], dim=-1))
        pi = self.policy_head(h)
        v = self.value_head(h).squeeze(-1)
        return pi, v


class WorldModel(nn.Module):
    """Wrapper exposing h, g, f as submodules."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.h = RepresentationNet(config)
        self.g = DynamicsNet(config)
        self.f = PredictionNet(config)

    def forward(self, obs: Tensor, action: Tensor, goal_emb: Tensor) -> dict[str, Tensor]:
        s = self.h(obs)
        s_next, obs_pred, r_pred = self.g(s, action)
        pi, v = self.f(s_next, goal_emb)
        return dict(s=s, s_next=s_next, obs_pred=obs_pred, r_pred=r_pred, pi=pi, v=v)
