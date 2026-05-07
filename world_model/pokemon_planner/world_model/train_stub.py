"""Synthetic-data training stub for the world model.

Generates a tiny synthetic environment where dynamics are linear and rewards
are predictable, so the joint loss (state, value, policy, reward, consistency)
should reliably decrease. Used only in Phase 0 to verify wiring; Phase 1a
replaces this with real PyBoy-trajectory training.
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig


def _synthetic_batch(cfg: WorldModelConfig, batch_size: int, generator: torch.Generator):
    """A learnable synthetic transition.

    obs ~ N(0, I)
    action ~ Uniform(action_dim)
    goal_emb ~ N(0, I)
    obs_next = A @ obs + B @ a_onehot + small noise   (linear, deterministic-ish)
    reward = (W @ goal_emb) . obs   (linear in (goal, state))
    target_pi = uniform (placeholder; loss uses cross-entropy with logits anyway)
    target_v = reward (1-step return for the stub)
    """
    obs = torch.randn(batch_size, cfg.obs_dim, generator=generator)
    action = torch.randint(0, cfg.action_dim, (batch_size,), generator=generator)
    goal_emb = torch.randn(batch_size, cfg.goal_emb_dim, generator=generator)

    A = torch.randn(cfg.obs_dim, cfg.obs_dim, generator=generator) * 0.01
    B = torch.randn(cfg.obs_dim, cfg.action_dim, generator=generator) * 0.01
    a_onehot = torch.nn.functional.one_hot(action, num_classes=cfg.action_dim).float()
    obs_next = obs @ A.T + a_onehot @ B.T + 0.01 * torch.randn(
        batch_size, cfg.obs_dim, generator=generator
    )

    W = torch.randn(cfg.obs_dim, cfg.goal_emb_dim, generator=generator) * 0.01
    reward = (goal_emb @ W.T * obs).sum(dim=-1)

    target_v = reward.clone()
    target_pi = torch.full((batch_size, cfg.action_dim), 1.0 / cfg.action_dim)

    return dict(
        obs=obs, action=action, goal_emb=goal_emb,
        obs_next=obs_next, reward=reward,
        target_pi=target_pi, target_v=target_v,
    )


def _joint_loss(out: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
    # L_obs: predicted next observation vs. actual
    l_obs = torch.nn.functional.mse_loss(out["obs_pred"], batch["obs_next"])
    # L_value: predicted V vs. target return
    l_value = torch.nn.functional.mse_loss(out["v"], batch["target_v"])
    # L_policy: predicted logits vs. target distribution (KL via cross-entropy)
    log_pi = torch.nn.functional.log_softmax(out["pi"], dim=-1)
    l_policy = -(batch["target_pi"] * log_pi).sum(dim=-1).mean()
    # L_reward: predicted reward vs. actual
    l_reward = torch.nn.functional.mse_loss(out["r_pred"], batch["reward"])
    # L_consist: latent dynamics should produce a usable next state (dummy here —
    # encourages dynamics output to have similar magnitude to encoder output)
    l_consist = torch.nn.functional.mse_loss(out["s_next"], out["s"].detach())

    # Weights from spec Section 4.4 (default-ish; tuning in Phase 1a)
    return 1.0 * l_obs + 0.25 * l_value + 1.0 * l_policy + 0.5 * l_reward + 0.1 * l_consist


def train_steps(
    wm: WorldModel,
    cfg: WorldModelConfig,
    n_steps: int,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> List[float]:
    """Train n_steps on synthetic data; return list of per-step loss values."""
    gen = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(wm.parameters(), lr=lr)
    losses: List[float] = []
    wm.train()
    for _ in range(n_steps):
        batch = _synthetic_batch(cfg, batch_size, gen)
        out = wm(batch["obs"], batch["action"], batch["goal_emb"])
        loss = _joint_loss(out, batch)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        optim.step()
        losses.append(loss.item())
    return losses
