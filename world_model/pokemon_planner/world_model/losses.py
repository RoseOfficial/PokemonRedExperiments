"""Joint MuZero-style loss with placeholder value/policy targets.

Per spec Sections 6.1-6.2:
    L_total = 1.0·L_obs + 0.25·L_value + 1.0·L_policy + 0.5·L_reward + 0.1·L_consist

Phase 1a placeholder targets:
- Policy: behavioral cloning on demo actions (one-hot)
- Value: Monte-Carlo returns over PPO's shaped reward
- Reward: actual demo reward
- Obs: cross-entropy / BCE per field
- Consistency: latent at t+1 matches re-encoded next state
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from pokemon_planner.state import GameState
from pokemon_planner.world_model.arch import WorldModelConfig


@dataclass
class JointLossWeights:
    obs: float = 1.0
    value: float = 0.25
    policy: float = 1.0
    reward: float = 0.5
    consistency: float = 0.1


def targets_from_states(states: list[GameState], cfg: WorldModelConfig) -> dict[str, Tensor]:
    """Extract per-field target tensors from a list of next-state observations."""
    B = len(states)

    def collect(extractor) -> Tensor:
        return torch.tensor([extractor(s) for s in states], dtype=torch.long)

    map_id = collect(lambda s: s.map_id)
    x = collect(lambda s: s.x)
    y = collect(lambda s: s.y)
    party_size = collect(lambda s: s.party_size)

    species_per_slot = torch.zeros(B, cfg.num_party_slots, dtype=torch.long)
    level_per_slot = torch.zeros(B, cfg.num_party_slots, dtype=torch.long)
    for b, s in enumerate(states):
        for i, slot in enumerate(s.party):
            if i >= cfg.num_party_slots:
                break
            species_per_slot[b, i] = slot.species_id
            level_per_slot[b, i] = slot.level

    badges = torch.zeros(B, 8, dtype=torch.float32)
    for b, s in enumerate(states):
        for i in range(8):
            badges[b, i] = float((s.badges >> i) & 1)

    event_flags = torch.zeros(B, cfg.event_flags_bytes * 8, dtype=torch.float32)
    for b, s in enumerate(states):
        for byte_idx, byte_val in enumerate(s.event_flags):
            for bit_idx in range(8):
                event_flags[b, byte_idx * 8 + bit_idx] = float((byte_val >> bit_idx) & 1)

    money = collect(lambda s: min(int(s.money * cfg.num_money_buckets // 1_000_000),
                                   cfg.num_money_buckets - 1))
    battle_in = torch.tensor([float(s.battle.in_battle) for s in states], dtype=torch.float32)
    battle_species = collect(lambda s: s.battle.opp_species_id if s.battle.in_battle else 0)
    battle_level = collect(lambda s: s.battle.opp_level if s.battle.in_battle else 0)

    tile_collision = torch.zeros(B, 256, dtype=torch.long)
    for b, s in enumerate(states):
        for i in range(256):
            code = s.tile_collision[i]
            tile_collision[b, i] = 6 if code == 255 else min(code, 5)

    menu_flags = collect(lambda s: s.menu_flags)

    return {
        "map_id": map_id,
        "x": x,
        "y": y,
        "party_size": party_size,
        "party_species": species_per_slot,
        "party_level": level_per_slot,
        "badges": badges,
        "event_flags": event_flags,
        "money": money,
        "battle_in": battle_in,
        "battle_species": battle_species,
        "battle_level": battle_level,
        "tile_collision": tile_collision,
        "menu_flags": menu_flags,
    }


def compute_joint_loss(
    *,
    wm_out: dict,
    action_targets: Tensor,
    next_state_targets: dict[str, Tensor],
    reward_targets: Tensor,
    mc_return_targets: Tensor,
    prev_latent: Tensor,
    next_latent_target: Tensor,
    weights: JointLossWeights,
) -> tuple[Tensor, dict[str, float]]:
    """Compute joint loss for one (B, k=1) training step."""
    obs_pred = wm_out["obs_pred"]
    s_next = wm_out["s_next"]
    pi = wm_out["pi"]
    v = wm_out["v"]
    r_pred = wm_out["r_pred"]

    # Move targets to same device as predictions
    device = pi.device
    targets = {k: t.to(device) if isinstance(t, Tensor) else t
               for k, t in next_state_targets.items()}

    # ---- L_obs: per-field XE / BCE ----
    l_obs = (
        F.cross_entropy(obs_pred["map_id"], targets["map_id"])
        + F.cross_entropy(obs_pred["x"], targets["x"])
        + F.cross_entropy(obs_pred["y"], targets["y"])
        + F.cross_entropy(obs_pred["party_size"], targets["party_size"])
        + F.binary_cross_entropy_with_logits(obs_pred["badges"], targets["badges"])
        + F.binary_cross_entropy_with_logits(obs_pred["event_flags"], targets["event_flags"])
        + F.cross_entropy(obs_pred["money"], targets["money"])
        + F.binary_cross_entropy_with_logits(
            obs_pred["battle_in"].unsqueeze(-1) if obs_pred["battle_in"].dim() == 1
            else obs_pred["battle_in"],
            targets["battle_in"].unsqueeze(-1) if targets["battle_in"].dim() == 1
            else targets["battle_in"],
        )
        + F.cross_entropy(obs_pred["battle_species"], targets["battle_species"])
        + F.cross_entropy(obs_pred["battle_level"], targets["battle_level"])
        + F.cross_entropy(obs_pred["menu_flags"], targets["menu_flags"])
    )

    # Per-slot species and level
    for i in range(len(obs_pred["party_species"])):
        l_obs = l_obs + F.cross_entropy(
            obs_pred["party_species"][i],
            targets["party_species"][:, i],
        )
        l_obs = l_obs + F.cross_entropy(
            obs_pred["party_level"][i],
            targets["party_level"][:, i],
        )

    # Tile collision: (B, 256, 7) logits vs. (B, 256) class indices
    tc_logits = obs_pred["tile_collision"]
    tc_logits_flat = tc_logits.reshape(-1, tc_logits.shape[-1])
    tc_targets_flat = targets["tile_collision"].reshape(-1)
    l_obs = l_obs + F.cross_entropy(tc_logits_flat, tc_targets_flat)

    # ---- L_value: MSE on Monte-Carlo returns ----
    l_value = F.mse_loss(v, mc_return_targets.to(device))

    # ---- L_policy: cross-entropy on demo actions (BC) ----
    l_policy = F.cross_entropy(pi, action_targets.to(device))

    # ---- L_reward: MSE on shaped reward ----
    l_reward = F.mse_loss(r_pred, reward_targets.to(device))

    # ---- L_consistency: latent at t+1 matches re-encoded next state ----
    l_consist = F.mse_loss(s_next, next_latent_target.to(device))

    total = (
        weights.obs * l_obs
        + weights.value * l_value
        + weights.policy * l_policy
        + weights.reward * l_reward
        + weights.consistency * l_consist
    )

    components = {
        "obs": float(l_obs.detach()),
        "value": float(l_value.detach()),
        "policy": float(l_policy.detach()),
        "reward": float(l_reward.detach()),
        "consistency": float(l_consist.detach()),
        "total": float(total.detach()),
    }
    return total, components
