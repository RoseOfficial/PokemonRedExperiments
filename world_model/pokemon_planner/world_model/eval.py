"""Eval-during-training — per-field accuracy on held-out trajectories.

Per spec Section 6.5. The Phase 1a DoD gate is checked here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from pokemon_planner.state import GameState
from pokemon_planner.world_model.arch import WorldModel
from pokemon_planner.world_model.losses import targets_from_states


@dataclass
class DODGate:
    thresholds: Mapping[str, float] = field(default_factory=lambda: {
        "acc/map_id": 0.80,
        "acc/x": 0.80,
        "acc/y": 0.80,
        "acc/party_species_slot_0": 0.80,
    })


def check_dod_gate(metrics: Mapping[str, float], gate: DODGate) -> bool:
    """Returns True if all gate thresholds are met."""
    return all(metrics.get(key, 0.0) >= threshold
               for key, threshold in gate.thresholds.items())


@torch.no_grad()
def run_eval(
    *,
    model: WorldModel,
    eval_states: list[GameState],
    eval_next_states: list[GameState],
    eval_actions: Tensor,
    eval_goal_embs: Tensor,
    batch_size: int = 64,
    max_batches: int = 20,
) -> dict[str, float]:
    """Per-field accuracy on held-out trajectories.

    Returns a dict of metrics suitable for W&B logging:
        acc/map_id, acc/x, acc/y, acc/party_species_slot_0,
        plus per-slot species/level accuracies.
    """
    was_training = model.training
    model.eval()

    metrics: dict[str, list[float]] = defaultdict(list)

    n = len(eval_states)
    n_batches = min(max_batches, max(1, (n + batch_size - 1) // batch_size))

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(n, start + batch_size)
        if start >= n:
            break
        states = eval_states[start:end]
        next_states = eval_next_states[start:end]
        actions = eval_actions[start:end]
        goal_emb = eval_goal_embs[start:end]
        targets = targets_from_states(next_states, model.config)

        out = model(states, actions, goal_emb)
        obs_pred = out["obs_pred"]

        # Move targets to model's device
        device = out["pi"].device
        targets = {k: v.to(device) if isinstance(v, Tensor) else v
                   for k, v in targets.items()}

        # Per-field accuracy (greedy argmax on logits)
        metrics["acc/map_id"].append(
            (obs_pred["map_id"].argmax(-1) == targets["map_id"]).float().mean().item()
        )
        metrics["acc/x"].append(
            (obs_pred["x"].argmax(-1) == targets["x"]).float().mean().item()
        )
        metrics["acc/y"].append(
            (obs_pred["y"].argmax(-1) == targets["y"]).float().mean().item()
        )
        metrics["acc/party_size"].append(
            (obs_pred["party_size"].argmax(-1) == targets["party_size"]).float().mean().item()
        )
        # Per-slot species and level
        for i, head_logits in enumerate(obs_pred["party_species"]):
            metrics[f"acc/party_species_slot_{i}"].append(
                (head_logits.argmax(-1) == targets["party_species"][:, i]).float().mean().item()
            )
        for i, head_logits in enumerate(obs_pred["party_level"]):
            metrics[f"acc/party_level_slot_{i}"].append(
                (head_logits.argmax(-1) == targets["party_level"][:, i]).float().mean().item()
            )

    if was_training:
        model.train()

    return {k: float(np.mean(v)) for k, v in metrics.items()}
