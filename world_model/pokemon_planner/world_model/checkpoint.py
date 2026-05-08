"""Full-state checkpoint save / load for resumable training.

Per spec Section 6.4: model + optimizer + scheduler + AMP scaler + RNG state +
step counter + replay buffer position + W&B run ID + config.

After load_checkpoint(), the next training step is bit-equivalent (modulo CUDA
non-determinism) to what it would have been without interruption.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class CheckpointState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    scaler: Any | None
    step: int
    ema: Any | None
    replay_buffer_position: int
    wandb_run_id: str | None
    config: dict


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    """Save full training state to path (atomic write via tmp + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": state.model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict() if state.scheduler is not None else None,
        "scaler": state.scaler.state_dict() if state.scaler is not None else None,
        "step": state.step,
        "ema": state.ema.shadow_state_dict() if state.ema is not None else None,
        "replay_buffer_position": state.replay_buffer_position,
        "wandb_run_id": state.wandb_run_id,
        "config": state.config,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    ema: Any | None = None,
    map_location: str | None = None,
) -> CheckpointState:
    """Load full training state into the given model + optimizer + scheduler + scaler."""
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_shadow_state_dict(payload["ema"])

    rng = payload.get("rng_state")
    if rng:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])

    return CheckpointState(
        model=model,
        optimizer=optimizer if optimizer is not None else _DummyOpt(),
        scheduler=scheduler,
        scaler=scaler,
        step=int(payload.get("step", 0)),
        ema=ema,
        replay_buffer_position=int(payload.get("replay_buffer_position", 0)),
        wandb_run_id=payload.get("wandb_run_id"),
        config=payload.get("config", {}),
    )


class _DummyOpt:
    """Sentinel returned when caller didn't pass an optimizer to load_checkpoint."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, _: dict) -> None:
        pass
