"""Training loop for Phase 1a — joint loss with placeholder targets, fp16-capable.

Public API:
    one_training_step()     — single forward + backward + optim step (used by tests)
    run_training_loop()     — full multi-step training with periodic eval, checkpointing,
                              W&B logging, and DoD gate checks (used by scripts/train.py)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from pokemon_planner.data.replay import Batch, ReplayBuffer
from pokemon_planner.goals.embedding import DummyGoalEmbedding
from pokemon_planner.world_model.arch import WorldModel
from pokemon_planner.world_model.checkpoint import CheckpointState, save_checkpoint
from pokemon_planner.world_model.losses import (
    JointLossWeights,
    compute_joint_loss,
    targets_from_states,
)


@dataclass
class TrainingConfig:
    batch_size: int = 32
    k_unroll: int = 5
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 500_000
    fp16: bool = True
    gradient_checkpointing: bool = True
    loss_weights: JointLossWeights = field(default_factory=JointLossWeights)


def one_training_step(
    *,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: TrainingConfig,
    step: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    goal_embedder: Optional[DummyGoalEmbedding] = None,
) -> dict[str, float]:
    """Run one optimizer step. Returns loss component dict."""
    model.train()

    device = next(model.parameters()).device

    if goal_embedder is None:
        goal_embedder = DummyGoalEmbedding(embed_dim=model.config.goal_emb_dim).to(device)

    batch: Batch = replay.sample_batch(config.batch_size, config.k_unroll)

    use_amp = config.fp16 and torch.cuda.is_available()

    optimizer.zero_grad(set_to_none=True)

    total_loss = torch.tensor(0.0, device=device)
    aggregate_components: dict[str, float] = {
        "obs": 0.0, "value": 0.0, "policy": 0.0,
        "reward": 0.0, "consistency": 0.0, "total": 0.0,
    }

    with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
        for t in range(config.k_unroll):
            states_t = [w[t] for w in batch.states]
            next_states_t = [w[t + 1] for w in batch.states]
            actions_t = torch.tensor(
                [w[t] for w in batch.actions], dtype=torch.long, device=device,
            )
            rewards_t = torch.tensor(
                [w[t] for w in batch.rewards], dtype=torch.float32, device=device,
            )
            mc_t = torch.tensor(
                [w[t] for w in batch.mc_returns], dtype=torch.float32, device=device,
            )
            goal_emb = goal_embedder.batch(config.batch_size)

            out = model(states_t, actions_t, goal_emb)
            targets = targets_from_states(next_states_t, model.config)
            targets = {
                k: v.to(device) if isinstance(v, Tensor) else v
                for k, v in targets.items()
            }

            with torch.no_grad():
                next_latent_target = model.h(next_states_t).detach()

            loss, components = compute_joint_loss(
                wm_out=out,
                action_targets=actions_t,
                next_state_targets=targets,
                reward_targets=rewards_t,
                mc_return_targets=mc_t,
                next_latent_target=next_latent_target,
                weights=config.loss_weights,
            )
            total_loss = total_loss + loss
            for key, value in components.items():
                aggregate_components[key] += float(value)

    if scaler is not None and use_amp:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

    return {k: v / config.k_unroll for k, v in aggregate_components.items()}


def run_training_loop(
    *,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    eval_replay: ReplayBuffer,
    config: TrainingConfig,
    checkpoint_dir: Path,
    save_every: int = 5000,
    eval_every: int = 2000,
    log_every: int = 100,
    wandb_logger=None,
    start_step: int = 0,
) -> None:
    """Multi-step training loop with checkpointing + periodic eval + W&B logging.

    Used by scripts/train.py. Tests use one_training_step() directly to avoid
    the long-running loop.
    """
    from pokemon_planner.world_model.eval import DODGate, check_dod_gate, run_eval

    scaler = torch.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    device = next(model.parameters()).device
    goal_embedder = DummyGoalEmbedding(embed_dim=model.config.goal_emb_dim).to(device)
    gate = DODGate()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Starting at step {start_step}/{config.total_steps}")

    for step in range(start_step, config.total_steps):
        components = one_training_step(
            model=model, optimizer=optimizer, replay=replay,
            config=config, step=step, scaler=scaler, goal_embedder=goal_embedder,
        )

        if step % log_every == 0:
            print(
                f"[train] step={step} loss={components['total']:.4f} "
                f"(obs={components['obs']:.3f} policy={components['policy']:.3f} "
                f"reward={components['reward']:.3f})"
            )
            if wandb_logger is not None:
                wandb_logger.log({f"loss/{k}": v for k, v in components.items()}, step=step)

        if step > 0 and step % eval_every == 0:
            eval_skipped = False
            try:
                eval_batch = eval_replay.sample_batch(config.batch_size * 2, k_unroll=1)
            except RuntimeError:
                print(f"[train] step={step} skipping eval (eval buffer too small)")
                eval_skipped = True

            if not eval_skipped:
                eval_states = [w[0] for w in eval_batch.states]
                eval_next_states = [w[1] for w in eval_batch.states]
                eval_actions = torch.tensor(
                    [w[0] for w in eval_batch.actions], dtype=torch.long, device=device,
                )
                eval_goal_emb = goal_embedder.batch(len(eval_states)).detach()

                metrics = run_eval(
                    model=model,
                    eval_states=eval_states, eval_next_states=eval_next_states,
                    eval_actions=eval_actions, eval_goal_embs=eval_goal_emb,
                    batch_size=config.batch_size, max_batches=20,
                )
                passes_gate = check_dod_gate(metrics, gate)
                print(
                    f"[train] step={step} eval: "
                    f"map={metrics.get('acc/map_id', 0):.3f} "
                    f"x={metrics.get('acc/x', 0):.3f} "
                    f"y={metrics.get('acc/y', 0):.3f} "
                    f"species={metrics.get('acc/party_species_slot_0', 0):.3f} "
                    f"DoD_gate={'PASS' if passes_gate else 'fail'}"
                )
                if wandb_logger is not None:
                    wandb_logger.log(
                        {**metrics, "eval/passes_doD_gate": int(passes_gate)},
                        step=step,
                    )

        if step > 0 and step % save_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
            save_checkpoint(ckpt_path, CheckpointState(
                model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
                step=step, ema=None,
                replay_buffer_position=replay.size,
                wandb_run_id=wandb_logger.run_id if wandb_logger else None,
                config={},
            ))
            latest = checkpoint_dir / "latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            try:
                latest.symlink_to(ckpt_path.name)
            except OSError:
                # Windows without admin can't symlink; just copy
                import shutil
                shutil.copy(ckpt_path, latest)
            print(f"[train] saved checkpoint to {ckpt_path}")

    print(f"[train] training completed at step {config.total_steps}")
