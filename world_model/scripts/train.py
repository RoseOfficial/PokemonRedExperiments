"""CLI entry point for Phase 1a training.

Usage:
    python scripts/train.py [--config configs/phase_1a.yaml] [--resume <checkpoint.pt>]

Reads config YAML, builds model + optimizer + replay buffer + W&B logger,
optionally resumes from a checkpoint, then calls run_training_loop().
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from pokemon_planner.data.replay import ReplayBuffer
from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.checkpoint import load_checkpoint
from pokemon_planner.world_model.losses import JointLossWeights
from pokemon_planner.world_model.train import TrainingConfig, run_training_loop
from pokemon_planner.world_model.wandb_logger import WandbLogger


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/phase_1a.yaml"))
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("runs/phase1a_wm"))
    p.add_argument("--replay-root", type=Path, default=Path("data/replay_buffer"))
    p.add_argument("--eval-replay-root", type=Path, default=Path("data/replay_buffer/eval"))
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_config(model_cfg: dict) -> WorldModelConfig:
    return WorldModelConfig(
        action_dim=model_cfg["action_dim"],
        latent_dim=model_cfg["latent_dim"],
        d_model=model_cfg["d_model"],
        goal_emb_dim=model_cfg["goal_emb_dim"],
        encoder_layers=model_cfg["encoder_layers"],
        encoder_heads=model_cfg["encoder_heads"],
        dynamics_layers=model_cfg["dynamics_layers"],
        dynamics_heads=model_cfg["dynamics_heads"],
        prediction_layers=model_cfg["prediction_layers"],
        prediction_heads=model_cfg["prediction_heads"],
        ffn_expansion=model_cfg["ffn_expansion"],
        dropout=model_cfg["dropout"],
    )


def build_train_config(train_cfg: dict) -> TrainingConfig:
    weights = JointLossWeights(**train_cfg["loss_weights"])
    return TrainingConfig(
        batch_size=train_cfg["batch_size"],
        k_unroll=train_cfg["k_unroll"],
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        grad_clip_norm=float(train_cfg["grad_clip_norm"]),
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["total_steps"],
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        loss_weights=weights,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    print(f"[train.py] Loading config from {args.config}")
    print(f"[train.py] Replay buffer root: {args.replay_root}")

    if not args.replay_root.exists():
        print(f"[train.py] FATAL: Replay buffer not found at {args.replay_root}")
        print("[train.py] Run scripts/bootstrap_demos.py first.")
        return 1

    model_cfg = build_model_config(cfg["model"])
    model = WorldModel(model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[train.py] Model on {device}, {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    start_step = 0
    wandb_run_id = None
    if args.resume is not None and args.resume.exists():
        print(f"[train.py] Resuming from {args.resume}")
        loaded = load_checkpoint(
            args.resume, model=model, optimizer=optimizer,
            scheduler=None, scaler=None,
        )
        start_step = loaded.step
        wandb_run_id = loaded.wandb_run_id

    replay = ReplayBuffer(root=args.replay_root)
    eval_replay = ReplayBuffer(root=args.eval_replay_root)
    print(f"[train.py] Replay: {replay.size} steps, eval: {eval_replay.size} steps")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg['wandb']['run_name_prefix']}-{timestamp}"
    wandb_logger = WandbLogger(
        project=cfg["wandb"]["project"],
        run_name=run_name,
        config=cfg,
        run_id=wandb_run_id,
    )
    if wandb_logger.enabled:
        print(f"[train.py] W&B logging enabled, run_id={wandb_logger.run_id}")
    else:
        print("[train.py] W&B logging disabled (WANDB_API_KEY not set)")

    train_config = build_train_config(cfg["training"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_training_loop(
            model=model,
            optimizer=optimizer,
            replay=replay,
            eval_replay=eval_replay,
            config=train_config,
            checkpoint_dir=args.checkpoint_dir,
            save_every=cfg["checkpoint"]["save_every_n_steps"],
            eval_every=cfg["eval"]["every_n_steps"],
            log_every=cfg["wandb"]["log_every_n_steps"],
            wandb_logger=wandb_logger,
            start_step=start_step,
        )
    finally:
        wandb_logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
