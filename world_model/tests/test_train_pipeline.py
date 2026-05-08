"""Tests for the training loop function — single-step training without crash."""
from pathlib import Path

import pytest
import torch

from pokemon_planner.state import (
    BattleState,
    BagSlot,
    GameState,
    PartySlot,
)
from pokemon_planner.data.replay import ReplayBuffer
from pokemon_planner.data.trajectory import Trajectory, TrajectoryStep
from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.train import (
    TrainingConfig,
    one_training_step,
)


def _state(map_id: int = 0, party_species: int = 0xB0) -> GameState:
    return GameState(
        map_id=map_id, x=5, y=5,
        party=(PartySlot(species_id=party_species, level=10, hp_cur=20, hp_max=24,
                         status=0, moves=(0, 0, 0, 0)),),
        bag=(BagSlot(item_id=0x04, qty=5),),
        badges=0,
        event_flags=bytes(256),
        money=100, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


def _populate_buffer(buf: ReplayBuffer, n_episodes: int = 3, ep_length: int = 10) -> None:
    for ep in range(n_episodes):
        steps = tuple(
            TrajectoryStep(
                state=_state(map_id=i, party_species=0xB0 + (i % 4)),
                action=i % 9, reward=0.5, done=(i == ep_length - 1), info={},
            )
            for i in range(ep_length)
        )
        traj = Trajectory(steps=steps, source="v2_1310720", episode_id=ep, seed=ep)
        buf.add(traj, priority=0.3)


@pytest.fixture
def small_config():
    return WorldModelConfig(
        action_dim=9, latent_dim=64, d_model=64,
        encoder_layers=2, encoder_heads=2,
        dynamics_layers=2, dynamics_heads=2,
        prediction_layers=2, prediction_heads=2,
        ffn_expansion=2, dropout=0.0,
        goal_emb_dim=64,
    )


def test_one_training_step_runs_without_crash(tmp_path: Path, small_config):
    """A single forward+backward+optim step on real replay data."""
    buf = ReplayBuffer(root=tmp_path)
    _populate_buffer(buf)

    wm = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)
    train_cfg = TrainingConfig(
        batch_size=4, k_unroll=2, lr=1e-4, fp16=False, gradient_checkpointing=False,
    )

    components = one_training_step(
        model=wm, optimizer=optimizer, replay=buf, config=train_cfg, step=0,
    )

    assert "total" in components
    assert components["total"] > 0
    assert all(c > 0 for c in [components["obs"], components["policy"], components["reward"]])


def test_one_training_step_produces_gradient_flow(tmp_path: Path, small_config):
    buf = ReplayBuffer(root=tmp_path)
    _populate_buffer(buf)

    wm = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)
    train_cfg = TrainingConfig(
        batch_size=4, k_unroll=2, lr=1e-4, fp16=False, gradient_checkpointing=False,
    )

    p0 = next(wm.parameters()).clone()

    one_training_step(model=wm, optimizer=optimizer, replay=buf,
                      config=train_cfg, step=0)

    p1 = next(wm.parameters())
    assert not torch.equal(p0, p1), "Parameters did not update"


def test_one_training_step_with_fp16(tmp_path: Path, small_config):
    """fp16 training should run on CUDA; on CPU it falls back to fp32 silently."""
    buf = ReplayBuffer(root=tmp_path)
    _populate_buffer(buf)

    wm = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
    train_cfg = TrainingConfig(
        batch_size=4, k_unroll=2, lr=1e-4,
        fp16=True, gradient_checkpointing=False,
    )

    components = one_training_step(
        model=wm, optimizer=optimizer, replay=buf, config=train_cfg, step=0, scaler=scaler,
    )
    assert torch.isfinite(torch.tensor(components["total"]))
