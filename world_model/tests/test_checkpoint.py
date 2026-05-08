"""Tests for checkpoint save / load round-trips."""
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def small_config() -> WorldModelConfig:
    return WorldModelConfig(
        action_dim=9, latent_dim=64, d_model=64,
        encoder_layers=2, encoder_heads=2,
        dynamics_layers=2, dynamics_heads=2,
        prediction_layers=2, prediction_heads=2,
        ffn_expansion=2, dropout=0.0,
        goal_emb_dim=64,
    )


def test_save_load_roundtrip_preserves_model_weights(tmp_path: Path, small_config):
    wm1 = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm1.parameters(), lr=1e-3)

    state = CheckpointState(
        model=wm1,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=42,
        ema=None,
        replay_buffer_position=100,
        wandb_run_id="test-run-id",
        config={"d_model": 64},
    )
    save_checkpoint(tmp_path / "ckpt.pt", state)

    wm2 = WorldModel(small_config)
    optimizer2 = torch.optim.Adam(wm2.parameters(), lr=1e-3)
    loaded = load_checkpoint(
        path=tmp_path / "ckpt.pt",
        model=wm2,
        optimizer=optimizer2,
        scheduler=None,
        scaler=None,
    )

    assert loaded.step == 42
    assert loaded.replay_buffer_position == 100
    assert loaded.wandb_run_id == "test-run-id"
    for (n1, p1), (n2, p2) in zip(wm1.named_parameters(), wm2.named_parameters()):
        assert torch.equal(p1, p2), f"weight mismatch in {n1}"


def test_save_load_preserves_optimizer_state(tmp_path: Path, small_config):
    wm = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)

    for _ in range(3):
        loss = sum(p.sum() for p in wm.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    state = CheckpointState(
        model=wm, optimizer=optimizer, scheduler=None, scaler=None,
        step=3, ema=None, replay_buffer_position=0,
        wandb_run_id=None, config={},
    )
    save_checkpoint(tmp_path / "ckpt.pt", state)

    wm2 = WorldModel(small_config)
    optimizer2 = torch.optim.Adam(wm2.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "ckpt.pt", model=wm2, optimizer=optimizer2,
                    scheduler=None, scaler=None)

    state1 = optimizer.state_dict()["state"]
    state2 = optimizer2.state_dict()["state"]
    assert state1.keys() == state2.keys()
    for k in state1:
        for sub_k in state1[k]:
            if isinstance(state1[k][sub_k], torch.Tensor):
                assert torch.equal(state1[k][sub_k], state2[k][sub_k])


def test_save_load_preserves_rng_state(tmp_path: Path, small_config):
    wm = WorldModel(small_config)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)

    # Set known RNG state
    torch.manual_seed(123)
    np.random.seed(456)
    random.seed(789)

    # Save the state
    state = CheckpointState(
        model=wm, optimizer=optimizer, scheduler=None, scaler=None,
        step=0, ema=None, replay_buffer_position=0,
        wandb_run_id=None, config={},
    )
    save_checkpoint(tmp_path / "ckpt.pt", state)

    # Capture what the generators produce immediately after the save (i.e. with
    # the saved-state generators advanced one draw)
    expected_torch = torch.rand(3)
    expected_np = np.random.rand(3)
    expected_py = random.random()

    # Corrupt the RNG state with a different seed
    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    # Load — should restore the saved-state generators (back to where they were
    # immediately after save_checkpoint completed)
    wm2 = WorldModel(small_config)
    optimizer2 = torch.optim.Adam(wm2.parameters(), lr=1e-3)
    load_checkpoint(tmp_path / "ckpt.pt", model=wm2, optimizer=optimizer2,
                    scheduler=None, scaler=None)

    # After load, the next draws should match exactly what we saw post-save
    actual_torch = torch.rand(3)
    actual_np = np.random.rand(3)
    actual_py = random.random()

    assert torch.equal(actual_torch, expected_torch), \
        f"torch RNG not restored: expected {expected_torch}, got {actual_torch}"
    assert np.array_equal(actual_np, expected_np), \
        f"numpy RNG not restored: expected {expected_np}, got {actual_np}"
    assert actual_py == expected_py, \
        f"python RNG not restored: expected {expected_py}, got {actual_py}"
