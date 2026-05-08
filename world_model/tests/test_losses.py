"""Tests for the joint loss with placeholder targets."""
import pytest
import torch

from pokemon_planner.state import (
    BattleState,
    BagSlot,
    GameState,
    PartySlot,
)
from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.losses import (
    JointLossWeights,
    compute_joint_loss,
    targets_from_states,
)


def _state(map_id: int = 0, x: int = 5, party_species: int = 0xB0) -> GameState:
    return GameState(
        map_id=map_id, x=x, y=0,
        party=(PartySlot(species_id=party_species, level=10, hp_cur=20, hp_max=24,
                         status=0, moves=(0, 0, 0, 0)),),
        bag=(BagSlot(item_id=0x04, qty=5),),
        badges=0,
        event_flags=bytes(256),
        money=100, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
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


def test_targets_from_states_shapes(small_config):
    """targets_from_states extracts per-field tensors from a list of next-states."""
    next_states = [_state(map_id=i, x=i, party_species=0xB0 + i) for i in range(4)]
    targets = targets_from_states(next_states, small_config)
    assert targets["map_id"].shape == (4,)
    assert targets["x"].shape == (4,)
    assert targets["party_size"].shape == (4,)
    assert targets["party_species"].shape == (4, small_config.num_party_slots)
    assert targets["badges"].shape == (4, 8)
    assert targets["event_flags"].shape == (4, small_config.event_flags_bytes * 8)
    assert targets["tile_collision"].shape == (4, 256)


def test_compute_joint_loss_returns_finite(small_config):
    wm = WorldModel(small_config)
    states = [_state(map_id=i) for i in range(4)]
    next_states = [_state(map_id=i + 1) for i in range(4)]
    actions = torch.randint(0, small_config.action_dim, (4,))
    rewards = torch.randn(4)
    mc_returns = torch.randn(4)
    goal_emb = torch.randn(4, small_config.goal_emb_dim)

    out = wm(states, actions, goal_emb)
    targets = targets_from_states(next_states, small_config)

    loss, components = compute_joint_loss(
        wm_out=out,
        action_targets=actions,
        next_state_targets=targets,
        reward_targets=rewards,
        mc_return_targets=mc_returns,
        prev_latent=out["s"].detach(),
        next_latent_target=wm.h(next_states).detach(),
        weights=JointLossWeights(),
    )
    assert torch.isfinite(loss).all()
    assert "obs" in components
    assert "value" in components
    assert "policy" in components
    assert "reward" in components
    assert "consistency" in components


def test_compute_joint_loss_supports_backward(small_config):
    """Verify the loss is differentiable end-to-end."""
    wm = WorldModel(small_config)
    states = [_state(map_id=i) for i in range(4)]
    next_states = [_state(map_id=i + 1) for i in range(4)]
    actions = torch.randint(0, small_config.action_dim, (4,))
    rewards = torch.randn(4)
    mc_returns = torch.randn(4)
    goal_emb = torch.randn(4, small_config.goal_emb_dim)

    out = wm(states, actions, goal_emb)
    targets = targets_from_states(next_states, small_config)

    loss, _ = compute_joint_loss(
        wm_out=out, action_targets=actions, next_state_targets=targets,
        reward_targets=rewards, mc_return_targets=mc_returns,
        prev_latent=out["s"].detach(),
        next_latent_target=wm.h(next_states).detach(),
        weights=JointLossWeights(),
    )
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.parameters())
    assert has_grad


def test_loss_weights_default_match_spec():
    w = JointLossWeights()
    assert w.obs == 1.0
    assert w.value == 0.25
    assert w.policy == 1.0
    assert w.reward == 0.5
    assert w.consistency == 0.1
