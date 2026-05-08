"""Tests for the eval-during-training module."""
import pytest
import torch

from pokemon_planner.state import (
    BattleState,
    BagSlot,
    GameState,
    PartySlot,
)
from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.eval import (
    DODGate,
    check_dod_gate,
    run_eval,
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
def small_config():
    return WorldModelConfig(
        action_dim=9, latent_dim=64, d_model=64,
        encoder_layers=2, encoder_heads=2,
        dynamics_layers=2, dynamics_heads=2,
        prediction_layers=2, prediction_heads=2,
        ffn_expansion=2, dropout=0.0,
        goal_emb_dim=64,
    )


def test_run_eval_returns_per_field_accuracies(small_config):
    wm = WorldModel(small_config)
    states = [_state(map_id=i) for i in range(8)]
    next_states = [_state(map_id=i + 1) for i in range(8)]
    actions = torch.randint(0, small_config.action_dim, (8,))
    goal_emb = torch.randn(8, small_config.goal_emb_dim)

    metrics = run_eval(
        model=wm,
        eval_states=states,
        eval_next_states=next_states,
        eval_actions=actions,
        eval_goal_embs=goal_emb,
        batch_size=4,
        max_batches=2,
    )

    # Required keys per spec 6.5 / DoD gate
    for key in ["acc/map_id", "acc/x", "acc/y", "acc/party_species_slot_0"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_dod_gate_pass():
    metrics = {
        "acc/map_id": 0.85,
        "acc/x": 0.83,
        "acc/y": 0.81,
        "acc/party_species_slot_0": 0.80,
    }
    gate = DODGate(thresholds={
        "acc/map_id": 0.80, "acc/x": 0.80, "acc/y": 0.80, "acc/party_species_slot_0": 0.80,
    })
    assert check_dod_gate(metrics, gate) is True


def test_dod_gate_fail():
    metrics = {
        "acc/map_id": 0.85,
        "acc/x": 0.83,
        "acc/y": 0.50,    # below 0.80
        "acc/party_species_slot_0": 0.80,
    }
    gate = DODGate(thresholds={
        "acc/map_id": 0.80, "acc/x": 0.80, "acc/y": 0.80, "acc/party_species_slot_0": 0.80,
    })
    assert check_dod_gate(metrics, gate) is False


def test_run_eval_does_not_modify_model_train_mode(small_config):
    wm = WorldModel(small_config)
    wm.train()
    states = [_state(map_id=i) for i in range(4)]
    next_states = [_state(map_id=i + 1) for i in range(4)]
    actions = torch.randint(0, small_config.action_dim, (4,))
    goal_emb = torch.randn(4, small_config.goal_emb_dim)

    run_eval(
        model=wm, eval_states=states, eval_next_states=next_states,
        eval_actions=actions, eval_goal_embs=goal_emb,
        batch_size=4, max_batches=1,
    )
    assert wm.training is True   # restored after eval
