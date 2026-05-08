"""Shape and forward-pass tests for the Phase 1a transformer architecture."""
import pytest
import torch

from pokemon_planner.state import (
    BattleState,
    GameState,
)
from pokemon_planner.world_model.arch import (
    DynamicsNet,
    PredictionNet,
    RepresentationNet,
    WorldModel,
    WorldModelConfig,
)


def _state(map_id: int = 0) -> GameState:
    return GameState(
        map_id=map_id, x=0, y=0, party=(), bag=(), badges=0,
        event_flags=bytes(256), money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


@pytest.fixture
def small_config() -> WorldModelConfig:
    """Tiny config for fast tests; production config will be in configs/phase_1a.yaml."""
    return WorldModelConfig(
        action_dim=9,
        latent_dim=128,
        d_model=128,
        encoder_layers=2,
        encoder_heads=4,
        dynamics_layers=2,
        dynamics_heads=4,
        prediction_layers=2,
        prediction_heads=4,
        ffn_expansion=4,
        dropout=0.1,
        goal_emb_dim=128,
    )


def test_representation_forward_shape(small_config):
    h = RepresentationNet(small_config)
    states = [_state(map_id=i) for i in range(4)]
    s = h(states)
    assert s.shape == (4, small_config.latent_dim)


def test_dynamics_forward_shapes(small_config):
    g = DynamicsNet(small_config)
    s = torch.randn(4, small_config.latent_dim)
    a = torch.randint(0, small_config.action_dim, (4,))
    out = g(s, a)
    assert out["s_next"].shape == (4, small_config.latent_dim)
    assert out["r_pred"].shape == (4,)
    # Per-field obs predictions
    assert "obs_pred" in out
    assert "map_id" in out["obs_pred"]
    assert out["obs_pred"]["map_id"].shape == (4, small_config.num_map_ids)


def test_prediction_forward_shapes(small_config):
    f = PredictionNet(small_config)
    s = torch.randn(4, small_config.latent_dim)
    goal_emb = torch.randn(4, small_config.goal_emb_dim)
    pi, v = f(s, goal_emb)
    assert pi.shape == (4, small_config.action_dim)
    assert v.shape == (4,)


def test_world_model_full_forward(small_config):
    wm = WorldModel(small_config)
    states = [_state(map_id=i) for i in range(4)]
    a = torch.randint(0, small_config.action_dim, (4,))
    goal_emb = torch.randn(4, small_config.goal_emb_dim)

    out = wm(states, a, goal_emb)
    assert out["s"].shape == (4, small_config.latent_dim)
    assert out["s_next"].shape == (4, small_config.latent_dim)
    assert out["pi"].shape == (4, small_config.action_dim)
    assert out["v"].shape == (4,)
    assert "obs_pred" in out


def test_world_model_no_nan_at_init(small_config):
    wm = WorldModel(small_config)
    states = [_state(map_id=i) for i in range(8)]
    a = torch.randint(0, small_config.action_dim, (8,))
    goal_emb = torch.randn(8, small_config.goal_emb_dim)
    out = wm(states, a, goal_emb)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert torch.isfinite(v).all(), f"NaN/Inf in {k}"
        elif isinstance(v, dict):
            for fk, fv in v.items():
                if isinstance(fv, torch.Tensor):
                    assert torch.isfinite(fv).all(), f"NaN/Inf in obs_pred[{fk}]"
                elif isinstance(fv, list):
                    for i, ffv in enumerate(fv):
                        assert torch.isfinite(ffv).all(), f"NaN/Inf in obs_pred[{fk}][{i}]"


def test_param_count_in_phase_1a_envelope():
    """Production config should land at ~67M params (allow 50M-90M envelope)."""
    cfg = WorldModelConfig()  # defaults match spec section 5.1-5.5
    wm = WorldModel(cfg)
    n = sum(p.numel() for p in wm.parameters())
    assert 50_000_000 <= n <= 90_000_000, f"Param count out of envelope: {n}"


def test_world_model_train_eval_modes(small_config):
    wm = WorldModel(small_config)
    wm.train()
    assert wm.training
    wm.eval()
    assert not wm.training
