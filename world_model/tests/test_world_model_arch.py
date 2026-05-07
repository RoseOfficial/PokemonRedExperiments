"""Shape and forward-pass tests for the world-model architecture stubs."""
import pytest
import torch

from pokemon_planner.world_model.arch import (
    DynamicsNet,
    PredictionNet,
    RepresentationNet,
    WorldModel,
    WorldModelConfig,
)


@pytest.fixture
def config() -> WorldModelConfig:
    return WorldModelConfig(
        obs_dim=128,            # placeholder — real Phase 1 obs is ~660 bytes typed
        action_dim=9,           # 8 buttons + no-op
        latent_dim=64,          # tiny for Phase 0 sanity tests
        goal_emb_dim=32,
        hidden_dim=128,
        num_predicate_types=6,  # catch/beat/reach/have_item/level/evolve
        num_entities=151,       # one for each Pokemon ID for now
    )


def test_representation_forward_shape(config):
    h = RepresentationNet(config)
    obs = torch.randn(4, config.obs_dim)
    s = h(obs)
    assert s.shape == (4, config.latent_dim)


def test_dynamics_forward_shapes(config):
    g = DynamicsNet(config)
    s = torch.randn(4, config.latent_dim)
    a = torch.randint(0, config.action_dim, (4,))
    s_next, obs_pred, r_pred = g(s, a)
    assert s_next.shape == (4, config.latent_dim)
    assert obs_pred.shape == (4, config.obs_dim)
    assert r_pred.shape == (4,)


def test_prediction_forward_shapes(config):
    f = PredictionNet(config)
    s = torch.randn(4, config.latent_dim)
    goal_emb = torch.randn(4, config.goal_emb_dim)
    pi, v = f(s, goal_emb)
    assert pi.shape == (4, config.action_dim)
    assert v.shape == (4,)


def test_world_model_full_forward(config):
    """End-to-end forward through h -> g -> f."""
    wm = WorldModel(config)
    obs = torch.randn(4, config.obs_dim)
    a = torch.randint(0, config.action_dim, (4,))
    goal_emb = torch.randn(4, config.goal_emb_dim)

    s = wm.h(obs)
    s_next, obs_pred, r_pred = wm.g(s, a)
    pi, v = wm.f(s_next, goal_emb)

    assert s.shape == (4, config.latent_dim)
    assert pi.shape == (4, config.action_dim)
    assert v.shape == (4,)


def test_world_model_no_nan_at_init(config):
    """Sanity: no NaNs from a fresh model on random input."""
    wm = WorldModel(config)
    obs = torch.randn(8, config.obs_dim)
    a = torch.randint(0, config.action_dim, (8,))
    goal_emb = torch.randn(8, config.goal_emb_dim)
    s = wm.h(obs)
    s_next, obs_pred, r_pred = wm.g(s, a)
    pi, v = wm.f(s_next, goal_emb)
    for t in (s, s_next, obs_pred, r_pred, pi, v):
        assert torch.isfinite(t).all(), f"NaN/Inf in tensor with shape {t.shape}"


def test_param_count_in_phase_0_budget(config):
    """Phase 0 stub should be small (<5M params) — full ~100M arrives later."""
    wm = WorldModel(config)
    n = sum(p.numel() for p in wm.parameters())
    assert n < 5_000_000
