"""Tests for the typed-field tokenizer."""
import pytest
import torch

from pokemon_planner.state import (
    BattleState,
    BagSlot,
    GameState,
    PartySlot,
)
from pokemon_planner.world_model.tokenizer import (
    EXPECTED_NUM_TOKENS,
    Tokenizer,
    TokenizerConfig,
)


def _state(map_id: int = 5) -> GameState:
    return GameState(
        map_id=map_id, x=10, y=12,
        party=(
            PartySlot(species_id=0xB0, level=12, hp_cur=20, hp_max=24,
                      status=0, moves=(0x21, 0x33, 0, 0)),
        ),
        bag=(BagSlot(item_id=0x04, qty=5), BagSlot(item_id=0x14, qty=3)),
        badges=0b0000_0011,
        event_flags=bytes(256),
        money=300, time_played_frames=42,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


def _config() -> TokenizerConfig:
    return TokenizerConfig(
        embed_dim=384,
        num_map_ids=256,
        num_species=256,
        num_items=256,
        num_x_buckets=32,
        num_y_buckets=32,
        num_level_buckets=10,
        num_hp_pct_buckets=10,
        num_qty_buckets=10,
        num_status_buckets=8,
        num_moves=256,
        num_money_buckets=16,
        num_time_buckets=16,
        num_turn_buckets=16,
    )


def test_expected_num_tokens_is_44():
    assert EXPECTED_NUM_TOKENS == 44


def test_tokenizer_construction():
    tok = Tokenizer(_config())
    assert isinstance(tok, torch.nn.Module)


def test_tokenizer_single_state_returns_44x384():
    tok = Tokenizer(_config())
    state = _state()
    out = tok([state])
    assert out.shape == (1, 44, 384)


def test_tokenizer_batch_returns_correct_shape():
    tok = Tokenizer(_config()).to(torch.device("cpu"))
    states = [_state(map_id=i) for i in range(4)]
    out = tok(states)
    assert out.shape == (4, 44, 384)


def test_tokenizer_no_nan_for_battle_state():
    tok = Tokenizer(_config())
    state = GameState(
        map_id=0, x=0, y=0, party=(), bag=(), badges=0,
        event_flags=bytes(256), money=0, time_played_frames=0,
        battle=BattleState(in_battle=True, opp_species_id=0x09, opp_level=20,
                           opp_hp=50, turn=3),
        tile_collision=bytes(256), menu_flags=0,
    )
    out = tok([state])
    assert torch.isfinite(out).all()


def test_tokenizer_field_type_embeddings_shape():
    tok = Tokenizer(_config())
    assert tok.field_type_emb.shape == (44, 384)


def test_tokenizer_param_count_reasonable():
    """Tokenizer should be a few million params at most."""
    tok = Tokenizer(_config())
    n = sum(p.numel() for p in tok.parameters())
    assert n < 10_000_000, f"Tokenizer too large: {n} params"
