"""Phase 1a world-model architecture — full transformer h/g/f.

Three networks per spec Section 5:
- RepresentationNet (h): GameState list → latent (B, latent_dim)
- DynamicsNet (g): (latent, action) → next_latent + per-field obs preds + reward
- PredictionNet (f): (latent, goal_emb) → policy logits + value

Total ~67M parameters at default config. fp16-safe; supports gradient
checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pokemon_planner.state import GameState
from pokemon_planner.world_model.tokenizer import (
    EXPECTED_NUM_TOKENS,
    Tokenizer,
    TokenizerConfig,
)


@dataclass
class WorldModelConfig:
    # Action / latent / goal dims
    action_dim: int = 9
    latent_dim: int = 256
    d_model: int = 384
    goal_emb_dim: int = 384

    # Encoder (representation) hyperparameters
    encoder_layers: int = 12
    encoder_heads: int = 8

    # Dynamics hyperparameters
    dynamics_layers: int = 12
    dynamics_heads: int = 8

    # Prediction hyperparameters
    prediction_layers: int = 4
    prediction_heads: int = 8

    # Shared transformer settings
    ffn_expansion: int = 4
    dropout: float = 0.1

    # Per-field output sizes for obs prediction (matches Phase 0 schema)
    num_map_ids: int = 256
    num_x_values: int = 256
    num_y_values: int = 256
    num_party_slots: int = 6
    num_species: int = 256
    num_levels: int = 100
    num_status_codes: int = 256
    num_moves: int = 256
    num_bag_slots: int = 20
    num_items: int = 256
    num_qty_levels: int = 100
    num_money_buckets: int = 16
    num_battle_species: int = 256
    num_battle_levels: int = 100
    tile_collision_codes: int = 7  # 0..5 + 255 → mapped to 6 codes for prediction (+1 unknown)
    menu_flags_size: int = 256
    event_flags_bytes: int = 256


def _make_transformer(d_model: int, nhead: int, num_layers: int,
                      ffn_expansion: int, dropout: float) -> nn.TransformerEncoder:
    """Build a stack of TransformerEncoderLayer with batch_first=True."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * ffn_expansion,
        dropout=dropout,
        batch_first=True,
        norm_first=True,    # pre-LN for training stability
        activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def _sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    """Standard sinusoidal positional encodings, shape (seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# ---- h ----

class RepresentationNet(nn.Module):
    """h(state_list) → latent (B, latent_dim)."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer(TokenizerConfig(embed_dim=cfg.d_model))
        self.register_buffer(
            "pos_encoding",
            _sinusoidal_pe(EXPECTED_NUM_TOKENS, cfg.d_model),
            persistent=False,
        )
        self.transformer = _make_transformer(
            cfg.d_model, cfg.encoder_heads, cfg.encoder_layers,
            cfg.ffn_expansion, cfg.dropout,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(self, states: list[GameState]) -> Tensor:
        tokens = self.tokenizer(states)                           # (B, 44, D)
        tokens = tokens + self.pos_encoding.unsqueeze(0)
        h = self.transformer(tokens)                              # (B, 44, D)
        h = self.norm(h)
        pooled = h.mean(dim=1)                                    # (B, D)
        return self.proj(pooled)                                  # (B, latent_dim)


# ---- g ----

class DynamicsNet(nn.Module):
    """g(s, a) → s' + per-field obs predictions + reward.

    Implementation: project (s, a_emb) into a 2-token sequence, run a transformer,
    pool to a single state vector, then split into the three heads.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action_emb = nn.Embedding(cfg.action_dim, cfg.d_model)
        self.s_proj = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.transformer = _make_transformer(
            cfg.d_model, cfg.dynamics_heads, cfg.dynamics_layers,
            cfg.ffn_expansion, cfg.dropout,
        )
        self.s_next_head = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.r_head = nn.Linear(cfg.d_model, 1)

        D = cfg.d_model
        H = cfg.d_model
        self.head_map_id = nn.Sequential(nn.Linear(D, H), nn.GELU(),
                                         nn.Linear(H, cfg.num_map_ids))
        self.head_x = nn.Sequential(nn.Linear(D, H), nn.GELU(),
                                    nn.Linear(H, cfg.num_x_values))
        self.head_y = nn.Sequential(nn.Linear(D, H), nn.GELU(),
                                    nn.Linear(H, cfg.num_y_values))
        self.head_party_size = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, cfg.num_party_slots + 1),
        )
        self.head_party_species = nn.ModuleList([
            nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, cfg.num_species))
            for _ in range(cfg.num_party_slots)
        ])
        self.head_party_level = nn.ModuleList([
            nn.Sequential(nn.Linear(D, H), nn.GELU(),
                          nn.Linear(H, cfg.num_levels + 1))
            for _ in range(cfg.num_party_slots)
        ])
        self.head_badges = nn.Linear(D, 8)               # 8 binary logits
        self.head_event_flags = nn.Linear(D, cfg.event_flags_bytes * 8)  # 2048 binary
        self.head_money = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, cfg.num_money_buckets),
        )
        self.head_battle_in = nn.Linear(D, 1)            # binary
        self.head_battle_species = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, cfg.num_battle_species),
        )
        self.head_battle_level = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, cfg.num_battle_levels + 1),
        )
        self.head_tile_collision = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, 256 * cfg.tile_collision_codes),
        )
        self.head_menu_flags = nn.Sequential(
            nn.Linear(D, H), nn.GELU(),
            nn.Linear(H, cfg.menu_flags_size),
        )

    def forward(self, s: Tensor, a: Tensor) -> dict[str, Tensor]:
        B = s.shape[0]
        a_emb = self.action_emb(a)                                # (B, D)
        s_in = self.s_proj(s)                                     # (B, D)
        seq = torch.stack([s_in, a_emb], dim=1)                  # (B, 2, D)
        h = self.transformer(seq)                                 # (B, 2, D)
        z = h[:, 0, :]                                            # (B, D) — state-position output

        s_next = self.s_next_head(z)
        r_pred = self.r_head(z).squeeze(-1)

        obs_pred = {
            "map_id": self.head_map_id(z),
            "x": self.head_x(z),
            "y": self.head_y(z),
            "party_size": self.head_party_size(z),
            "party_species": [head(z) for head in self.head_party_species],
            "party_level": [head(z) for head in self.head_party_level],
            "badges": self.head_badges(z),
            "event_flags": self.head_event_flags(z),
            "money": self.head_money(z),
            "battle_in": self.head_battle_in(z).squeeze(-1),
            "battle_species": self.head_battle_species(z),
            "battle_level": self.head_battle_level(z),
            "tile_collision": self.head_tile_collision(z).view(
                B, 256, self.cfg.tile_collision_codes
            ),
            "menu_flags": self.head_menu_flags(z),
        }

        return {"s_next": s_next, "r_pred": r_pred, "obs_pred": obs_pred}


# ---- f ----

class PredictionNet(nn.Module):
    """f(s, goal_emb) → (policy logits, value)."""

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.latent_dim + cfg.goal_emb_dim, cfg.d_model)
        self.register_buffer(
            "pos_encoding",
            _sinusoidal_pe(1, cfg.d_model),
            persistent=False,
        )
        self.transformer = _make_transformer(
            cfg.d_model, cfg.prediction_heads, cfg.prediction_layers,
            cfg.ffn_expansion, cfg.dropout,
        )
        self.policy_head = nn.Linear(cfg.d_model, cfg.action_dim)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, s: Tensor, goal_emb: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([s, goal_emb], dim=-1)                      # (B, latent+goal)
        x = self.input_proj(x).unsqueeze(1)                       # (B, 1, D)
        x = x + self.pos_encoding.unsqueeze(0)
        h = self.transformer(x).squeeze(1)                        # (B, D)
        pi = self.policy_head(h)
        v = self.value_head(h).squeeze(-1)
        return pi, v


# ---- WorldModel ----

class WorldModel(nn.Module):
    """Phase 1a wrapper exposing h, g, f as submodules."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.h = RepresentationNet(config)
        self.g = DynamicsNet(config)
        self.f = PredictionNet(config)

    def forward(self, states: list[GameState], action: Tensor,
                goal_emb: Tensor) -> dict:
        s = self.h(states)
        g_out = self.g(s, action)
        pi, v = self.f(g_out["s_next"], goal_emb)
        return {
            "s": s,
            "s_next": g_out["s_next"],
            "obs_pred": g_out["obs_pred"],
            "r_pred": g_out["r_pred"],
            "pi": pi,
            "v": v,
        }
