"""Typed-field tokenizer — GameState → (44, 384) token sequence for transformer encoder.

Per spec Section 5.1–5.2:
    1 position + 6 party slots + 20 bag slots + 1 badges + 8 event-flag chunks
    + 1 money + 1 time + 1 battle + 4 tile-patches + 1 menu = 44 tokens

Each token is a sum of typed sub-embeddings, plus a learned field-type embedding
added before the first transformer layer. Bucketing for continuous values uses
linear quantization with bounds defined here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pokemon_planner.state import GameState


EXPECTED_NUM_TOKENS = 44

# Token-position layout (token index → group)
TOKEN_LAYOUT = {
    "position": (0, 1),
    "party": (1, 7),
    "bag": (7, 27),
    "badges": (27, 28),
    "events": (28, 36),
    "money": (36, 37),
    "time": (37, 38),
    "battle": (38, 39),
    "tiles": (39, 43),
    "menu": (43, 44),
}


@dataclass
class TokenizerConfig:
    embed_dim: int = 384
    # Discrete vocabularies
    num_map_ids: int = 256
    num_species: int = 256
    num_items: int = 256
    num_moves: int = 256
    # Continuous bucket counts
    num_x_buckets: int = 32
    num_y_buckets: int = 32
    num_level_buckets: int = 10
    num_hp_pct_buckets: int = 10
    num_qty_buckets: int = 10
    num_status_buckets: int = 256  # direct lookup over full status byte range
    num_money_buckets: int = 16
    num_time_buckets: int = 16
    num_turn_buckets: int = 16


def _bucket(value: int, max_value: int, num_buckets: int) -> int:
    """Quantize value in [0, max_value] to a bucket index in [0, num_buckets)."""
    if max_value == 0:
        return 0
    idx = int(value * num_buckets / (max_value + 1))
    return max(0, min(num_buckets - 1, idx))


class Tokenizer(nn.Module):
    """Maps a list of GameState to a (B, 44, embed_dim) tensor."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim

        # Discrete embeddings
        self.emb_map = nn.Embedding(cfg.num_map_ids, D)
        self.emb_x = nn.Embedding(cfg.num_x_buckets, D)
        self.emb_y = nn.Embedding(cfg.num_y_buckets, D)

        self.emb_species = nn.Embedding(cfg.num_species, D)
        self.emb_level = nn.Embedding(cfg.num_level_buckets, D)
        self.emb_hp_pct = nn.Embedding(cfg.num_hp_pct_buckets, D)
        self.emb_status = nn.Embedding(cfg.num_status_buckets, D)
        self.emb_move = nn.Embedding(cfg.num_moves, D)

        self.emb_item = nn.Embedding(cfg.num_items, D)
        self.emb_qty = nn.Embedding(cfg.num_qty_buckets, D)

        # Badges: 8 binary bits → small MLP
        self.mlp_badges = nn.Sequential(nn.Linear(8, D), nn.GELU(), nn.Linear(D, D))

        # Event flag chunks: 32 bytes (256 bits) → MLP per chunk
        self.mlp_event_chunk = nn.Sequential(nn.Linear(256, D), nn.GELU(), nn.Linear(D, D))

        self.emb_money = nn.Embedding(cfg.num_money_buckets, D)
        self.emb_time = nn.Embedding(cfg.num_time_buckets, D)

        # Battle: combined embedding from species + level + hp_pct + turn
        self.emb_opp_species = nn.Embedding(cfg.num_species, D)
        self.emb_opp_level = nn.Embedding(cfg.num_level_buckets, D)
        self.emb_opp_hp = nn.Embedding(cfg.num_hp_pct_buckets, D)
        self.emb_turn = nn.Embedding(cfg.num_turn_buckets, D)
        self.battle_zero = nn.Parameter(torch.zeros(D))  # used when not in battle

        # Tile patches: 16x16 grid → 4 patches of 8x8 = 64 bytes each
        self.mlp_tile_patch = nn.Sequential(nn.Linear(64, D), nn.GELU(), nn.Linear(D, D))

        # Menu flags
        self.emb_menu = nn.Embedding(256, D)

        # Empty-slot sentinel for empty bag slots
        self.empty_bag_token = nn.Parameter(torch.randn(D) * 0.02)
        self.empty_party_token = nn.Parameter(torch.randn(D) * 0.02)

        # Field-type embeddings — one per token position
        self.field_type_emb = nn.Parameter(torch.randn(EXPECTED_NUM_TOKENS, D) * 0.02)

    def forward(self, states: list[GameState]) -> Tensor:
        """Tokenize a batch of states. Returns (B, 44, D)."""
        B = len(states)
        D = self.cfg.embed_dim
        device = self.emb_map.weight.device

        out = torch.zeros(B, EXPECTED_NUM_TOKENS, D, device=device)

        for b, s in enumerate(states):
            # Position token (0)
            x_b = _bucket(s.x, 255, self.cfg.num_x_buckets)
            y_b = _bucket(s.y, 255, self.cfg.num_y_buckets)
            out[b, 0] = (
                self.emb_map(torch.tensor(s.map_id, device=device))
                + self.emb_x(torch.tensor(x_b, device=device))
                + self.emb_y(torch.tensor(y_b, device=device))
            )

            # Party slots 1..6
            for i in range(6):
                if i < len(s.party):
                    slot = s.party[i]
                    lvl_b = _bucket(slot.level, 100, self.cfg.num_level_buckets)
                    hp_b = _bucket(
                        slot.hp_cur * 10 // max(1, slot.hp_max), 9, self.cfg.num_hp_pct_buckets
                    )
                    # status is a raw Game Boy byte; bit-flags 4/8/16/32/64 are
                    # meaningful distinct values — bucketing would collapse them.
                    tok = (
                        self.emb_species(torch.tensor(slot.species_id, device=device))
                        + self.emb_level(torch.tensor(lvl_b, device=device))
                        + self.emb_hp_pct(torch.tensor(hp_b, device=device))
                        + self.emb_status(torch.tensor(slot.status, device=device))
                    )
                    for m in slot.moves:
                        tok = tok + self.emb_move(torch.tensor(m, device=device))
                else:
                    tok = self.empty_party_token
                out[b, 1 + i] = tok

            # Bag slots 7..26
            for i in range(20):
                if i < len(s.bag):
                    slot = s.bag[i]
                    qty_b = _bucket(slot.qty, 99, self.cfg.num_qty_buckets)
                    tok = (
                        self.emb_item(torch.tensor(slot.item_id, device=device))
                        + self.emb_qty(torch.tensor(qty_b, device=device))
                    )
                else:
                    tok = self.empty_bag_token
                out[b, 7 + i] = tok

            # Badges (27)
            badge_bits = torch.tensor(
                [(s.badges >> i) & 1 for i in range(8)], dtype=torch.float32, device=device
            )
            out[b, 27] = self.mlp_badges(badge_bits)

            # Event flag chunks 28..35 (8 chunks of 32 bytes = 256 bits each)
            for i in range(8):
                chunk = s.event_flags[i * 32 : (i + 1) * 32]
                bits = torch.tensor(
                    [(byte >> j) & 1 for byte in chunk for j in range(8)],
                    dtype=torch.float32,
                    device=device,
                )  # 256 bits
                out[b, 28 + i] = self.mlp_event_chunk(bits)

            # Money (36)
            money_b = _bucket(s.money, 999_999, self.cfg.num_money_buckets)
            out[b, 36] = self.emb_money(torch.tensor(money_b, device=device))

            # Time (37)
            # 200K frames ≈ 55 min @ 60fps. Sessions longer clamp to last bucket;
            # acceptable for Phase 1a bootstrap. Phase 1b/c may revisit with log-bucket.
            time_b = _bucket(s.time_played_frames, 200_000, self.cfg.num_time_buckets)
            out[b, 37] = self.emb_time(torch.tensor(time_b, device=device))

            # Battle (38)
            if s.battle.in_battle:
                opp_lvl_b = _bucket(s.battle.opp_level, 100, self.cfg.num_level_buckets)
                # opp_hp is raw battle HP (not percentage). Max plausible value ~700;
                # 999 cap matches Pokemon Red's 3-digit HP display.
                opp_hp_b = _bucket(s.battle.opp_hp, 999, self.cfg.num_hp_pct_buckets)
                turn_b = _bucket(s.battle.turn, 99, self.cfg.num_turn_buckets)
                out[b, 38] = (
                    self.emb_opp_species(torch.tensor(s.battle.opp_species_id, device=device))
                    + self.emb_opp_level(torch.tensor(opp_lvl_b, device=device))
                    + self.emb_opp_hp(torch.tensor(opp_hp_b, device=device))
                    + self.emb_turn(torch.tensor(turn_b, device=device))
                )
            else:
                out[b, 38] = self.battle_zero

            # Tile patches 39..42 — 4 patches of 64 bytes
            tc = s.tile_collision  # 256 bytes
            for i in range(4):
                patch = torch.tensor(
                    list(tc[i * 64 : (i + 1) * 64]), dtype=torch.float32, device=device
                ) / 255.0
                out[b, 39 + i] = self.mlp_tile_patch(patch)

            # Menu (43)
            out[b, 43] = self.emb_menu(torch.tensor(s.menu_flags, device=device))

        # Add field-type embeddings (broadcast over batch)
        out = out + self.field_type_emb.unsqueeze(0)
        return out
