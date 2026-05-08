# Phase 1a — Bootstrap Data + First Real World-Model Training — Design Spec

**Status:** Draft (pending user approval)
**Date:** 2026-05-07
**Author:** Brainstormed collaboratively via Claude Code
**Working directory:** `world_model/` (extends Phase 0)
**Parent spec:** `docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md`
**Prior phase:** Phase 0 closed at commit `70e2674`; 54 tests green, package skeleton + state extractor + KB + goal DSL atoms + world-model architecture stubs in place.

---

## Executive Summary

Phase 1a delivers a trained ~67M-parameter goal-conditioned world model for Pokemon Red, validated on held-out trajectories with prediction accuracy ≥80% on `map_id`, `(x, y)` position, and `party_species`.

It builds on Phase 0's package skeleton by adding:

1. **Bootstrap data extraction** from all 4 v2 PPO checkpoints (~1M demonstration decisions to disk).
2. **Real tile-collision extraction** replacing Phase 0's 256-zero stub.
3. **Typed-field tokenizer** mapping `GameState` to a 44-token sequence.
4. **Full transformer architecture** for h, g, f networks (replacing Phase 0's MLP stubs).
5. **Production training pipeline** with joint loss, fp16, gradient checkpointing, replay buffer with persistence, eval-during-training, W&B integration, and auto-restart wrappers.

**Known caveat baked into the design:** The full joint loss is trained with placeholder targets — behavioral cloning on PPO actions for the policy head, Monte-Carlo returns over PPO's shaped reward for the value head. These will be retrained in Phase 1b once MCTS exists and can produce real goal-conditioned targets. The architecture is correct; the targets are temporary. This trade is explicit and ADR-tracked.

---

## Constraints (carried forward from parent spec, with Phase 1a additions)

| Property | Value |
|---|---|
| Hardware | Single consumer GPU, 8–12 GB VRAM |
| Time commitment | Full-time |
| Success criterion | Personal exploration; ≥80% per-field prediction accuracy gate |
| LLM dependency | None |
| Algorithm family | MuZero variant (goal-conditioned, stochastic) — Phase 1a trains dynamics + placeholder value/policy |
| Scope | One mega-plan covering all of Phase 1a (foundations + training) |
| Branch strategy | Direct commits to master |
| Author identity | RoseOfficial \<christopherscottkeller@gmail.com\> via per-commit `-c` overrides |

---

## Section 1 — Scope and Definition of Done

### 1.1 What Phase 1a delivers

A trained ~67M-parameter goal-conditioned world model, validated on held-out trajectories.

### 1.2 The five workstreams

```
1. Demonstration data extraction
   ├─ Run each of v2/runs/poke_{1310720,2621440,3932160,5242880}_steps.zip
   ├─ Record (state, action, reward) tuples per episode
   └─ Persist to a Parquet replay buffer (~5–10GB on disk)

2. Real tile-collision extraction
   ├─ Read current tileset ID from RAM (0xD367)
   ├─ Hard-coded collision tables per tileset (8 most-encountered for Phase 1a)
   ├─ Compute player tile coords from state.x, state.y
   └─ Index 16x16 tiles around player → 256 collision codes

3. Typed-field obs encoder (replaces Phase 0's flat-vector representation)
   ├─ Each GameState field → token (type embedding + value embedding)
   ├─ ~44 tokens per state; 384-d embeddings
   └─ ~22M-param transformer encoder over the sequence

4. Full ~67M-param world model (upgrades h, g, f from Phase 0 stubs)
   ├─ h: transformer encoder (Workstream 3): ~22M
   ├─ g: dynamics function with per-field obs heads: ~35M
   └─ f: prediction function (goal-conditioned policy + value): ~7M

5. Production training pipeline
   ├─ Joint loss with placeholder value/policy targets
   ├─ fp16 + gradient checkpointing for 8–12GB VRAM
   ├─ Checkpoint/resume + go-forever auto-restart (bash + ps1)
   ├─ W&B integration (no-op when WANDB_API_KEY unset)
   ├─ Eval-during-training on held-out trajectories every 2,000 steps
   └─ Replay buffer persisted across runs
```

### 1.3 Definition of done

1. ~1M demonstration steps extracted to disk across 4 v2 checkpoints
2. Real tile-collision extraction working (no more 256-zero stub)
3. Full transformer encoder + ~67M-param world model architecturally present and trainable
4. First 24h training run completes (or completes the data pass; whichever first)
5. Validation: ≥80% prediction accuracy on `map_id`, `(x, y)`, `party_species_slot_0` on held-out trajectories
6. STATE.md, progress.md, tuning.md updated with results
7. ADR-0002 documents the placeholder-targets decision

### 1.4 Known caveats baked in by design

**Placeholder targets for value and policy heads.** During Phase 1a:
- Policy targets are one-hot demo actions (behavioral cloning on PPO).
- Value targets are Monte-Carlo returns over PPO's shaped reward.
- A single learned dummy goal embedding is used during training (no goal sampling yet — that needs MCTS).

These trained heads will be **retrained in Phase 1b** once MCTS produces real goal-conditioned targets (visit-distribution policy targets, simulated-return value targets across actual goals). The architecture is correct; the targets are temporary. We're trading "first-training optimality" for "end-to-end pipeline working from day one" per the explicit user choice during brainstorm.

---

## Section 2 — File Structure

```
world_model/
├── pokemon_planner/
│   ├── env.py                              # MODIFIED: real tile-collision (replace stub)
│   ├── _tilesets.py                        # NEW: collision-table data per tileset
│   │
│   ├── data/                               # NEW package
│   │   ├── __init__.py
│   │   ├── trajectory.py                   # Trajectory format + pyarrow schema
│   │   └── replay.py                       # Priority-weighted replay buffer + persistence
│   │
│   ├── goals/
│   │   └── embedding.py                    # NEW: Goal → torch.Tensor for f-head conditioning
│   │
│   └── world_model/
│       ├── arch.py                         # REWRITTEN: full transformer h/g/f, no longer stubs
│       ├── tokenizer.py                    # NEW: GameState → typed-token sequence
│       ├── losses.py                       # NEW: joint loss (state/value/policy/reward/consistency)
│       ├── train.py                        # NEW: training loop (replaces train_stub.py role)
│       ├── eval.py                         # NEW: held-out validation
│       ├── checkpoint.py                   # NEW: save/load full training state
│       ├── wandb_logger.py                 # NEW: optional W&B integration
│       └── train_stub.py                   # KEPT as a regression smoke test (synthetic-data sanity)
│
├── configs/
│   └── phase_1a.yaml                       # NEW: all hyperparameters in one place
│
├── scripts/
│   ├── bootstrap_demos.py                  # NEW: run PPO ckpts → record trajectories
│   ├── train.py                            # NEW: CLI entry point for training
│   ├── validate.py                         # NEW: standalone validation runner
│   ├── go_forever.sh                       # NEW: auto-restart wrapper (bash)
│   └── go_forever.ps1                      # NEW: auto-restart wrapper (PowerShell)
│
├── tests/
│   ├── test_tile_collision.py              # NEW
│   ├── test_tilesets.py                    # NEW (unit tests for collision-table data)
│   ├── test_replay_buffer.py               # NEW
│   ├── test_trajectory.py                  # NEW
│   ├── test_trajectory_extraction.py       # NEW (integration: actually runs PPO briefly)
│   ├── test_tokenizer.py                   # NEW
│   ├── test_encoder.py                     # NEW
│   ├── test_world_model_arch.py            # MODIFIED: tests for new transformer arch
│   ├── test_losses.py                      # NEW
│   ├── test_goals_embedding.py             # NEW
│   ├── test_train_pipeline.py              # NEW (1-step training without crash)
│   ├── test_eval.py                        # NEW
│   ├── test_checkpoint.py                  # NEW (save → load → state matches)
│   └── test_e2e_phase_1a.py                # NEW (small training run, validate on held-out)
│
├── docs/
│   ├── adr/
│   │   └── 0002-full-joint-loss-with-placeholders.md  # NEW
│   └── data/
│       └── replay_format.md                # NEW: trajectory schema documentation
│
└── data/                                   # gitignored
    └── replay_buffer/
        ├── meta.json                       # per-trajectory metadata
        ├── traj_<source>_<seed>.parquet    # one file per source-checkpoint+episode batch
        └── eval/                           # held-out 5% for validation
```

### 2.1 Module responsibilities

| Module | Job | Depends on |
|---|---|---|
| `_tilesets.py` | Hard-coded collision tables per tileset (ROM-derived) | nothing |
| `env.py` (mod) | `read_state` now uses `_tilesets` for real tile-collision | `_tilesets`, Phase 0 schema |
| `data/trajectory.py` | One Trajectory = list of (obs, action, reward, done) + metadata | `state.py` |
| `data/replay.py` | Priority-weighted sampler + on-disk persistence | `data/trajectory` |
| `world_model/tokenizer.py` | `GameState → tensor[44, 384]` for transformer | `state.py` |
| `world_model/arch.py` | h (transformer encoder), g (transformer dynamics), f (transformer + heads) | `tokenizer`, torch |
| `world_model/losses.py` | Joint loss with placeholder targets | `arch` |
| `goals/embedding.py` | `Atom → tensor[384]` for f-head goal conditioning | `goals/dsl` |
| `world_model/train.py` | Training loop: sample batch, forward, loss, optim, log, checkpoint | all of the above |
| `world_model/eval.py` | Held-out trajectory eval; reports per-field accuracy | `arch`, `data/replay` |
| `world_model/checkpoint.py` | Save/load model+optim+lr_sched+rng_state+buffer_position | torch |
| `world_model/wandb_logger.py` | Wraps wandb calls; no-op if wandb not configured | wandb (optional) |
| `scripts/bootstrap_demos.py` | Standalone: load PPO checkpoint, run N steps, dump to replay buffer | `data/`, v2 env |
| `scripts/train.py` | CLI: parse args, build trainer, call `train.py` loop | `world_model/train` |
| `scripts/go_forever.{sh,ps1}` | Auto-resume wrapper: while true; do python ... --resume; done | bash / PowerShell |

### 2.2 Layout rationale

- **`data/` is its own subpackage**, not crammed into `world_model/`. The replay buffer is reused across phases — Phase 1b adds verification-loop divergences, Phase 2 adds full-memory observations. Out-of-`world_model` location makes that future independence cheap.
- **`tokenizer.py` is split from `arch.py`** so field-to-token logic is testable on its own and the encoder file stays focused.
- **`train.py` is in `pokemon_planner/world_model/`, not `scripts/`**. The CLI entry in `scripts/` is thin — argparse and configure — and the actual loop is importable code, testable without a CLI shell.
- **`_tilesets.py` is a leading-underscore module**, same convention as `_ram_addresses.py`. Internal data, not for external consumption.
- **Existing `train_stub.py` stays** as a fast pre-flight smoke test — verifies the architecture doesn't NaN on synthetic data using real Phase 1a `WorldModelConfig` dimensions.

---

## Section 3 — Demonstration Data Extraction

### 3.1 The capture pattern

The v2 PPO models output actions over v2's *original* observation format. We don't convert PPO; we run it on v2's env, but at each step we *also* call our `read_state(pyboy)` to capture our typed `GameState`.

```
v2 PPO model ─▶ action ─▶ v2 env step ─▶ pyboy advances 24 frames
                                          │
                                          ├─▶ v2 obs (PPO sees this — discarded)
                                          │
                                          └─▶ pokemon_planner.env.read_state(pyboy)
                                                          │
                                                          ▼
                                              GameState (we record this)
```

This sidesteps the "PPO was trained on v2 obs" / "we want our typed obs" mismatch — PPO acts in its native env; we observe through our own lens. Action sequence is unchanged.

### 3.2 Per-step record

```python
@dataclass
class TrajectoryStep:
    state: GameState              # our typed state from read_state()
    action: int                   # 0..8 (mapped from v2's action space — see 3.4)
    reward: float                 # v2's shaped reward (used for placeholder MC value targets)
    done: bool                    # episode boundary
    info: dict                    # PPO source ckpt, episode_id, step_within_episode
```

### 3.3 Per-source extraction targets

| Source | Episodes | Avg ep length (decisions) | Total decisions |
|---|---|---|---|
| `v2/runs/poke_1310720_steps.zip` | ~120 | ~2,083 | ~250K |
| `v2/runs/poke_2621440_steps.zip` | ~120 | ~2,083 | ~250K |
| `v2/runs/poke_3932160_steps.zip` | ~120 | ~2,083 | ~250K |
| `v2/runs/poke_5242880_steps.zip` | ~120 | ~2,083 | ~250K |
| **Total** | **~480** | | **~1.0M decisions** |

Decision = one MCTS-equivalent action; one decision = `action_freq=24` frames in v2's setup. PPO sampling stochastic (`predict(deterministic=False)`) for diversity; each episode seeded for reproducibility.

### 3.4 Action-space alignment

v2's env exposes 7 actions (UP, DOWN, LEFT, RIGHT, A, B, START — no SELECT). Our spec uses 9. Mapping:

```python
V2_ACTION_TO_OURS = {
    0: 0,  # UP
    1: 1,  # DOWN
    2: 2,  # LEFT
    3: 3,  # RIGHT
    4: 4,  # A
    5: 5,  # B
    6: 6,  # START
    # 7=SELECT, 8=NO-OP — never produced by v2 PPO; left for future data sources
}
```

### 3.5 On-disk format (Parquet)

One Parquet file per (checkpoint, batch-of-episodes). Columns:

| Column | Type | Notes |
|---|---|---|
| `episode_id` | int64 | Globally unique across all sources |
| `step_index` | int32 | 0-based within the episode |
| `state_bytes` | binary | msgpack-serialized GameState (compact, fast) |
| `action` | int8 | 0–8 |
| `reward` | float32 | v2's shaped reward |
| `done` | bool | true on last step of episode |
| `mc_return` | float32 | computed at extraction time, used as placeholder value target |
| `source` | dict (encoded as int) | which checkpoint produced this episode |
| `seed` | int64 | RNG seed for the episode (reproducibility) |

**msgpack over JSON or pickle:** GameState's 660 typed bytes serialize to ~700 bytes msgpack (vs. ~3KB JSON, vs. pickle's security risk). 1M decisions ≈ 720 MB raw, ~250–400 MB compressed (snappy or zstd via Parquet).

**Sidecar metadata** at `data/replay_buffer/meta.json` records sources, episode counts, total steps, eval split, schema version.

### 3.6 Train/validation split

Hold out **5% of episodes per checkpoint** (~24 episodes total, ~50K decisions) for validation. Selected by `episode_id % 20 == 0` for reproducibility. Split at episode level, not step level — prevents temporal leakage.

### 3.7 ReplayBuffer interface

```python
class ReplayBuffer:
    """Priority-weighted sampler over on-disk Parquet trajectories.

    Phase 1a: only demonstration source (priority=0.3 for all entries).
    Phase 1b: divergence (1.0), success (0.7), exploration (0.5) sources added.
    """

    def __init__(self, root: Path): ...
    def sample_batch(self, batch_size: int, k_unroll: int) -> Batch: ...
    def add(self, traj: Trajectory, priority: float) -> None: ...
    def stats(self) -> dict: ...
```

`sample_batch` returns sequences of length `k_unroll+1` — windows that span episode boundaries are rejected and resampled. Memory budget ~50 MB for the in-memory index over 1M entries.

### 3.8 Failure modes

- **Mid-checkpoint crash** — episode-level checkpointing in `bootstrap_demos.py`; resume skips already-extracted (source, episode_id) pairs via `meta.json`.
- **Disk full** — script does free-space check before starting; minimum required is configurable.
- **PPO checkpoint won't load** — log error, skip that source, continue. 750K decisions across 3 sources is still a valid Phase 1a starting point.
- **Action mapping drift** — assert `v2_action_dim == 7` at load time; fail loud on schema change.

---

## Section 4 — Real Tile-Collision Extraction

### 4.1 The problem

Phase 0's `_read_tile_collision_stub` returns 256 zero bytes. Phase 1a replaces this with a real 16x16 grid of "is this tile walkable?" codes centered on the player, so the world model can learn navigation dynamics — predict whether `UP` actually moves the player or bumps into a wall.

Pokemon Red's collision logic is two-step:
1. The current map loads tile IDs from the tile map into WRAM `0xC6E8+`.
2. When the player tries to move, the game looks up the destination tile's ID in the **current tileset's collision table** and either allows or blocks the move.

We replicate step 2 read-only.

### 4.2 Data sources

| What | Where | Access pattern |
|---|---|---|
| Current tileset ID | `0xD367` (CURRENT_TILESET) | Single byte, already in `_ram_addresses.py` |
| Player's tile coords | `state.x`, `state.y` | From Phase 0 schema |
| Loaded tile IDs around player | WRAM `0xC6E8+` (OverworldMap) | 360-byte region for the current screen |
| Per-tileset collision tables | Hard-coded from pret/pokered | Static Python data in `_tilesets.py` |

### 4.3 Collision codes

```
0   = walkable
1   = blocked (wall, water without Surf, building edge, etc.)
2   = ledge_down (one-way drop south)
3   = ledge_left
4   = ledge_right
5   = door / warp tile (causes map transition)
255 = unknown (off-map, in transition, or tileset not in our table)
```

### 4.4 Tilesets covered in Phase 1a

The 8 most-encountered tilesets in early-to-mid game (~95% coverage of bootstrap data):

```
OVERWORLD, HOUSE, MART, POKECENTER, FOREST, CAVERN, GYM, GATE
```

Remaining 9+ tilesets fall through to a default "all walkable" table — strictly worse, but only matters for late-game maps that bootstrap data barely visits. Phase 1c fills the rest.

### 4.5 The `_tilesets.py` module

Pure data:

```python
@dataclass(frozen=True)
class CollisionTable:
    blocked: frozenset[int]
    ledge_down: frozenset[int] = frozenset()
    ledge_left: frozenset[int] = frozenset()
    ledge_right: frozenset[int] = frozenset()
    warp: frozenset[int] = frozenset()

    def lookup(self, tile_id: int) -> int:
        if tile_id in self.warp: return 5
        if tile_id in self.ledge_down: return 2
        if tile_id in self.ledge_left: return 3
        if tile_id in self.ledge_right: return 4
        if tile_id in self.blocked: return 1
        return 0


COLLISION_TABLES: dict[int, CollisionTable] = {
    OVERWORLD: CollisionTable(blocked=frozenset({...}), ...),
    HOUSE: CollisionTable(...),
    # ... 8 total filled in Phase 1a
}

DEFAULT_TABLE = CollisionTable(blocked=frozenset())
```

Tile IDs sourced from a one-time read of `pret/pokered/data/tilesets/*_collision.asm`. The module's docstring records the upstream commit SHA.

### 4.6 The extraction algorithm

```python
def extract_tile_collision_16x16(pyboy: PyBoy, state: GameState) -> bytes:
    """256 bytes, row-major: out[y*16 + x] = collision code at (player_x-8+x, player_y-8+y)."""
    tileset_id = pyboy.memory[ram.CURRENT_TILESET]
    table = _tilesets.COLLISION_TABLES.get(tileset_id, _tilesets.DEFAULT_TABLE)

    out = bytearray(256)
    cx, cy = state.x, state.y
    for dy in range(-8, 8):
        for dx in range(-8, 8):
            tile_id = _read_tile_id_at(pyboy, cx + dx, cy + dy)
            code = 255 if tile_id is None else table.lookup(tile_id)
            out[(dy + 8) * 16 + (dx + 8)] = code
    return bytes(out)
```

### 4.7 Edge cases

| Case | Approach |
|---|---|
| Player on warp tile mid-transition | Return 255 for all 256 cells. World model treats transition states as noisy. |
| Map doesn't fit in 16x16 view | Off-map cells → 255. Distinct from walkable/blocked. |
| Tilesets not in Phase 1a's 8-table set | Falls through to `DEFAULT_TABLE`. Deliberately suboptimal; Phase 1c fills the rest. |
| Surf / Cut / Strength / Rock-Smash blockers | Encoded as `blocked`. HM-based unblocking is learned by the world model from data, not via tile codes. |
| Ledges (one-way drops) | Distinct codes (2/3/4); world model learns directional asymmetry. |

### 4.8 Testing

- **Unit tests** on `CollisionTable.lookup()` per tileset.
- **Integration test** against `init.state`: extract collision matrix, verify mix of codes appears, verify deterministic across consecutive reads.
- **Manual visual verification** in the plan: render extracted collision matrix alongside PyBoy screen for ~5 example states; eyeball-match walls in screen to `1`s in matrix. One-time, documented in the plan, not automated.

### 4.9 Phase 0 stub replacement

`pokemon_planner/env.py:_read_tile_collision_stub` renames to `_read_tile_collision` (no `_stub`) and rewrites to call `_tilesets`. Function signature unchanged; all Phase 0 tests still pass.

---

## Section 5 — Tokenizer + Full Transformer Architecture

### 5.1 Token layout (44 tokens × 384 dimensions)

| Token group | Count | Encoding |
|---|---|---|
| Position | 1 | `Embed(map_id) + Embed(x_bucket) + Embed(y_bucket)` summed |
| Party slots | 6 | `Embed(species_id) + Embed(level) + Embed(hp_pct_bucket) + Embed(status) + sum(move_emb[i])` |
| Bag slots | 20 | `Embed(item_id) + Embed(qty_bucket)` |
| Badges | 1 | 8-bit one-hot pattern → small MLP → 384d |
| Event flags | 8 | 256 bytes split into 8 chunks of 32; each chunk's bit pattern → small MLP → 384d |
| Money | 1 | `Embed(log_money_bucket)` |
| Time / progress | 1 | `Embed(time_played_bucket)` |
| Battle | 1 | When in battle: `Embed(opp_species) + Embed(opp_level) + Embed(opp_hp_pct) + Embed(turn_bucket)`. Else: zero. |
| Tile collision | 4 | 16x16 grid → 4 patches of 8x8 → MLP over 64-byte patch → 384d |
| Menu / dialogue mode | 1 | `Embed(menu_flags)` |
| **Total** | **44** | |

Each token is the **sum** of typed sub-embeddings (not concatenation). Constant 384d throughout. Field-type embeddings added before first transformer layer give typed identity.

Bucketing for continuous values (HP%, money, time, level): linear or log quantization into 16–64 buckets; boundaries documented in `tokenizer.py`.

### 5.2 Representation function `h`

```
Input:  tokens (B, 44, 384)
        + sinusoidal position encoding
        ↓
       12 × TransformerEncoderLayer(d_model=384, nhead=8, dim_feedforward=1536, dropout=0.1)
        ↓
       LayerNorm
        ↓
       Mean-pool → (B, 384) → Linear → (B, 256)        ← latent_dim
```

Per-layer params: 4 × 384² (attention) + 2 × 384 × 1536 (FFN) ≈ 1.77M. Twelve layers ≈ 21M. Plus tokenizer embeddings ≈ 1M. **Total `h`: ~22M.**

### 5.3 Dynamics function `g`

```
Inputs:  s (B, 256)   latent state
         a (B,)       action index 0..8

Action embedding: a → (B, 384)
State expansion:  s → (B, 1, 384)
Action concat:    [s; a_emb] → (B, 2, 384)

       ↓
       12 × TransformerLayer(d_model=384, ...)
       ↓
       Take output position 0 → (B, 384)
       ↓
       ┌──────────────────────────────────────────────┐
       │ Three parallel decoder heads:                │
       │   • next-latent head:     Linear → (B, 256) │
       │   • obs-prediction:       per-field MLPs     │
       │   • reward head:          Linear → (B,)     │
       └──────────────────────────────────────────────┘
```

**Per-field obs-prediction heads** (heaviest part of `g`):

| Field | Output | Loss |
|---|---|---|
| `map_id`, `x`, `y` | logits over 256 each | cross-entropy |
| `party_size` | logits over 7 | cross-entropy |
| Per-slot species/level/hp_cur/hp_max/status/moves | logits per byte | cross-entropy |
| `badges` | 8 binary logits | binary cross-entropy |
| `event_flags` | 256 binary logits | binary cross-entropy |
| `money` | logits over bucketed 1M | cross-entropy on bucket |
| `battle.*` | mixed | mixed |
| `tile_collision` | 256 categorical (6 codes) | cross-entropy |
| `menu_flags` | logits over 256 | cross-entropy |

Each head is a small MLP (384 → 384 → output_dim). Aggregate ~12M across all heads.

**Total `g`:** ~22M (transformer) + ~12M (heads) + ~1M (action) ≈ **~35M**.

### 5.4 Prediction function `f`

```
Inputs:  s (B, 256)         latent state
         goal (B, 384)      goal embedding from goals/embedding.py

Concat:  [s; goal] → (B, 640)
         ↓
         Linear → (B, 384)
         ↓
        4 × TransformerLayer(d_model=384, ...)
         ↓
        Mean-pool → (B, 384)
         ↓
         ┌──────────────────────────────────────────────┐
         │  policy head:   Linear → (B, 9)              │
         │  value head:    Linear → (B, 1)              │
         └──────────────────────────────────────────────┘
```

**Goal embedding:**
```python
def embed_goal(goal: Atom, predicate_emb: nn.Embedding, entity_emb: nn.Embedding) -> Tensor:
    pred_idx = PREDICATE_TO_INDEX[goal.predicate_type]   # 0..5
    ent_idx  = ENTITY_TO_INDEX[goal.entity]              # 0..255
    return torch.cat([predicate_emb(pred_idx), entity_emb(ent_idx)], dim=-1)  # (384,)
```

Predicate embeddings: 6 × 192 = 1.2K params. Entity embeddings: 256 × 192 = 49K params.

**Total `f`:** ~7M (transformer) + 50K (goal embeddings) + 4K (heads) ≈ **~7M**.

### 5.5 Total parameter budget

| Component | Params |
|---|---|
| Tokenizer (sub-embeddings + field types) | ~3M |
| `h` (representation transformer) | ~22M |
| `g` (dynamics transformer + per-field heads) | ~35M |
| `f` (prediction transformer + heads) | ~7M |
| Goal embedding tables | ~0.05M |
| **Total** | **~67M** |

Conservative for first training; can scale to 100M by widening layers (384 → 512) once convergence is verified.

### 5.6 VRAM budget at 8–12GB target

Training-time, fp16 + gradient checkpointing, batch=32, k=5 unroll:

| Component | VRAM |
|---|---|
| Model weights (fp16) | ~134 MB |
| Optimizer states (Adam, fp32 master) | ~536 MB |
| Gradients (fp16) | ~134 MB |
| Activations (k=5 unroll, batch=32, with checkpointing) | ~2.5 GB |
| Replay buffer staging | ~50 MB |
| Eval batch (no_grad, batch=64) | ~100 MB |
| Misc (W&B buffer, mp staging) | ~200 MB |
| **Subtotal** | **~3.7 GB** |
| Headroom | ~4.3 GB on 8GB / ~8.3 GB on 12GB |

**Knobs if VRAM-pressed:** batch 32→16, k 5→3, hidden dim 384→256 (drops to ~30M), disable gradient checkpointing on `f` for speed.

### 5.7 Activation checkpointing

Apply `torch.utils.checkpoint.checkpoint` to:
- Each transformer block in `h` (12 segments)
- Each transformer block in `g` (12 segments)
- The k-unroll loop in `g` (most VRAM-heavy)

Don't checkpoint `f` (recompute cost outweighs savings at its size).

### 5.8 Mixed precision

`torch.amp.autocast` wraps forward; loss in fp32 for stability; gradients via `torch.amp.GradScaler`. Standard PyTorch AMP recipe.

### 5.9 What changes from Phase 0's `arch.py`

| Phase 0 | Phase 1a |
|---|---|
| `RepresentationNet`: MLP over flat 64d obs | Transformer over 44-token sequence |
| `DynamicsNet`: MLP over (latent, action_emb) | Transformer + per-field obs prediction heads |
| `PredictionNet`: MLP over (latent, goal_emb) | Smaller transformer + heads |
| Total ~370K params | Total ~67M params |
| `WorldModelConfig`: 7 fields | `WorldModelConfig`: 12 fields (transformer dims, layer counts, bucket sizes) |

Interfaces (`WorldModel(obs, action, goal_emb) -> dict`) stay identical. Phase 0's shape tests still work with updated config.

---

## Section 6 — Training Pipeline

### 6.1 Training step structure

```
1. Sample batch from replay buffer
   → (B, k+1) trajectory windows: states[t], actions[t..t+k-1], rewards[t..t+k-1]

2. Encode initial state
   s_0 = h(states[0])

3. K-step latent rollout
   For t in 0..k-1:
       s_t+1, obs_pred[t+1], r_pred[t+1] = g(s_t, actions[t])
       pi_pred[t], v_pred[t] = f(s_t, goal_embeddings[t])

4. Compute joint loss
   L_obs       = sum_t XE/BCE(obs_pred[t+1], states[t+1])
   L_value     = sum_t MSE(v_pred[t], mc_return[t])
   L_policy    = sum_t XE(pi_pred[t], actions[t])
   L_reward    = sum_t MSE(r_pred[t+1], rewards[t+1])
   L_consist   = sum_t MSE(s_t+1, h(states[t+1]).detach())

   L_total = 1.0·L_obs + 0.25·L_value + 1.0·L_policy + 0.5·L_reward + 0.1·L_consist

5. Backward + optimizer step
   L_total.backward()
   clip_grad_norm_(model.parameters(), 1.0)
   optimizer.step()
   lr_scheduler.step()
   ema.update(model)

6. Logging (W&B + JSONL)
```

K-unroll default: **k=5**. >5 destabilizes (compounding latent errors); <3 under-supervises dynamics.

### 6.2 Placeholder targets

**Policy targets:** `target_pi[t] = one_hot(actions[t], num_classes=9)`. Behavioral cloning on PPO.

**Value targets:** `target_v[t] = sum(γ^i * rewards[t+i] for i in range(remaining_episode_length))`. γ=0.997. Computed once at extraction, stored in replay buffer.

**Reward targets:** `target_r[t+1] = rewards[t+1]`. v2's shaped reward.

**Goal embedding for placeholder training:** A single learned dummy goal embedding `DUMMY_GOAL = nn.Parameter(torch.randn(384) * 0.02)` for all training samples. Phase 1b replaces with goal sampling from MCTS-derived targets.

### 6.3 Replay buffer sampling

```python
def sample_batch(self, batch_size: int, k_unroll: int) -> Batch:
    indices = self._weighted_choice(batch_size)
    windows = [self._read_window(i, k_unroll + 1) for i in indices]
    valid = [w for w in windows if w is not None]
    while len(valid) < batch_size:
        extra = self._weighted_choice(batch_size - len(valid))
        valid.extend(w for w in (self._read_window(i, k_unroll + 1) for i in extra) if w)
    return collate(valid[:batch_size])
```

Episode-boundary handling: windows spanning `done=True` rejected and resampled.

Decoding: msgpack → GameState → tokenizer → tensor batch. ~16ms per batch of 32 with k=5; small fraction of ~50ms forward+backward.

Phase 1a priority weights: all entries 0.3 (demo source). Phase 1b adds divergence (1.0), success (0.7), exploration (0.5). Architecture supports adding sources via one-line change.

### 6.4 Checkpointing

```python
def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler,
                    step, ema, replay_buffer_position, rng_state, wandb_run_id):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "ema": ema.shadow_state_dict() if ema else None,
        "replay_buffer_position": replay_buffer_position,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "wandb_run_id": wandb_run_id,
        "config": asdict(config),
    }, path)
```

**Cadence:** every 5,000 steps + at end of each epoch. Keep 5 most recent + every 100K-step milestone. ~600 MB per checkpoint × 10 = ~6 GB on disk.

**Resume contract:** `train.py --resume runs/wm_v0/latest.pt` re-establishes every piece of state. Next training step bit-equivalent (modulo CUDA non-determinism).

### 6.5 Eval-during-training

```python
@torch.no_grad()
def run_eval(model, eval_buffer, batch_size=64, max_batches=20) -> dict:
    metrics = defaultdict(list)
    model.eval()
    for batch in eval_buffer.iter_batches(batch_size, k_unroll=1, limit=max_batches):
        s_0 = model.h(batch.states[0])
        s_1, obs_pred, r_pred = model.g(s_0, batch.actions[0])
        metrics["acc/map_id"].append((obs_pred.map_id.argmax(-1) == batch.states[1].map_id).float().mean().item())
        # ... per-field accuracies for x, y, party_species, etc.
        metrics["mse/reward"].append(F.mse_loss(r_pred, batch.rewards[1]).item())
    model.train()
    return {k: float(np.mean(v)) for k, v in metrics.items()}
```

**Cadence:** every 2,000 training steps + at every checkpoint. ~1 second per eval run. Phase 1a definition-of-done gate: `acc/map_id ≥ 0.80`, `acc/x ≥ 0.80`, `acc/y ≥ 0.80`, `acc/party_species_slot_0 ≥ 0.80`.

### 6.6 W&B integration

```python
class WandbLogger:
    """No-op when WANDB_API_KEY env var is unset; logs everything when set."""
    def __init__(self, project, run_name, config, run_id=None):
        if "WANDB_API_KEY" not in os.environ:
            self.enabled = False
            return
        import wandb
        self.run = wandb.init(project=project, name=run_name, config=config,
                              id=run_id, resume="allow")
        self.enabled = True

    def log(self, metrics, step): ...
    def finish(self): ...
```

**At training time** (every step or every 100):
- `loss/total`, `loss/obs`, `loss/value`, `loss/policy`, `loss/reward`, `loss/consist`
- `optim/lr`, `optim/grad_norm`, `optim/loss_scale`
- `throughput/steps_per_sec`, `throughput/samples_per_sec`
- `replay/buffer_size`, `replay/priority_distribution_*`

**At eval cadence:**
- All `acc/*` and `mse/*`
- `eval/passes_doD_gate` (boolean)

**At checkpoint save:** optionally save checkpoint as W&B artifact.

Run name: `phase1a-wm-{timestamp}-{git_short_sha}`. Run ID persisted in checkpoint for resume continuity.

### 6.7 `go_forever` wrappers

Bash (`go_forever.sh`) and PowerShell (`go_forever.ps1`) variants. Both:
- While true: launch `python scripts/train.py` (with `--resume` if a checkpoint exists)
- On crash, sleep 10s and restart
- On clean exit (training completes), exit cleanly

Mirrors v2/go_forever.sh in spirit. Windows variant uses `Get-ChildItem | Sort-Object LastWriteTime`. Both pipe stdout/stderr to a rotating log.

### 6.8 Configuration

Single `world_model/configs/phase_1a.yaml` holds every hyperparameter — model, training, eval, replay, checkpoint, W&B. Read once at training start, immutable per run, saved into checkpoint for sanity-checking on resume.

### 6.9 Expected runtime

| Metric | Estimate |
|---|---|
| Forward + backward per step | 80–150 ms |
| Steps per hour | ~30K |
| Steps to ≥80% accuracy gate | 100K–300K (estimated; first run will calibrate) |
| Wall-clock to gate | 4–10 hours |
| Steps in 24h | ~700K (~2 epochs over 1M decisions) |

Plateau below gate after 24h → tune (smaller model, larger batch, different LR), document in `tuning.md`.
Hit gate in <8h → save milestone checkpoint, declare Phase 1a done early.

---

## Section 7 — Implementation Order, Milestones, Risks

### 7.1 Dependency-ordered implementation groups

```
Group A (independent — start in parallel):
├── Replay buffer infrastructure
├── Real tile-collision extraction
└── Bootstrap data extraction (long-running; runs in background)

Group B (depends on A):
├── Typed-field tokenizer
└── Full transformer architecture (h, g, f)

Group C (depends on B):
├── Joint loss with placeholder targets
└── Training pipeline scaffolding

Group D (depends on C):
├── W&B integration
├── Eval-during-training
├── Checkpoint/resume infrastructure
└── go_forever wrappers

Group E (the gate):
├── First full training run
├── Validation against ≥80% accuracy gate
└── STATE.md / progress.md / tuning.md updates
```

Within an inline-execution session work runs serially, but dependencies allow some background work (e.g., bootstrap extraction overnight while building the architecture).

### 7.2 Milestones

| Milestone | Duration | Definition of done |
|---|---|---|
| **M1: Foundations** | Week 1 | Replay buffer round-trips. Real tile-collision extracts non-zero codes. Bootstrap extraction runs end-to-end on one v2 checkpoint. |
| **M2: Bulk data + architecture** | Week 2 | Bootstrap data extraction running across all 4 v2 checkpoints (~24h compute). Tokenizer implements all 44 token types. Full transformer h/g/f instantiates and forward-passes shape-correctly. |
| **M3: Training pipeline** | Week 3 | Joint loss computes without NaN on real replay batch. Training-step skeleton runs. Checkpoint round-trip preserves all state. |
| **M4: Production infra** | Week 3–4 | W&B logger no-ops cleanly when unset. Eval-during-training computes per-field accuracies. go_forever scripts survive a kill-and-restart. |
| **M5: First training run + validation** | Week 4–5 | First 24h training run completes. Per-field accuracies recorded. Either ≥80% gate passes (Phase 1a closed) or tuning iterations documented. |

Best case: ~3 weeks (gate hit on first try). Realistic: 4–5 weeks with 1–2 tuning iterations.

### 7.3 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tile-collision tables have typos | High | Manual visual verification step in M1: render extracted matrix alongside PyBoy screen. |
| Transformer doesn't converge in 24h | Medium | Conservative ~67M starting size. LR warmup. EMA shadow weights. If still flat: shrink to 30M, retry. |
| Per-field loss imbalance | High | Per-field loss weights tunable. First run uses uniform; eval reveals bottlenecks; iterate. |
| Replay buffer disk usage exceeds 10GB | Low | Compression at write (Parquet snappy/zstd). Free-space check at extraction start. |
| W&B resume doesn't preserve continuity | Medium | Test resume flow explicitly in M4. Fall back to "run X continues run Y" marker if `wandb.init(resume="allow")` fails. |
| Placeholder targets actively hurt f-head | Medium | f-head is small (~7M); Phase 1b retraining converges in ~10K steps. Document in ADR-0002. |
| Bootstrap extraction crashes mid-run | Medium | Episode-level checkpointing; resume via `meta.json`. |
| PPO checkpoint won't load under our env | Medium | Use v2's env directly for PPO; observe through our `read_state` separately. Bypasses compatibility issue. |
| k_unroll=5 destabilizes training | Medium | Standard MuZero practice. If diverges: drop to k=3. |
| Scope creep into Phase 1b | High | Hard rule: no MCTS, no real goal-conditioned training, no verification flywheel until M5 closes. New ideas → ADR or "Phase 1b backlog" in STATE.md. |

### 7.4 Explicitly *not* in Phase 1a

- MCTS — Phase 1b
- Real goal-conditioned training (replacing placeholder targets) — Phase 1b
- Verification flywheel — Phase 1b
- Hierarchical planner — Phase 1c
- Cost estimator — Phase 1c
- Endgame data via targeted exploration — Phase 1d
- Goal interface CLI/REPL — Phase 1c
- Full memory observation — Phase 2
- Glitch dynamics / Mew — Phase 3
- Stochastic chance nodes — Phase 1b
- Save-state branching — Phase 1b
- Targeted exploration worker — Phase 1b
- Coverage tracking dashboard — Phase 1b

### 7.5 Repository changes summary

After Phase 1a closes, `world_model/` will have:

- ~12 new modules (replay buffer, tile collision, tokenizer, full arch, losses, training pipeline, eval, checkpoint, W&B logger, goal embedding, two CLI scripts)
- ~12 new test files
- Two new go-forever wrappers (bash + PowerShell)
- New `configs/phase_1a.yaml`
- New ADR-0002 + `docs/data/replay_format.md`
- Modified: `env.py` (real tile-collision), `arch.py` (full transformer), `world_model/__init__.py`, STATE.md, progress.md, tuning.md
- Gitignored: `data/replay_buffer/` (~5–10 GB), `runs/phase1a_wm/` (~6 GB at peak)

### 7.6 Closing properties

1. **Architecture interface stays stable.** Phase 1b can replace value/policy targets without touching tokenizer, encoder, or dynamics. Phase 1c can add hierarchy without touching the model. Phase 2 changes observation schema but tokenize-then-transformer is unchanged.

2. **Every Phase 1a output is independently useful.** Replay buffer reusable. Real tile-collision improves any future training. Bootstrap data reusable. Trained world model is the artifact, but each piece below has standalone value.

3. **Failure of M5 gate is acceptable.** Per-field accuracy diagnostics are themselves a learning artifact. Phase 1b's MCTS gives a tool to fix it (verification-driven retraining). Honest negative result is its own contribution to the personal-exploration framing.

---

## Appendix — Phase 1a at a Glance

| Question | Answer |
|---|---|
| What's the artifact? | Trained ~67M-param goal-conditioned world model |
| What does it predict? | Next state (per-field), reward, goal-conditioned policy + value |
| How is it trained? | MuZero-style joint loss with placeholder value/policy targets |
| Where does data come from? | All 4 v2 PPO checkpoints, ~250K decisions each |
| What's the gate? | ≥80% per-field accuracy on `map_id`, `x`, `y`, `party_species_slot_0` |
| What's deferred? | MCTS, goal-conditioned training, verification flywheel — all Phase 1b |
| What's the timeline? | 3–5 weeks, full-time, single 8–12 GB GPU |
| What changes from Phase 0? | ~370K param stub → ~67M trained world model; tile-collision real; replay buffer + training pipeline added |
