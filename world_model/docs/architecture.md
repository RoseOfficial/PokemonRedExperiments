# Pokemon Red Goal-Conditioned World-Model Planner — Design Spec

**Status:** Draft (pending user approval)
**Date:** 2026-05-07
**Author:** Brainstormed collaboratively via Claude Code
**Working directory:** `world_model/` (new top-level dir in this repo)

---

## Executive Summary

Build a goal-conditioned planner for Pokemon Red. The user invokes it with goals
like `catch(ODDISH)`, `beat_champion()`, or `catch_all()`, and the system
executes the optimal action sequence to satisfy that goal.

The system is **not** PPO-based reinforcement learning. It is a **MuZero-family
algorithm**: a learned world model + Monte Carlo Tree Search, with two
modifications — goal conditioning in the value/policy heads, and stochastic
chance nodes for in-game RNG.

Three layers of architecture:

1. **Goal Interface (Layer 3)** — Python DSL + curated knowledge base.
2. **Hierarchical Planner (Layer 2)** — hand-coded decomposition of complex
   goals into ordered atomic subgoals.
3. **World Model + MCTS (Layer 1)** — the learned core, ~100M params,
   transformer-based, runs on a single 8–12GB consumer GPU.

A verification + retraining flywheel takes the system from
"demonstration-data competent" to "Pokemon Red competent" by feeding every
model-vs-reality divergence back into prioritized training.

The project lives in a new `world_model/` directory; `v2/` and `baselines/`
are untouched and remain as PPO comparison baselines.

---

## Constraints and Goals

| Property | Value |
|---|---|
| Hardware | Single consumer GPU, 8–12GB VRAM |
| Time commitment | Full-time |
| Success criterion | Personal exploration — learn the field deeply; revolutionary outcomes are stretch goals |
| LLM dependency | None at runtime; tiny shipping LMs allowed only if they earn their keep |
| Algorithm family | MuZero variant (goal-conditioned, stochastic) — explicitly *not* PPO |
| Code location | `world_model/` (new top-level dir); `v2/` and `baselines/` left intact |
| Phase progression | Phase 1 (RAM-derived state) → Phase 2 (full WRAM observation) |

User-facing capability target: arbitrary goals expressible in a small Python
DSL, including atomic (`catch(ODDISH)`), compositional
(`then(beat(BROCK), beat(MISTY))`), and completionist (`catch_all()`).

---

## Section 1 — High-Level Architecture

Three layers with well-defined interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: GOAL INTERFACE                                    │
│  - Goal DSL (Python combinators: catch, beat, then, forall) │
│  - Knowledge base (per-Pokemon affordances, gym order, etc.)│
│  - User-facing CLI / Python API                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ (parsed goal tree + KB)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: HIERARCHICAL PLANNER                              │
│  - Decomposes complex goals into atomic subgoal queue       │
│  - Solves precedence-constrained routing (TSP-like)         │
│  - Re-plans on failure / model-reality divergence           │
└──────────────────────────┬──────────────────────────────────┘
                           │ (single executable atomic subgoal)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: WORLD MODEL + GOAL-CONDITIONED MCTS               │
│  - Learned dynamics (representation, dynamics, prediction)  │
│  - Goal-conditioned V(state, goal) and π(action|state, goal)│
│  - MCTS at inference; MuZero-style training                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ (action sequence)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  EXECUTION + VERIFICATION                                   │
│  - PyBoy emulator (real game)                               │
│  - Records (state, action, predicted, actual)               │
│  - Detects divergence; feeds retraining                     │
└─────────────────────────────────────────────────────────────┘
```

**Why three layers:**
- Two layers (planner-less MCTS) collapses on long-horizon goals — search
  cannot reach 100,000-step horizons.
- Four layers (e.g., adding a learned meta-controller above the planner) is
  premature. Hand-coded decomposition + KB suffices for v1.
- Each layer is independently testable: planner against real PyBoy with
  stubbed model; MCTS on hand-crafted subgoals with stubbed planner; goal
  parser as pure unit tests.

---

## Section 2 — Goal Specification Language

A small Python-first DSL — no parser, no LM. Goals are composable Python
objects the planner introspects.

```python
from pokemon_planner import goals as g

# Atoms
g.catch(g.ODDISH)              # have Oddish in party or boxes
g.beat(g.LANCE)                # defeated trainer
g.reach(g.PEWTER_CITY)         # current map matches
g.have_item(g.BICYCLE)         # bag contains
g.level(g.CHARIZARD, 50)       # any party member of species ≥ level
g.evolve(g.GROWLITHE, g.ARCANINE)

# Combinators
g.then(g.beat(g.BROCK), g.beat(g.MISTY))
g.and_(g.catch(g.PIKACHU), g.catch(g.RAICHU))
g.forall(g.ALL_KANTO_GRASS, lambda p: g.catch(p))
g.or_(g.catch(g.HITMONLEE), g.catch(g.HITMONCHAN))

# Top-level shortcuts
g.catch_all()                  # forall(ALL_DEX, catch)
g.beat_champion()              # then(beat each gym leader, beat E4, beat champ)
g.complete_pokedex()           # catch_all + dex-flag check

# Optimization criteria (default: minimize emulator frames)
g.beat_champion(minimize=g.Frames)
g.beat(g.LANCE, minimize=g.Damage)
```

Each goal compiles to:

1. A **predicate** — `bool(state) → goal achieved?`
2. A **decomposition hint** — `Goal → list[Goal]` for composites
3. An **affordance set** — relevant KB entries (encounter zones, prerequisites)

**Knowledge base** is data, not code. YAML files under `world_model/pokemon_planner/kb/`:

| File | Contents |
|---|---|
| `species.yaml` | Per-Pokemon: encounter zones, methods (wild/gift/trade), required badges, version-exclusivity, evolution requirements |
| `trainers.yaml` | Gym leaders, rivals, route trainers; teams; locations |
| `items.yaml` | TMs, key items, sources, prices |
| `regions.yaml` | Map graph, badge gates, blockers (Snorlax, Cut tree, Surf water) |

KB sourced once from `pret/pokered` disassembly + community resources.
Maintenance is hours of work, not days.

**Edge cases the KB flags:**
- Mew is uncatchable without trainer-fly glitch (Phase 2).
- Trade evolutions (Alakazam, Machamp, Golem, Gengar) marked `requires-trade`;
  v1 skips or substitutes pre-evolution.
- Choice exclusivity (Hitmonlee/Hitmonchan, starter trio) encoded as XOR groups.
- Version exclusives — v1 targets Red specifically.

---

## Section 3 — Hierarchical Planner (Layer 2)

Hand-coded rules over the KB. Takes a parsed goal tree, produces an ordered
queue of atomic subgoals each plausibly solvable in a single MCTS horizon.

### Decomposition by goal type

| Goal type | Decomposition strategy |
|---|---|
| Atom (`catch`, `beat`, etc.) | Expand prerequisites recursively from KB |
| `then(g1, g2, …)` | Decompose each in order, concatenate |
| `and_(g1, g2, …)` | Optimize ordering via precedence-constrained routing |
| `forall(set, fn)` | Compile to `and_([fn(x) for x in set])` |
| `or_(g1, …)` | Pick cheapest by cost estimate, decompose just that branch |

### Components

1. **Affordance resolver** — atom + state → executable subgoal chain via KB.
   `catch(ODDISH)` → `[reach(ROUTE_24), encounter_until(ODDISH), catch_in_battle()]`.
   Recursive: `reach(ROUTE_24)` itself expands per its prerequisites.
2. **Cost estimator** — `(subgoal, state) → estimated_frames`. Heuristic
   initially (zone distance + average encounter rate); empirical over time.
3. **Route optimizer** — for `and_` / `forall`, computes ordering that
   minimizes total estimated cost. Greedy nearest-neighbor + 2-opt local
   search. Not optimal, but world-model uncertainty dwarfs ordering
   suboptimality.
4. **Subgoal queue** — worklist; pop, hand to Layer 1, record outcome.
5. **Re-planner** — on failure, decides: retry with bigger search budget,
   insert remediation (`heal()`, `buy(POKEBALL, qty=10)`), skip, or abort.

### Core loop

```python
def plan_and_execute(goal, kb, world_model):
    queue = decompose(goal, kb, current_state())
    while queue:
        sub = queue.pop_front()
        if sub.predicate(current_state()):
            continue
        plan = mcts_search(world_model, current_state(), sub, budget=N)
        for action in plan:
            step_pyboy(action)
            if model_divergence_detected():
                log_divergence(); break
            if sub.predicate(current_state()):
                break
        if not sub.predicate(current_state()):
            handle_failure(sub, queue, kb)
    return summarize()
```

### Why hand-coded vs. learned

The decomposition structure ("to catch X, reach zone Y with badge Z") is human
knowledge already captured in the game. Re-discovering it via RL is wasteful.
Hand-coded rules are debuggable. Phase 4-ish ambition could replace this layer
with a learned high-level policy; the interface (goal in, subgoal queue out)
is stable, so the rest of the system is unaffected.

### Resource handling

Reactive: detect resource exhaustion during execution, insert remediation
subgoals (`heal()`, `buy(POKEBALL, qty=10)`), resume. Proactive resource
planning explodes search space; deferred indefinitely.

### Failure budget

Each subgoal: N retries (default 3) with escalating MCTS budget. After N,
invoke targeted exploration or mark unreachable in this run.

---

## Section 4 — World Model + Goal-Conditioned MCTS (Layer 1)

The core ML component. Learns dynamics; provides goal-conditioned value/policy
estimates for MCTS. MuZero-flavored, with explicit state prediction (for
verification) and goal conditioning baked into value/policy heads.

### 4.1 State representation (Phase 1)

Curated subset of WRAM, ~660 bytes per step:

| Field | Size | Notes |
|---|---|---|
| Position (`map_id`, `x`, `y`) | 3 B | |
| Party slots × 6 (`species`, `level`, `hp_cur`, `hp_max`, `status`, top-4 moves) | ~80 B | |
| Bag (top 20 items × `(item_id, qty)`) | 40 B | |
| Badges (8 bits) | 1 B | |
| Event flags (curated subset of `0xD747`–`0xD87E`) | 256 B | |
| Money (BCD), time-played | 7 B | |
| Battle state (opp species/level/HP, turn) | 8 B | Zeroed when not in battle |
| Tile collision (16×16 around player) | 256 B | Navigation grounding |
| Menu / dialogue / mode flags | 4 B | |

Each field gets its own learned embedding — not raw byte tokenization. ~50
typed slots, not ~1024 byte tokens.

**Phase 2:** observation graduates to full WRAM (~8KB). Larger model needed;
state schema migration documented as ADR.

### 4.2 Architecture

Three networks, MuZero-style:

- **`h(obs) → s`** — representation. Transformer encoder, ~30M params, output
  256-d latent.
- **`g(s, a) → (s', ŝ_obs', r̂)`** — dynamics. Predicts next latent +
  next observation + scalar reward (goal-progress signal). Explicit
  observation prediction enables verification.
- **`f(s, goal_emb) → (π, V)`** — prediction. Goal-conditioned policy prior
  and value.

Total: ~100M parameters. fp16 + gradient checkpointing through the dynamics
unroll. Fits 8–12GB.

### 4.3 Goal conditioning

Atomic goals (post-planner-decomposition) compile to:

```
goal_emb = concat(
    embed_predicate_type[Catch | Beat | Reach | HaveItem | Level | Evolve],
    embed_entity[species_id | trainer_id | map_id | item_id]
)
```

Concrete: `catch(ODDISH)` → `concat(embed_predicate[CATCH], embed_species[ODDISH])`.
This embedding scheme generalizes — `catch(ODDISH)` and `catch(BELLSPROUT)`
share most representation.

Composite goals never reach this layer; the planner has already atomized them.

### 4.4 Training objectives

Joint loss:

```
L = λ_obs · L_obs        # next-observation prediction (BCE / MSE per field)
  + λ_value · L_value    # value matches Monte Carlo / n-step returns
  + λ_policy · L_policy  # policy matches MCTS visit distribution
  + λ_reward · L_reward  # predicted r̂ matches actual goal-progress signal
  + λ_consist · L_consist # latent at t+1 matches dynamics prediction from t
```

K-step unrolling (default K=5): dynamics function unrolled forward 5 steps,
losses computed at each step. Forces dynamics coherence over horizons longer
than a single step.

### 4.5 Training data sources (priority order)

1. PWhiddy's pretrained checkpoint at `baselines/session_4da05e87_main_good/`.
   ~1M demonstration steps, fast.
2. Existing v2 checkpoints (`v2/runs/poke_*_steps.zip`).
3. Random play with auto-mash through dialogues.
4. TAS recordings from speedrun.com / TASvideos for endgame coverage.
5. Targeted exploration (Section 6.5).

Replay buffer: ~10M steps × ~1KB/step ≈ 10GB on disk.

### 4.6 Data coverage and targeted exploration

The world model can only predict in regions seen in training. Coverage tracking:

- Cluster state space (initially: `map_id × party_size × badges`, ~10K
  buckets; later: k-means on learned latents).
- Per cluster: visit count, model predictive variance, last-touched timestamp.
- Surface low-coverage clusters in the dashboard.

Targeted exploration loop (Section 6.5) launches episodes from save states
in low-coverage clusters with curiosity-driven actions.

### 4.7 VRAM budget (8–12GB target)

| Component | VRAM |
|---|---|
| Model weights (fp16) | ~200 MB |
| Optimizer states (Adam, fp32 master) | ~800 MB |
| Activations (K=5 unroll, batch=64) | ~3–5 GB |
| Replay sampling | ~1 GB |
| MCTS during eval | ~1 GB |
| Headroom | ~2 GB |

Tight on 10GB, comfortable on 12GB. Knobs if over budget: smaller batch,
shorter K-unroll, smaller transformer dimension.

---

## Section 5 — Goal-Conditioned MCTS

MuZero MCTS with three Pokemon-specific adaptations: goal-conditioned
value/policy, stochastic-dynamics chance nodes, and shaped goal-progress
signal.

### 5.1 Per-action algorithm

```
1. SELECTION — descend tree from root via PUCB:
       a* = argmax_a [ Q(s,a)
                     + c_puct · π(a|s, goal) · √Σ_b N(s,b) / (1 + N(s,a)) ]

2. EXPANSION — at leaf, apply g(s, a) → (s', ŝ_obs', r̂); compute (π, V)
   from f(s', goal); cache new node.

3. EVALUATION — V from prediction head bootstraps; no rollout.

4. BACKUP — propagate discounted return up the path:
       Q(s,a) ← (N · Q + (r̂ + γ · V_subtree)) / (N + 1)
```

### 5.2 Stochastic dynamics

Pokemon RNG (encounter slots, accuracy, crits, damage rolls) requires explicit
chance nodes. Three options:

| Approach | Cost | Fidelity |
|---|---|---|
| Pretend deterministic | Cheapest | Low — verification catches divergence |
| **Stochastic MuZero (chance nodes)** | ~5× compute | **High — recommended default** |
| Save-state branching | Real PyBoy compute | Ground truth — reserved for high-stakes RNG |

Default: stochastic MuZero. Save-state branching reserved for legendary
captures, boss-fight crit gambles.

### 5.3 Goal-progress signal

Sparse predicate goals starve MCTS. Shaped per-step `r̂` from three sources:

1. KB-derived heuristic distance (Manhattan to encounter zone, etc.).
2. Learned progress predictor — `(state, goal) → estimated_steps_remaining`.
3. Predicate-near indicator — "is this state on a known successful trajectory
   toward this goal?"

`r̂` head trained on `(progress_t+1 - progress_t)` so backup accumulates
correctly.

### 5.4 Search budget

| Goal scale | Sims/action | Wall clock |
|---|---|---|
| Easy navigation | 50 | ~50 ms |
| Wild catch | 200 | ~200 ms |
| Gym battle | 500 | ~500 ms |
| Glitch sequence (Phase 2) | 2000 | ~2 s |

**Early-stop:** predicate satisfied at any tree node → terminate simulation,
back up max-value.

**Adaptive budget:** top action visit share > 80% → cut search early.

### 5.5 Pokemon-specific tweaks

- Action mask (no-ops pruned: `SELECT`, `UP` against wall).
- Repeated-state detection (hash latents; loops within a simulation
  terminate).
- Frame-skipping (~24 frames per action via existing v2 env).
- Action filtering for menu/dialogue states.

### 5.6 How long-horizon goals work

The planner has decomposed `beat_champion()` into ~50 sequential atomic
subgoals. Each is short-horizon (1K–10K steps); MCTS only ever solves
short-horizon problems. Long-horizon competence emerges from composition,
not from any single deep search.

---

## Section 6 — Verification + Data Flywheel

The engine that takes the system from demo-data competent to Pokemon Red
competent.

### 6.1 Flywheel structure

```
World Model + MCTS  →  proposes plan
        ↓
PyBoy execution     →  records (state, action, predicted, actual)
        ↓
Divergence detector →  prioritized labels
        ↓
Replay buffer       →  priority-weighted training batches
        ↓
WM training step    →  sharpens dynamics
        ↓
(loop)
```

Three properties make this a flywheel rather than a generic training loop:
1. Search drives data collection — model gets divergence labels for regions
   the user's goals require.
2. Divergence amplifies signal — wrong (state, action) pairs oversampled next
   training step.
3. Goal-driven prioritization — hard goals produce more divergences;
   improvement concentrates where most needed.

### 6.2 Verification step

```python
def execute_with_verification(plan, world_model, pyboy, subgoal):
    state = read_state(pyboy)
    for action in plan:
        next_pred, r_pred = world_model.dynamics(state.latent, action)
        actual = step_pyboy(pyboy, action)
        if divergence(actual, next_pred) > THRESHOLD:
            replay_buffer.add((state, action, next_pred, actual),
                              priority=PRIORITY_DIVERGENCE)
            return ABORT
        if subgoal.predicate(actual):
            replay_buffer.add((state, action, actual, terminal=True),
                              priority=PRIORITY_SUCCESS)
            return SUCCESS
        state = actual
    return PLAN_EXHAUSTED
```

### 6.3 Divergence metric

Field-weighted L1 over the structured state vector:

| Field | Tolerance |
|---|---|
| `map_id`, `position` | Exact match |
| `event_flags`, `bag_items`, `tile_collision` | Exact match |
| `party_hp` | ±2 HP (RNG damage rolls) |
| `battle_state.opp_hp` | ±5% |

Threshold tuned so genuine RNG variance doesn't trigger. Initially
conservative; relaxed as model proves itself.

### 6.4 Prioritized replay buffer

| Source | Priority |
|---|---|
| Divergence (model wrong) | 1.0 |
| Successful subgoal completion | 0.7 |
| Targeted exploration rollouts | 0.5 |
| Bootstrap demos (PPO checkpoints) | 0.3 |
| Random play | 0.1 |

Sampling probability ∝ priority. Each batch is a mix.

### 6.5 Targeted exploration

When coverage tracker flags low-coverage cluster:

1. Find save state landing in or near that cluster.
2. Load into separate PyBoy instance.
3. Run curiosity-driven rollout (RND-bonus, NGU episodic novelty, or
   biased-random).
4. Add to replay buffer at priority 0.5.

Runs in parallel with main goal-execution. One worker per spare CPU core.

### 6.6 Save-state branching for RNG-critical decisions

For high-stakes RNG (legendary captures, boss-fight crit gambles):

1. Detect RNG-critical subgoal (KB-tagged).
2. Save PyBoy state immediately before RNG roll.
3. Try action; succeed → continue. Fail → load save, try different action
   sequence.

"Save scumming" — humans do it too. Reserved for cases where retry-from-scratch
is prohibitive.

### 6.7 Training schedule

| Loop | Trigger | Purpose |
|---|---|---|
| Online updates | Every 4 plans completed | Fast adaptation to recent divergences |
| Background batch | Every 30 minutes | Larger batch, broader sampling, stabilizes weights |

Both update the same weights. Online biases toward priority-1; background
samples uniformly to prevent catastrophic forgetting.

### 6.8 Logging

Per run, persist:

- **Divergence log** — every model-vs-reality mismatch (state, action,
  predicted, actual).
- **Coverage heatmap** — per-cluster visit count + predictive variance.
- **Goal completion log** — every goal attempted, decomposition, outcome,
  frames spent.
- **Search statistics** — average tree depth, top-action visit share.
- **Replay buffer composition** — fraction per priority class over time.

These are the artifact of "learn the field deeply" — the dashboard that
explains what the system is doing and why.

---

## Section 7 — Goal Interface

User-facing surfaces: Python API, CLI, interactive REPL.

### 7.1 Python API

```python
from pokemon_planner import Planner, goals as g

planner = Planner(
    rom_path="../PokemonRed.gb",
    world_model_path="runs/wm_latest.pt",
    knowledge_base_path="kb/",
)

result = planner.execute(
    goal=g.catch(g.ODDISH),
    start_state="../init.state",
    time_limit_hours=2,
)
print(result.summary())

# Compositional + chained
result = planner.execute(
    goal=g.then(g.beat(g.BROCK), g.catch(g.BELLSPROUT), g.beat(g.MISTY)),
    start_state=result.end_state,
)

# Long-horizon
result = planner.execute(
    goal=g.beat_champion(),
    time_limit_hours=24,
    checkpoint_every_minutes=30,
)
```

`Planner` is stateful — owns world model, KB, replay buffer, PyBoy. Multiple
`execute()` calls share the world model and improve it via the flywheel.

### 7.2 CLI

```bash
poke-plan run --goal "catch(ODDISH)"
poke-plan run --goal "then(beat(BROCK), catch(BELLSPROUT))"
poke-plan run --goal-file goals/elite_four_speedrun.py
poke-plan run --goal "catch_all" --time-limit 48h --start-state init.state
poke-plan run --goal "beat_champion" --resume runs/exec_042/
poke-plan goals list
poke-plan plan --goal "catch_all" --dry-run    # decomposition without execution
```

`--dry-run` outputs the planner's decomposition + cost estimate without
spinning up PyBoy. Critical for sanity-checking before long runs.

### 7.3 Interactive REPL

```
$ poke-plan repl --start-state init.state
[loaded WM from runs/wm_latest.pt — 47M states trained]
[loaded KB — 151 species, 39 trainers, 247 items]
[PyBoy initialized at init.state — Pallet Town, party=NIDORAN♂ L5]

> catch(ODDISH)
[Planning ...]
  Decomposed into: reach(ROUTE_5) → encounter(ODDISH) → catch_in_battle()
  Cost estimate: ~14,000 frames (24 game-min)
[Executing ...]
  ✓ reach(ROUTE_5)            done in 8,432 frames
  ✓ encounter(ODDISH)         done in 4,891 frames (RNG retries: 3)
  ✓ catch_in_battle()         done in 964 frames
[ACHIEVED catch(ODDISH) in 14,287 frames — 1 divergence logged]

> party
NIDORAN♂  L8   25/25 HP
ODDISH    L9   29/29 HP
```

Live status: current subgoal, MCTS visit counts, divergence flags. Ctrl-C
interrupts with save/resume/abort options.

### 7.4 Goal files

Reusable, version-controllable:

```python
# goals/elite_four_speedrun.py
from pokemon_planner import goals as g

GOAL = g.then(
    *[g.beat(leader) for leader in g.KANTO_GYM_LEADERS],
    g.beat(g.LORELEI), g.beat(g.BRUNO),
    g.beat(g.AGATHA), g.beat(g.LANCE), g.beat(g.CHAMPION),
)

CONSTRAINTS = dict(
    minimize=g.Frames,
    forbid_glitches=True,
    party_size_max=3,
)
```

### 7.5 Output artifacts

Every `execute()` produces `runs/exec_<N>/`:

```
runs/exec_042/
├── goal.txt              # serialized goal
├── plan.json             # decomposition + cost estimates
├── execution.log         # step-by-step subgoal results
├── divergences.jsonl     # one mismatch per line
├── mcts_stats.jsonl      # tree statistics per real action
├── trajectory.parquet    # full (state, action) tuples
├── start.state, end.state
├── replay.mp4 (optional)
└── SUMMARY.md            # one-page natural-language summary
```

### 7.6 Out of scope for v1

- Web UI
- Natural-language goal parser (DSL covers everything in scope)
- Multi-instance parallel execution sharing the world model

---

## Section 8 — Evaluation

### 8.1 Three evaluation surfaces

| Surface | Purpose | Cadence |
|---|---|---|
| Goal benchmarks | "Does it achieve user-facing objectives?" | After every retrain |
| Per-component metrics | "Which layer is the bottleneck?" | Continuous |
| Coverage dashboards | "What does it know vs. not know?" | Continuous |

### 8.2 Goal benchmarks (four tiers)

**T1 — Early-game atoms (Phase 1 milestone):**
```
[reach(VIRIDIAN_CITY), reach(PEWTER_CITY), catch(PIDGEY), catch(RATTATA),
 catch(WEEDLE), beat(YOUNGSTER_ROUTE_2_1), beat(BROCK), buy(POTION, qty=5),
 level(party_lead, 10)]
```
Pass: ≥8/9 within 15K frames each.

**T2 — Mid-game compositional (Phase 1 graduation):**
```
[then(beat(BROCK), beat(MISTY)),
 then(reach(VERMILION_CITY), beat(SURGE)),
 catch_n_distinct(20),
 evolve(CHARMANDER, CHARMELEON),
 then(catch(BELLSPROUT), catch(ODDISH), catch(WEEPINBELL)),
 have_item(BICYCLE),
 reach(CELADON_DEPT_STORE)]
```
Pass: ≥6/7 within 60K frames each.

**T3 — Endgame and completionist (Phase 1 endpoint / Phase 2 entry):**
```
[beat_champion(),
 catch_n_distinct(100),
 catch_all_starters(),
 speedrun(beat_champion, max_frames=2.5M)]
```
Pass: any successful `beat_champion()` is the gate to Phase 2.

**T4 — Phase 2 (full memory):**
```
[catch(MEW),                       # via trainer-fly glitch
 catch_all(),
 rediscover_known_glitch(MISSINGNO)]
```
Pass: at least one Phase-2-only goal succeeds.

### 8.3 Per-component metrics

| Component | Metric | Healthy band |
|---|---|---|
| `h` (representation) | Reconstruction error on held-out | Drops monotonically, plateaus < threshold |
| `g` (dynamics) | Per-field next-state accuracy | >95% common fields, >80% event flags |
| `f` (prediction) | Value MSE; policy KL | Both decrease; KL ~0.1–0.5 |
| Planner | Decomposition correctness on test goals | 100% on fixed test set of 30 goals |
| MCTS | Mean tree depth; root visit share at end | >10 average; >50% (decisive) |
| Verification | Divergence rate | 20–40% initially, <5% with mature flywheel |

### 8.4 Baselines

- **PPO floor** — PWhiddy's pretrained checkpoint, where goals overlap.
- **Random floor** — sanity check.
- **TAS ceiling** — published TAS for `beat_champion`, glitch sequences.

### 8.5 Continuous regression checks

Before any model retrain commits weights, run:
```
[reach(VIRIDIAN_CITY), catch(PIDGEY), beat(BROCK)]
```
Failure → reject new weights, rollback. Catches catastrophic forgetting from
prioritized replay. Set grows over time as new capabilities solidify.

### 8.6 Coverage dashboards

- **Coverage heatmap** — `map_id × badge_count` grid, colored by visit
  density and predictive variance.
- **Divergence timeline** — `(timestamp, map_id, severity)` rows.
- **Goal completion log** — every goal ever attempted, filterable.
- **Replay buffer composition** — priority class fractions over time.

These dashboards are the *learning artifact* for personal exploration —
more informative than any single metric.

---

## Section 9 — Repository Layout, Milestones, Risks

### 9.1 Directory layout

```
PokemonRedExperiments/
├── PokemonRed.gb                 # existing (gitignored)
├── init.state, etc.              # existing save states (reused)
├── baselines/                    # existing — left alone
├── v2/                           # existing — left alone, PPO baseline
└── world_model/                  # NEW — entire project
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── pokemon_planner/
    │   ├── env.py                # PyBoy wrapper + state extraction
    │   ├── state.py              # structured RAM state schema
    │   ├── kb/                   # YAML knowledge base
    │   ├── goals/                # DSL: dsl.py, atoms.py, named.py, compile.py
    │   ├── planner/              # decompose.py, route.py, cost.py, execute.py
    │   ├── world_model/          # arch.py, train.py, replay.py, checkpoint.py
    │   ├── search/               # mcts.py, chance.py, savestate_branch.py
    │   ├── flywheel/             # verify.py, divergence.py, coverage.py, explore.py
    │   ├── eval/                 # benchmarks.py, regression.py, metrics.py
    │   └── cli/                  # run.py, repl.py, plan.py
    ├── scripts/                  # bootstrap_demos.py, ingest_tas.py, visualize_runs.py
    ├── tests/                    # one file per surface + e2e_smoke
    ├── data/                     # gitignored: replay_buffer/, coverage/, divergences/
    ├── runs/                     # gitignored: per-execution artifacts
    ├── docs/
    │   ├── architecture.md       # this design doc (copy of canonical spec)
    │   ├── adr/                  # ADR-NNNN-<slug>.md
    │   ├── progress.md           # append-only narrative timeline
    │   ├── tuning.md             # append-only experiment table
    │   └── CURRENT_STATE.md      # auto-generated aggregated report
    ├── STATE.md                  # always-current project state (≤200 lines)
    └── CLAUDE_BOOTSTRAP.md       # session-start protocol for Claude
```

### 9.2 External dependencies

| Package | Purpose |
|---|---|
| `pyboy==2.4.0` | Game Boy emulation; same as v2/ |
| `torch` | NN; fp16 + gradient checkpointing |
| `numpy<2`, `pyyaml`, `pydantic` | Data plumbing (numpy<2 per Windows pin) |
| LightZero (likely fork) | MuZero implementation reference |
| `gymnasium` | Env interface compat |
| `tensorboard`, `wandb` (opt) | Metrics |
| `faiss-cpu` | Replay-buffer similarity for coverage |
| `pytest`, `ruff`, `mypy` | Hygiene |

### 9.3 Milestones (full-time, single 8–12GB GPU)

| Phase | Duration | Definition of done |
|---|---|---|
| **Phase 0 — Foundation** | Week 1–2 | Repo scaffolded, KB schema + 30 species filled, state extractor works, world-model stubs train without NaN, CI green |
| **Phase 1a — Bootstrap data + first WM** | Week 3–4 | ~1M demo steps from PPO checkpoints, world model trains 24h, prediction head >80% on `map_id`, `position`, `party_species` |
| **Phase 1b — First goals work** | Week 5–6 | T1 benchmark passes; verification loop end-to-end; first divergences logged and retrained |
| **Phase 1c — Mid-game + compositional** | Month 2 | T2 benchmark passes; compositional goals work; mid-game data via flywheel + targeted exploration |
| **Phase 1d — Endgame** | Month 3–4 | T3 benchmark: at least one successful `beat_champion()`; targeted exploration covers Elite Four; `catch_n_distinct(100)` works |
| **Phase 2 — Full memory observation** | Month 5–6 | Observation graduates to full WRAM; world model retrained; Phase 1 capabilities preserved (regression guard); T4 entry |
| **Phase 3 — Glitch + strategy discovery** | Month 6+ | T4: `catch(MEW)` via trainer-fly verifies; search rediscovers ≥1 known speedrun glitch; optional: novel exploit |

Slip factors: training compute is the main bottleneck (24h runs become 3 days
during iteration); endgame targeted exploration is most uncertain piece;
Phase 2 observation graduation is ~2 weeks of refactor.

### 9.4 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| World model doesn't converge | Medium | Smallest-possible model on smallest-possible state region first |
| VRAM exceeds budget | Medium | Knob list ready; cloud GPU only for milestone runs |
| Verification reveals fundamental dynamics gaps retraining can't fix | Medium | Targeted exploration; mark unreachable + document if intractable |
| Hierarchical planner produces suboptimal decompositions | High | Cost estimator improves over time; suboptimality is acceptable |
| Long-horizon goals compose poorly | Medium | Re-planner handles single failures; multi-failure triggers full re-decomposition |
| Scope creep into Phase 2 during Phase 1 | High | Hard rule: no Phase 2 work until T3 passes; new ideas as ADRs not implementations |
| Catastrophic forgetting from prioritized replay | Medium | Regression guard before every weight commit; background uniform-sampling loop |
| Whole approach wrong for Pokemon | Low–medium | Phase 1a is the early-validation gate; honest negative result is itself learning |

### 9.5 Explicitly out of scope for v1

- LLM components (any size, any role)
- Web UI / browser dashboard
- Multi-game generalization (Blue, Yellow, Crystal)
- Distributed / multi-GPU training
- Self-play
- Battle-only specialist policy
- Mechanistic interpretability
- Goal-conditioned skill *library* with named skills

---

## Section 10 — Observability for Agent-Assisted Development

The system is designed so that any Claude session (or fresh human reader) can
walk into the project cold and orient in under a minute by reading 2–3 specific
files.

### 10.1 Single entry point: `world_model/STATE.md`

Always-current. Updated after every significant change. Hard cap: 200 lines.
Standard schema:

```markdown
# Project State — <last-updated timestamp>

## Phase
<current phase>

## Working
- <bulleted current capabilities>

## Broken / Known Issues
- <issue> — <cause> — <fix in progress>

## Next Up
1. <next action>
2. ...

## Recent Changes (last 7 days)
- <date>: <change> (ADR-NNNN if applicable)

## Where To Look
- Architecture: docs/architecture.md
- Recent decisions: docs/adr/
- Last training run: data/training_runs/<latest>/SUMMARY.md
- Last 50 divergences: data/divergences/recent.md
- Test status: tests/LAST_RUN.md
```

A pre-commit hook or `update_state.py` script enforces the schema.

### 10.2 Architectural Decision Records: `docs/adr/NNNN-<slug>.md`

Sequentially numbered. Newest = highest. Never edited after acceptance —
superseded by a new ADR. Schema (~30–80 lines each):

```markdown
# ADR-NNNN: <title>

Status: Accepted | Superseded by ADR-MMMM | Proposed
Date: YYYY-MM-DD
Supersedes: <ADRs or "none">

## Context
<why this decision is needed>

## Decision
<what was decided>

## Consequences
<expected effects, both positive and negative>

## Alternatives considered
<options rejected and why>
```

### 10.3 Progress timeline: `docs/progress.md`

Append-only narrative. ~1–3 entries per week. Datestamped.

### 10.4 Tuning log: `docs/tuning.md`

Append-only Markdown table:
```
| Date | Run | Change | T1 pass rate | Notes |
```
Prevents re-suggesting things already tried.

### 10.5 Per-run summary: `runs/exec_<N>/SUMMARY.md`

One-page natural-language summary alongside structured artifacts (Section 7.5).
Greppable across runs.

### 10.6 Aggregated report: `docs/CURRENT_STATE.md`

Auto-generated by `scripts/generate_current_state.py`. 1–2 pages. Pulls from
STATE.md + recent ADRs + recent divergences + tests + training metrics.

### 10.7 Conventions for Claude-readability

1. **Stable paths** — STATE.md, docs/architecture.md, docs/adr/, etc., never
   move or rename.
2. **Bounded files** — STATE.md ≤200 lines, ADRs ≤80 lines, SUMMARY.md ≤80 lines.
3. **Stable schemas** — section orders and table columns don't change.
4. **English first, structure second** — each artifact starts with a 1-sentence
   headline.
5. **Cross-link by path, not name** — paths survive; names rot.
6. **JSONL for high-frequency events** — divergences, MCTS stats. Tail-able.
7. **Markdown for narrative** — STATE.md, ADRs, SUMMARY.md, progress.md.

### 10.8 Claude session bootstrap protocol

`world_model/CLAUDE_BOOTSTRAP.md` documents, in order:

1. Read `STATE.md` — "where are we right now"
2. Read `docs/CURRENT_STATE.md` if regenerated within 24h — "aggregated state"
3. Read last 3 ADRs — "what was recently decided"
4. (Optional) Read most recent `runs/exec_<N>/SUMMARY.md` — "what just happened"

Total: ~600 lines of input. Full project context. Ready to pair on building or
tuning.

---

## Closing Properties

Three properties of the design worth flagging because they fall out of the
structure rather than being explicitly designed-in:

1. **Each layer is independently testable** — stub the world model, exercise
   the planner; stub the planner, exercise MCTS; replace search without
   touching the model.
2. **The flywheel makes data quality a derivative of system use** — the more
   you use it, the better it gets at the things you use it for. "Learn the
   field deeply" operationalized.
3. **Phase 1 is a useful artifact independent of Phase 2/3** — even if
   full-memory observation never works, a goal-conditioned planner that
   completes Pokemon Red end-to-end is a real demonstrable thing.

---

## Appendix — Algorithm at a Glance

| Question | Answer |
|---|---|
| What's the algorithm? | MuZero variant — goal-conditioned, stochastic |
| Is it PPO? | No. PPO is model-free policy gradient; MuZero is model-based with planning |
| What does the agent learn? | Game dynamics (the "world model") — not strategy |
| Where does strategy come from? | The KB (hand-coded affordances) + MCTS at inference |
| How does it improve over time? | Verification loop feeds divergences back as labeled training data |
| How does it handle long horizons? | Hierarchical decomposition — many short-horizon searches, not one deep search |
| How does it handle RNG? | Stochastic chance nodes in the MCTS tree; save-state branching for high-stakes RNG |
| How does it handle goals like "catch all"? | Planner decomposes into 150 atomic subgoals + precedence-constrained route optimization |
| Does it need an LLM? | No. Tiny LMs allowed if they earn their keep; v1 has none |
| What's the hardware budget? | Single 8–12GB consumer GPU; ~100M-param world model |
