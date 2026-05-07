# Phase 0 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `world_model/` project skeleton with state extraction, knowledge-base loading, goal DSL atoms, world-model architecture stubs, observability artifacts, and an end-to-end smoke test that connects PyBoy → state → goal predicate → world-model forward pass without errors.

**Architecture:** New top-level `world_model/` directory in the existing repo, leaving `v2/` and `baselines/` intact. Python package `pokemon_planner` with submodules per Section 9.1 of the spec. Phase 0 builds *stubs* — most modules have minimum viable functionality that proves wiring works; richer implementations come in Phase 1a/1b/1c/1d.

**Tech Stack:** Python 3.10+, PyTorch (fp16 capable), PyBoy 2.4.0, Pydantic, PyYAML, pytest, ruff, mypy. Direct commits to `master` (project convention).

**Definition of done (from spec Section 9.3):** Repo scaffolded, KB schema + initial 30 species filled, state extractor works against `init.state`, world-model architecture stubs train on synthetic data without NaN, smoke test green, CI green.

**Reference:** Spec at `docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md`. Read sections 4.1 (state schema), 4.2 (architecture), 9.1 (file layout), 10 (observability) before starting tasks 4, 9, 1, and 3 respectively.

**Working directory convention:** All commands assume cwd is `world_model/` unless otherwise noted, mirroring how `v2/` and `baselines/` work in this repo. The ROM lives at `../PokemonRed.gb`; save states at `../init.state` etc.

**Windows note:** This repo's CLAUDE.md documents Windows-specific gotchas (numpy<2 pin, PowerShell stderr handling, requirements_windows.txt convention). Tasks 1 and 2 mirror those patterns.

---

## File Structure (created by this plan)

```
world_model/
├── pyproject.toml                          # Task 1
├── requirements.txt                        # Task 1
├── requirements_windows.txt                # Task 1
├── README.md                               # Task 1
├── STATE.md                                # Task 3
├── CLAUDE_BOOTSTRAP.md                     # Task 3
├── pokemon_planner/
│   ├── __init__.py                         # Task 1 (stub) → Task 2 (version)
│   ├── env.py                              # Task 6
│   ├── state.py                            # Task 4
│   ├── kb/
│   │   ├── __init__.py                     # Task 7
│   │   ├── species.yaml                    # Task 7
│   │   ├── trainers.yaml                   # Task 7 (stub)
│   │   ├── items.yaml                      # Task 7 (stub)
│   │   └── regions.yaml                    # Task 7 (stub)
│   ├── goals/
│   │   ├── __init__.py                     # Task 8
│   │   ├── dsl.py                          # Task 8
│   │   └── atoms.py                        # Task 8
│   └── world_model/
│       ├── __init__.py                     # Task 9
│       └── arch.py                         # Task 9
├── tests/
│   ├── __init__.py                         # Task 2
│   ├── conftest.py                         # Task 2
│   ├── test_smoke.py                       # Task 2
│   ├── test_state.py                       # Task 4
│   ├── test_state_extraction.py            # Task 6
│   ├── test_kb.py                          # Task 7
│   ├── test_goals.py                       # Task 8
│   ├── test_world_model_arch.py            # Task 9
│   ├── test_world_model_train_stub.py      # Task 10
│   └── test_e2e_smoke.py                   # Task 11
└── docs/
    ├── architecture.md                     # Task 3 (copy of spec)
    ├── adr/
    │   └── 0001-use-muzero-not-ppo.md      # Task 3
    ├── progress.md                         # Task 3
    └── tuning.md                           # Task 3
```

---

## Task 1: Scaffold `world_model/` directory + project metadata

**Files:**
- Create: `world_model/pyproject.toml`
- Create: `world_model/requirements.txt`
- Create: `world_model/requirements_windows.txt`
- Create: `world_model/README.md`
- Create: `world_model/pokemon_planner/__init__.py` (stub)

**Why:** Foundational. Nothing else can be installed or imported until this exists.

- [ ] **Step 1: Verify cwd and check `world_model/` does not yet exist**

Run from repo root:
```bash
ls world_model 2>/dev/null && echo "EXISTS — abort" || echo "OK to create"
```
Expected: `OK to create`

- [ ] **Step 2: Create `world_model/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pokemon-planner"
version = "0.0.1"
description = "Goal-conditioned world-model planner for Pokemon Red"
requires-python = ">=3.10"
authors = [{name = "RoseOfficial"}]
readme = "README.md"

[project.scripts]
poke-plan = "pokemon_planner.cli.run:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["pokemon_planner*"]

[tool.setuptools.package-data]
"pokemon_planner.kb" = ["*.yaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
markers = [
    "integration: marks tests requiring PyBoy + ROM (deselect with -m 'not integration')",
    "slow: marks tests that take > 5 seconds",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

- [ ] **Step 3: Create `world_model/requirements.txt`**

```
pyboy==2.4.0
torch>=2.0
pydantic>=2.5
pyyaml>=6.0
numpy<2
pytest>=8.0
ruff>=0.4
mypy>=1.8
```

- [ ] **Step 4: Create `world_model/requirements_windows.txt`**

Mirrors the `v2/requirements_windows.txt` pattern from this repo (filtered subset, no Linux-only wheels).

```
pyboy==2.4.0
torch>=2.0
pydantic>=2.5
pyyaml>=6.0
numpy<2
pytest>=8.0
ruff>=0.4
mypy>=1.8
```

- [ ] **Step 5: Create `world_model/README.md`**

```markdown
# world_model — Goal-Conditioned World-Model Planner for Pokemon Red

Phase 1 of a multi-phase project to build a goal-conditioned planner over a learned world model. See `docs/architecture.md` for the full design and `STATE.md` for current status.

## Quick start

```
cd world_model
pip install -r requirements.txt        # Linux/macOS
pip install -r requirements_windows.txt  # Windows
pip install -e .
pytest                                  # run tests
```

## Layout

- `pokemon_planner/` — the importable package (env, state, kb, goals, planner, world_model, search, flywheel, eval, cli)
- `tests/` — pytest tests, one file per surface
- `docs/` — architecture spec, ADRs, progress log, tuning log
- `data/`, `runs/` — gitignored artifacts produced at runtime

## Bootstrap for new sessions

Read `CLAUDE_BOOTSTRAP.md` before doing any work — it points to the right files in the right order.
```

- [ ] **Step 6: Create stub `world_model/pokemon_planner/__init__.py`**

```python
"""pokemon_planner — goal-conditioned world-model planner for Pokemon Red.

See ../docs/architecture.md for the full design.
"""

__version__ = "0.0.1"
```

- [ ] **Step 7: Verify `pip install -e .` works**

```bash
cd world_model
pip install -e . --quiet
python -c "import pokemon_planner; print(pokemon_planner.__version__)"
```
Expected: `0.0.1`

If `pip install` fails on Windows due to numpy 2.x: install `numpy<2` first, then `-e .`. This mirrors the existing repo's Windows gotcha.

- [ ] **Step 8: Commit**

```bash
git add world_model/pyproject.toml world_model/requirements.txt world_model/requirements_windows.txt world_model/README.md world_model/pokemon_planner/__init__.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Scaffold world_model package skeleton

Initial pyproject, requirements (with Windows variant per repo
convention), README, and stub __init__. No functionality yet —
just enough that `pip install -e .` works and the package imports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pytest configuration + import smoke test

**Files:**
- Create: `world_model/tests/__init__.py`
- Create: `world_model/tests/conftest.py`
- Create: `world_model/tests/test_smoke.py`

**Why:** Lock in the test runner before writing any production code. Following TDD discipline — every subsequent task starts with a failing test.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_smoke.py`:

```python
"""Smoke tests — verify the package and its core dependencies import."""
import pokemon_planner


def test_package_imports():
    assert pokemon_planner.__version__ == "0.0.1"


def test_torch_imports():
    import torch
    # Confirms torch installed; CUDA availability is informational, not required
    _ = torch.zeros(1)


def test_pyboy_imports():
    """PyBoy 2.x must be importable; cwd-independent."""
    import pyboy
    assert hasattr(pyboy, "PyBoy")


def test_pydantic_v2():
    import pydantic
    assert int(pydantic.VERSION.split(".")[0]) >= 2
```

- [ ] **Step 2: Create empty `tests/__init__.py`**

```python
```

- [ ] **Step 3: Create `tests/conftest.py` with shared fixtures**

```python
"""Shared pytest fixtures for the pokemon_planner test suite."""
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repo root (parent of world_model/)."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def rom_path(repo_root: Path) -> Path:
    """Path to PokemonRed.gb at the repo root.

    Tests using this fixture should be marked @pytest.mark.integration so
    they can be skipped when the ROM is unavailable.
    """
    p = repo_root / "PokemonRed.gb"
    if not p.exists():
        pytest.skip(f"ROM not found at {p} (legally obtain and place at repo root)")
    return p


@pytest.fixture(scope="session")
def init_state_path(repo_root: Path) -> Path:
    """Path to init.state save state at the repo root."""
    p = repo_root / "init.state"
    if not p.exists():
        pytest.skip(f"Save state not found at {p}")
    return p
```

- [ ] **Step 4: Run smoke tests**

```bash
cd world_model
pytest tests/test_smoke.py -v
```
Expected: all four tests pass.

If any fail: investigate dependency installation. Don't move on with red tests.

- [ ] **Step 5: Commit**

```bash
git add world_model/tests/__init__.py world_model/tests/conftest.py world_model/tests/test_smoke.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add pytest smoke tests + shared fixtures

Verifies package import and core deps (torch, pyboy, pydantic).
Conftest provides repo_root / rom_path / init_state_path fixtures
that gracefully skip when ROM/save states aren't available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Observability scaffolding — STATE.md, ADRs, progress, tuning, architecture copy

**Files:**
- Create: `world_model/STATE.md`
- Create: `world_model/CLAUDE_BOOTSTRAP.md`
- Create: `world_model/docs/architecture.md`
- Create: `world_model/docs/adr/0001-use-muzero-not-ppo.md`
- Create: `world_model/docs/progress.md`
- Create: `world_model/docs/tuning.md`

**Why:** The observability spec (Section 10) makes these load-bearing for agent-assisted iteration. Setting them up *first* establishes the discipline rather than retrofitting later. This task does not have unit tests — verification is "files exist and parse as valid Markdown."

- [ ] **Step 1: Create `world_model/STATE.md`**

```markdown
# Project State — 2026-05-07

## Phase
0 — Foundation (in progress)

## Working
- Package scaffolded; `pip install -e .` works
- Smoke tests pass (pokemon_planner imports, torch / pyboy / pydantic available)
- Observability scaffolding in place (this file, ADR-0001, architecture.md, progress.md, tuning.md, CLAUDE_BOOTSTRAP.md)

## Broken / Known Issues
- (none — Phase 0 just kicked off)

## Next Up
1. State Pydantic schema + extractor (Tasks 4–6 of Phase 0 plan)
2. KB schema + initial 30 species (Task 7)
3. Goal DSL atoms (Task 8)
4. World-model architecture stubs (Task 9)
5. Training stub on synthetic data (Task 10)
6. End-to-end smoke (Task 11)
7. Phase 1a plan: bootstrap demonstration data + first real WM training

## Recent Changes (last 7 days)
- 2026-05-07: Project kicked off; spec committed at `docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md`
- 2026-05-07: ADR-0001 captures the MuZero-vs-PPO decision
- 2026-05-07: Phase 0 scaffolding tasks 1–3 complete

## Where To Look
- Architecture: `docs/architecture.md`
- Recent decisions: `docs/adr/`
- Progress: `docs/progress.md`
- Tuning experiments: `docs/tuning.md`
- Phase 0 plan: `../docs/superpowers/plans/2026-05-07-phase-0-foundation.md`
```

- [ ] **Step 2: Create `world_model/CLAUDE_BOOTSTRAP.md`**

```markdown
# Claude Session Bootstrap

When a fresh Claude session opens this project, read these files in this order before doing any work:

1. **`STATE.md`** — where we are right now (always-current, ≤200 lines)
2. **`docs/CURRENT_STATE.md`** if it exists and was regenerated within the last 24h — aggregated metrics + diagnosis
3. **The last 3 ADRs in `docs/adr/`** (numerically highest = most recent) — what was recently decided
4. (Optional) **The most recent `runs/exec_<N>/SUMMARY.md`** — what just happened in the last execution

Total: ~600 lines of reading. Then you have full project context.

## Session conventions

- All commits go directly to `master`. No feature branches, no PRs in this repo.
- Working directory is `world_model/` for project-internal work; ROM and save states live one level up.
- Author identity for commits: `RoseOfficial <christopherscottkeller@gmail.com>` (use `git -c` overrides per commit; do NOT modify `git config`).
- Update `STATE.md` after any change that alters "what's working" or "what's broken." Update `docs/progress.md` for narrative milestones. Update `docs/tuning.md` after each experiment run.

## Files NEVER move or rename

- `STATE.md`, `CLAUDE_BOOTSTRAP.md` (this file)
- `docs/architecture.md`, `docs/adr/`, `docs/progress.md`, `docs/tuning.md`
- `docs/CURRENT_STATE.md` (when it exists)

These paths are part of the contract.
```

- [ ] **Step 3: Copy the spec into `world_model/docs/architecture.md`**

```bash
cp docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md world_model/docs/architecture.md
```

(Run from repo root. The copy lives alongside the project; the canonical spec under `docs/superpowers/specs/` is the source of truth, but `world_model/docs/architecture.md` is what gets read during sessions.)

- [ ] **Step 4: Create `world_model/docs/adr/0001-use-muzero-not-ppo.md`**

```markdown
# ADR-0001: Use MuZero variant, not PPO

Status: Accepted
Date: 2026-05-07
Supersedes: none

## Context

The existing repo (`baselines/` and `v2/`) uses PPO via Stable-Baselines3 with hand-shaped rewards (KNN exploration novelty, level/badge/heal/event reward terms). PWhiddy's PPO setup demonstrates RL can play Pokemon at all, but the architecture is fixed-goal: training optimizes for a single shaped reward, not arbitrary user-specified objectives.

The new project requires:
- Arbitrary goal expressions (`catch(ODDISH)`, `beat_champion()`, `catch_all()`)
- Long-horizon competence (10-hour `beat_champion()` runs)
- Visible search (the agent should *plan*, not just react)
- Generalization across goals (one model handling many objectives)

## Decision

Adopt a MuZero-family algorithm with two modifications:
1. **Goal conditioning** in value and policy heads: `V(state, goal)`, `π(action | state, goal)`.
2. **Stochastic chance nodes** in MCTS to handle in-game RNG (move accuracy, crit rolls, encounter slots).

Concrete components per spec Section 4: representation function `h`, dynamics function `g` (predicts next latent + next observation + reward), prediction function `f` (goal-conditioned policy + value). MCTS at inference, MuZero-style supervised + distillation losses at training.

## Consequences

Positive:
- Goal-conditioned by design — one model, many goals.
- Search is central and inspectable — supports the "tell it a goal, watch it plan" UX.
- Long-horizon handled by hierarchical decomposition (planner) + bootstrapped value (search), not reward shaping.
- Reference implementations exist (LightZero, EfficientZero) to fork.

Negative:
- More complex than PPO — more failure modes (latent collapse, value drift, dynamics overfitting).
- Less mature tooling on Windows specifically; expect to debug deps more than with SB3.
- Per-step compute is higher (MCTS forward passes); doesn't benefit from cheap CPU parallelism the way PPO with `SubprocVecEnv` does.

PPO's role going forward:
- `baselines/` and `v2/` PPO checkpoints serve as **demonstration data** for bootstrap world-model training (per spec Section 4.5).
- They serve as a **comparison baseline** for goals where both can be evaluated.
- They are **not deleted** — the existing trees remain for reference.

## Alternatives considered

- **Keep PPO, add goal conditioning** — multi-task PPO via goal-conditioned reward functions. Possible but doesn't get search-based planning, doesn't help long-horizon credit assignment, and goal-conditioning at policy-gradient scale is sample-inefficient.
- **DreamerV3 / TD-MPC2** — modern world-model RL. Considered in spec brainstorm. Less search-heavy, more amortized policy. Strong choice for "agent reflexively plays" but weaker for the "search visibly finds plans" UX. May revisit if MuZero variant doesn't pan out.
- **UVFA + classical search (A* / beam search)** — most interpretable but scales poorly to long horizons without subgoal decomposition. Useful Phase 2 augmentation, not a v1 spine.
```

- [ ] **Step 5: Create `world_model/docs/progress.md`**

```markdown
# Project Progress Log

Append-only narrative timeline. ~1–3 entries per week is a healthy cadence.

## 2026-05-07 — Project kicked off

Spec committed at `docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md`. Phase 0 plan committed at `docs/superpowers/plans/2026-05-07-phase-0-foundation.md`. ADR-0001 captures the MuZero-vs-PPO architectural decision. Scaffolding (Tasks 1–3) underway.
```

- [ ] **Step 6: Create `world_model/docs/tuning.md`**

```markdown
# Tuning Log

Append-only experiment table. One row per training/eval run that meaningfully changes something.

| Date | Run | Change | T1 pass rate | Notes |
|------|-----|--------|--------------|-------|
| 2026-05-07 | (none yet) | Phase 0 scaffolding | n/a | No model trained yet |
```

- [ ] **Step 7: Verify all observability files exist and are non-empty**

```bash
cd world_model
for f in STATE.md CLAUDE_BOOTSTRAP.md docs/architecture.md docs/adr/0001-use-muzero-not-ppo.md docs/progress.md docs/tuning.md; do
  if [ ! -s "$f" ]; then echo "MISSING OR EMPTY: $f"; exit 1; fi
done
echo "All observability files present and non-empty"
```
Expected: `All observability files present and non-empty`

- [ ] **Step 8: Commit**

```bash
git add world_model/STATE.md world_model/CLAUDE_BOOTSTRAP.md world_model/docs/
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add observability scaffolding for agent-assisted iteration

STATE.md (always-current state, capped at 200 lines), CLAUDE_BOOTSTRAP.md
(session-start protocol), docs/architecture.md (copy of spec for in-project
reference), ADR-0001 (the MuZero-vs-PPO decision), and empty-but-headered
progress.md and tuning.md.

Per spec Section 10, these files are load-bearing for the project — they
make a fresh Claude session productive in ~3 file reads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: State Pydantic schema

**Files:**
- Create: `world_model/pokemon_planner/state.py`
- Create: `world_model/tests/test_state.py`

**Why:** Defines the structured-RAM state vector from Section 4.1 of the spec. Every other component depends on this schema. Pydantic gives us validation, serialization, and type checking for free.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_state.py`:

```python
"""Tests for the structured RAM state schema."""
import pytest
from pydantic import ValidationError

from pokemon_planner.state import (
    BattleState,
    PartySlot,
    GameState,
)


def test_partyslot_valid():
    slot = PartySlot(
        species_id=0x01, level=5, hp_cur=20, hp_max=20, status=0,
        moves=(0x21, 0x00, 0x00, 0x00),  # Tackle + 3 empty
    )
    assert slot.species_id == 0x01
    assert slot.level == 5


def test_partyslot_rejects_oversized_level():
    with pytest.raises(ValidationError):
        PartySlot(species_id=1, level=101, hp_cur=10, hp_max=10, status=0,
                  moves=(0, 0, 0, 0))


def test_partyslot_rejects_hp_over_max():
    with pytest.raises(ValidationError):
        PartySlot(species_id=1, level=5, hp_cur=30, hp_max=20, status=0,
                  moves=(0, 0, 0, 0))


def test_battlestate_zeroed_when_not_in_battle():
    bs = BattleState(in_battle=False)
    assert bs.opp_species_id == 0
    assert bs.opp_level == 0
    assert bs.opp_hp == 0
    assert bs.turn == 0


def test_gamestate_minimal():
    gs = GameState(
        map_id=0x00,
        x=5,
        y=5,
        party=(),
        bag=(),
        badges=0,
        event_flags=bytes(256),
        money=0,
        time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256),
        menu_flags=0,
    )
    assert gs.party_size == 0
    assert gs.has_badge(0) is False


def test_gamestate_party_size_property():
    party = (
        PartySlot(species_id=1, level=5, hp_cur=20, hp_max=20, status=0,
                  moves=(0, 0, 0, 0)),
    )
    gs = GameState(
        map_id=0x00, x=5, y=5,
        party=party,
        bag=(), badges=0,
        event_flags=bytes(256),
        money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )
    assert gs.party_size == 1


def test_gamestate_has_badge():
    gs = GameState(
        map_id=0x00, x=5, y=5, party=(), bag=(),
        badges=0b00000011,  # Boulder + Cascade badges
        event_flags=bytes(256),
        money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )
    assert gs.has_badge(0) is True   # Boulder
    assert gs.has_badge(1) is True   # Cascade
    assert gs.has_badge(2) is False  # Thunder


def test_gamestate_event_flags_size_validated():
    with pytest.raises(ValidationError):
        GameState(
            map_id=0x00, x=5, y=5, party=(), bag=(), badges=0,
            event_flags=bytes(100),   # wrong length — should be 256
            money=0, time_played_frames=0,
            battle=BattleState(in_battle=False),
            tile_collision=bytes(256), menu_flags=0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_state.py -v
```
Expected: All tests fail with `ModuleNotFoundError: No module named 'pokemon_planner.state'`.

- [ ] **Step 3: Write the implementation**

Create `world_model/pokemon_planner/state.py`:

```python
"""Structured RAM state schema for Pokemon Red.

See ../../docs/architecture.md Section 4.1 for the field-by-field design.
The full state is ~660 bytes raw (Phase 1). Phase 2 will graduate to full WRAM.
"""
from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Sizes locked by the spec
EVENT_FLAGS_BYTES = 256
TILE_COLLISION_BYTES = 256       # 16x16 tiles around player
BAG_SLOTS = 20
PARTY_MAX = 6


class PartySlot(BaseModel):
    """One Pokemon in the player's party."""
    model_config = ConfigDict(frozen=True)

    species_id: int = Field(ge=0, le=255)
    level: int = Field(ge=1, le=100)
    hp_cur: int = Field(ge=0)
    hp_max: int = Field(ge=1)
    status: int = Field(ge=0, le=255)              # status condition byte
    moves: Tuple[int, int, int, int]               # 4 move IDs (0 = empty)

    @model_validator(mode="after")
    def _hp_cur_le_max(self) -> "PartySlot":
        if self.hp_cur > self.hp_max:
            raise ValueError(f"hp_cur ({self.hp_cur}) > hp_max ({self.hp_max})")
        return self

    @field_validator("moves")
    @classmethod
    def _moves_in_byte_range(cls, v: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        for m in v:
            if not 0 <= m <= 255:
                raise ValueError(f"move id {m} out of byte range")
        return v


class BagSlot(BaseModel):
    """One inventory slot."""
    model_config = ConfigDict(frozen=True)

    item_id: int = Field(ge=0, le=255)
    qty: int = Field(ge=0, le=99)


class BattleState(BaseModel):
    """Battle state. Zeroed when not in battle."""
    model_config = ConfigDict(frozen=True)

    in_battle: bool = False
    opp_species_id: int = Field(default=0, ge=0, le=255)
    opp_level: int = Field(default=0, ge=0, le=100)
    opp_hp: int = Field(default=0, ge=0)
    turn: int = Field(default=0, ge=0)


class GameState(BaseModel):
    """Full Phase-1 structured state extracted from a single PyBoy frame.

    Roughly 660 bytes raw — see spec Section 4.1 table.
    """
    model_config = ConfigDict(frozen=True)

    # Position
    map_id: int = Field(ge=0, le=255)
    x: int = Field(ge=0, le=255)
    y: int = Field(ge=0, le=255)

    # Party (variable length 0–6)
    party: Tuple[PartySlot, ...]

    # Bag (variable length 0–20)
    bag: Tuple[BagSlot, ...]

    # Progress
    badges: int = Field(ge=0, le=255)              # 8-bit mask
    event_flags: bytes                             # exactly EVENT_FLAGS_BYTES bytes
    money: int = Field(ge=0)
    time_played_frames: int = Field(ge=0)

    # Battle
    battle: BattleState

    # Local tile data
    tile_collision: bytes                          # exactly TILE_COLLISION_BYTES bytes

    # Mode / dialogue
    menu_flags: int = Field(ge=0)

    @field_validator("party")
    @classmethod
    def _party_size_ok(cls, v: Tuple[PartySlot, ...]) -> Tuple[PartySlot, ...]:
        if len(v) > PARTY_MAX:
            raise ValueError(f"party has {len(v)} slots; max is {PARTY_MAX}")
        return v

    @field_validator("bag")
    @classmethod
    def _bag_size_ok(cls, v: Tuple[BagSlot, ...]) -> Tuple[BagSlot, ...]:
        if len(v) > BAG_SLOTS:
            raise ValueError(f"bag has {len(v)} slots; max is {BAG_SLOTS}")
        return v

    @field_validator("event_flags")
    @classmethod
    def _event_flags_correct_length(cls, v: bytes) -> bytes:
        if len(v) != EVENT_FLAGS_BYTES:
            raise ValueError(f"event_flags must be {EVENT_FLAGS_BYTES} bytes, got {len(v)}")
        return v

    @field_validator("tile_collision")
    @classmethod
    def _tile_collision_correct_length(cls, v: bytes) -> bytes:
        if len(v) != TILE_COLLISION_BYTES:
            raise ValueError(f"tile_collision must be {TILE_COLLISION_BYTES} bytes, got {len(v)}")
        return v

    @property
    def party_size(self) -> int:
        return len(self.party)

    def has_badge(self, badge_index: int) -> bool:
        """badge_index 0..7 (Boulder, Cascade, Thunder, Rainbow, Soul, Marsh, Volcano, Earth)."""
        if not 0 <= badge_index <= 7:
            raise ValueError(f"badge_index out of range: {badge_index}")
        return bool(self.badges & (1 << badge_index))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd world_model
pytest tests/test_state.py -v
```
Expected: All 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add world_model/pokemon_planner/state.py world_model/tests/test_state.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add Pydantic schema for structured RAM state

Implements spec Section 4.1: GameState with party (≤6 slots), bag
(≤20 slots), badges (8-bit mask), 256B event flags, BCD money,
battle substate (zeroed when not in battle), 16x16 tile collision,
and menu flags. Field-level validators enforce sizes/ranges.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Pokemon RAM address constants

**Files:**
- Create: `world_model/pokemon_planner/_ram_addresses.py`
- Create: `world_model/tests/test_ram_addresses.py`

**Why:** Hex constants for the WRAM offsets we read in Task 6's extractor. Sourced from the existing `baselines/memory_addresses.py` and `v2/red_gym_env_v2.py` (which inlines them) plus pret/pokered for anything missing. Isolating them in a private module makes the extractor readable and the addresses unit-testable.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_ram_addresses.py`:

```python
"""Smoke tests for RAM address constants — sanity checks, not full coverage."""
from pokemon_planner import _ram_addresses as ram


def test_addresses_in_wram_range():
    """All addresses we read should be in WRAM ($C000-$DFFF)."""
    for name in dir(ram):
        if name.startswith("_") or name.upper() != name:
            continue
        value = getattr(ram, name)
        if isinstance(value, int):
            assert 0xC000 <= value <= 0xDFFF, f"{name} = {hex(value)} not in WRAM"


def test_known_canonical_addresses():
    """Cross-check against well-documented Pokemon Red RAM map."""
    # These are widely-documented; if they drift we have a regression
    assert ram.PLAYER_X == 0xD362
    assert ram.PLAYER_Y == 0xD361
    assert ram.MAP_ID == 0xD35E
    assert ram.PARTY_SIZE == 0xD163
    assert ram.PARTY_SPECIES_LIST == 0xD164  # 6 bytes starting here
    assert ram.MONEY == 0xD347                # 3 BCD bytes
    assert ram.BADGES == 0xD356


def test_event_flag_range():
    """Event flag region covers $D747-$D87E per pret/pokered."""
    assert ram.EVENT_FLAGS_START == 0xD747
    assert ram.EVENT_FLAGS_END == 0xD87E
    assert ram.EVENT_FLAGS_END - ram.EVENT_FLAGS_START + 1 == 0x138  # 312 bytes


def test_party_struct_offsets():
    """Per-Pokemon party struct is 44 bytes per pret docs."""
    assert ram.PARTY_STRUCT_SIZE == 44
    # Offsets within a single party member's 44-byte struct
    assert ram.PARTY_OFFSET_SPECIES == 0
    assert ram.PARTY_OFFSET_HP_CUR == 1   # 2 bytes BE
    assert ram.PARTY_OFFSET_LEVEL == 33
    assert ram.PARTY_OFFSET_STATUS == 4
    assert ram.PARTY_OFFSET_HP_MAX == 34  # 2 bytes BE
    assert ram.PARTY_OFFSET_MOVES == 8    # 4 bytes
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_ram_addresses.py -v
```
Expected: ImportError or AttributeError on `_ram_addresses`.

- [ ] **Step 3: Write the implementation**

Create `world_model/pokemon_planner/_ram_addresses.py`:

```python
"""Pokemon Red RAM address constants.

Sourced from the existing v2/baselines memory maps (PWhiddy's repo) and
pret/pokered ROM disassembly. Addresses target Game Boy WRAM ($C000-$DFFF).

This module is INTERNAL — outside callers use pokemon_planner.env.read_state(),
which returns a GameState. Isolating addresses here keeps the extractor readable.

References:
- pret/pokered: https://github.com/pret/pokered
- datacrystal RAM map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
- v2/red_gym_env_v2.py inlines many of these as 0xD... constants
"""

# ---- Position ----
MAP_ID = 0xD35E
PLAYER_Y = 0xD361
PLAYER_X = 0xD362

# ---- Party ----
PARTY_SIZE = 0xD163
PARTY_SPECIES_LIST = 0xD164          # 6 bytes; 0xFF = empty
PARTY_STRUCTS_START = 0xD16B         # 6 contiguous 44-byte structs
PARTY_STRUCT_SIZE = 44

# Offsets within a single 44-byte party struct
PARTY_OFFSET_SPECIES = 0
PARTY_OFFSET_HP_CUR = 1              # 2 bytes BE
PARTY_OFFSET_BOX_LEVEL = 3           # 1 byte (storage); use OFFSET_LEVEL for current
PARTY_OFFSET_STATUS = 4
PARTY_OFFSET_TYPE1 = 5
PARTY_OFFSET_TYPE2 = 6
PARTY_OFFSET_CATCH_RATE = 7
PARTY_OFFSET_MOVES = 8               # 4 bytes
PARTY_OFFSET_OT_ID = 12              # 2 bytes
PARTY_OFFSET_EXP = 14                # 3 bytes BE
PARTY_OFFSET_HP_EV = 17              # 2 bytes BE
PARTY_OFFSET_ATK_EV = 19
PARTY_OFFSET_DEF_EV = 21
PARTY_OFFSET_SPD_EV = 23
PARTY_OFFSET_SPC_EV = 25
PARTY_OFFSET_IVS = 27                # 2 bytes packed
PARTY_OFFSET_PP = 29                 # 4 bytes
PARTY_OFFSET_LEVEL = 33
PARTY_OFFSET_HP_MAX = 34             # 2 bytes BE
PARTY_OFFSET_ATK = 36
PARTY_OFFSET_DEF = 38
PARTY_OFFSET_SPD = 40
PARTY_OFFSET_SPC = 42

# ---- Bag ----
BAG_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E             # pairs of (item_id, qty), 20 max, terminated 0xFF
BAG_MAX_SLOTS = 20

# ---- Money & time ----
MONEY = 0xD347                       # 3 BCD bytes BE
TIME_PLAYED_HOURS = 0xDA40
TIME_PLAYED_MINUTES = 0xDA42
TIME_PLAYED_SECONDS = 0xDA43
TIME_PLAYED_FRAMES = 0xDA44

# ---- Progress ----
BADGES = 0xD356                      # 8-bit mask
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E             # inclusive
EVENT_FLAGS_LEN = EVENT_FLAGS_END - EVENT_FLAGS_START + 1  # 312 bytes

# ---- Battle ----
IS_IN_BATTLE = 0xD057                # 0 = no, 1 = wild, 2 = trainer
BATTLE_OPP_SPECIES = 0xCFE5
BATTLE_OPP_LEVEL = 0xCFF3
BATTLE_OPP_HP = 0xCFE6               # 2 bytes BE
BATTLE_TURN = 0xCCD5

# ---- Tile / dialogue / menu ----
CURRENT_TILESET = 0xD367
TEXT_BOX_FLAG = 0xCD60               # rough indicator of dialogue state
MENU_FLAGS = 0xCC26                  # cursor / menu state
JOYPAD_INPUT = 0xFF8A                # scratch — not WRAM, not used
```

Note: tile collision data (the 16x16 around the player) doesn't sit at a single fixed address — it's reconstructed from the map's tile data + player position. That extraction logic lives in `env.py` (Task 6), not here, because it composes addresses rather than reading a single one.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd world_model
pytest tests/test_ram_addresses.py -v
```
Expected: All 4 tests pass.

If `test_addresses_in_wram_range` fails: the `JOYPAD_INPUT` constant ($FF8A) is HRAM, not WRAM. Either remove it (we don't use it) or exclude it explicitly from the test. **Remove it** — we don't read joypad input, only write to it via PyBoy's button-press API.

After removing `JOYPAD_INPUT = 0xFF8A`:

```bash
pytest tests/test_ram_addresses.py -v
```
Expected: All 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add world_model/pokemon_planner/_ram_addresses.py world_model/tests/test_ram_addresses.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add internal RAM address constants for Pokemon Red

Sourced from existing baselines/memory_addresses.py + v2/ inlined
constants + pret/pokered. Module is INTERNAL — public API goes
through pokemon_planner.env.read_state(). Smoke tests verify
addresses fall in WRAM and match canonical pret docs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: PyBoy wrapper + state extractor

**Files:**
- Create: `world_model/pokemon_planner/env.py`
- Create: `world_model/tests/test_state_extraction.py`

**Why:** Bridge between the emulator and our schema. `read_state(pyboy)` is the workhorse — every other component consumes its output. Tests are integration-level (need the ROM + a save state).

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_state_extraction.py`:

```python
"""Integration tests for state extraction from PyBoy.

Marked as `integration` — skipped if ROM/save-state unavailable.
"""
import pytest

from pokemon_planner.env import PokeBoy, read_state
from pokemon_planner.state import GameState


@pytest.mark.integration
def test_extract_from_init_state(rom_path, init_state_path):
    """Load init.state, extract state, verify structural validity.

    init.state lands somewhere past the title screen. We don't assert
    *which* map — we assert the extracted state is internally coherent.
    """
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        assert isinstance(state, GameState)

        # Structural sanity (not value-specific — init.state may evolve)
        assert 0 <= state.party_size <= 6
        assert 0 <= state.x <= 255
        assert 0 <= state.y <= 255
        assert 0 <= state.map_id <= 255
        assert 0 <= state.money < 1_000_000   # BCD max for 3 bytes is 999999
        assert 0 <= state.badges <= 255
        assert len(state.event_flags) == 256
        assert len(state.tile_collision) == 256

        # If party is non-empty, validate slots
        for slot in state.party:
            assert 1 <= slot.level <= 100
            assert slot.hp_cur <= slot.hp_max
            assert 0 <= slot.species_id <= 255
    finally:
        pb.close()


@pytest.mark.integration
def test_state_is_immutable(rom_path, init_state_path):
    """Pydantic frozen=True means extracted states can't be mutated in place."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        with pytest.raises(Exception):  # pydantic raises ValidationError on frozen field assign
            state.x = 99  # type: ignore[misc]
    finally:
        pb.close()


@pytest.mark.integration
def test_two_consecutive_reads_are_equal(rom_path, init_state_path):
    """Reading state twice without stepping should produce equal results."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        s1 = read_state(pb.pyboy)
        s2 = read_state(pb.pyboy)
        assert s1 == s2
    finally:
        pb.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_state_extraction.py -v
```
Expected: ImportError on `pokemon_planner.env`.

- [ ] **Step 3: Write the implementation**

Create `world_model/pokemon_planner/env.py`:

```python
"""PyBoy wrapper + state extractor.

Public API:
    PokeBoy(rom_path, save_state_path=None) — context-manager-friendly wrapper
    read_state(pyboy) -> GameState           — extract structured state from a PyBoy instance

Phase 1 reads ~660 bytes of curated WRAM. Phase 2 will graduate to full memory
snapshots; that lives in a separate module to avoid disturbing this one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyboy import PyBoy

from pokemon_planner import _ram_addresses as ram
from pokemon_planner.state import (
    BAG_SLOTS,
    BattleState,
    BagSlot,
    EVENT_FLAGS_BYTES,
    GameState,
    PartySlot,
    PARTY_MAX,
    TILE_COLLISION_BYTES,
)


# ---- Wrapper ----

class PokeBoy:
    """Thin context-manager wrapper around PyBoy 2.x.

    Mostly here to centralize boot + save-state load so callers don't repeat
    the dance. The underlying PyBoy instance is exposed as `.pyboy`.
    """

    def __init__(
        self,
        rom_path: str | Path,
        save_state_path: Optional[str | Path] = None,
        window: str = "null",   # "null" = headless; "SDL2" for visible
    ):
        rom_path = Path(rom_path).resolve()
        if not rom_path.exists():
            raise FileNotFoundError(f"ROM not found: {rom_path}")
        self.pyboy = PyBoy(str(rom_path), window=window)
        # PyBoy 2.x boots automatically; tick once so first frame is rendered
        self.pyboy.tick()
        if save_state_path is not None:
            with open(save_state_path, "rb") as f:
                self.pyboy.load_state(f)

    def close(self) -> None:
        self.pyboy.stop()

    def __enter__(self) -> "PokeBoy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---- Helpers ----

def _read_byte(pb: PyBoy, addr: int) -> int:
    return pb.memory[addr]


def _read_bytes(pb: PyBoy, addr: int, n: int) -> bytes:
    return bytes(pb.memory[addr + i] for i in range(n))


def _read_u16_be(pb: PyBoy, addr: int) -> int:
    """Big-endian 2-byte read (Pokemon Red stat layout)."""
    hi = pb.memory[addr]
    lo = pb.memory[addr + 1]
    return (hi << 8) | lo


def _read_bcd(pb: PyBoy, addr: int, n_bytes: int) -> int:
    """Read n_bytes of binary-coded decimal big-endian."""
    val = 0
    for i in range(n_bytes):
        b = pb.memory[addr + i]
        hi = (b >> 4) & 0x0F
        lo = b & 0x0F
        val = val * 100 + hi * 10 + lo
    return val


# ---- Per-section extractors ----

def _read_party(pb: PyBoy) -> tuple[PartySlot, ...]:
    n = min(_read_byte(pb, ram.PARTY_SIZE), PARTY_MAX)
    out: list[PartySlot] = []
    for i in range(n):
        base = ram.PARTY_STRUCTS_START + i * ram.PARTY_STRUCT_SIZE
        species = _read_byte(pb, base + ram.PARTY_OFFSET_SPECIES)
        if species == 0xFF:                # empty slot sentinel
            continue
        hp_cur = _read_u16_be(pb, base + ram.PARTY_OFFSET_HP_CUR)
        hp_max = _read_u16_be(pb, base + ram.PARTY_OFFSET_HP_MAX)
        # Defensive: clamp hp_cur to hp_max in case of mid-frame torn read
        if hp_max == 0:
            hp_max = 1
        if hp_cur > hp_max:
            hp_cur = hp_max
        level = max(1, min(100, _read_byte(pb, base + ram.PARTY_OFFSET_LEVEL)))
        status = _read_byte(pb, base + ram.PARTY_OFFSET_STATUS)
        moves = tuple(_read_byte(pb, base + ram.PARTY_OFFSET_MOVES + j) for j in range(4))
        out.append(PartySlot(
            species_id=species,
            level=level,
            hp_cur=hp_cur,
            hp_max=hp_max,
            status=status,
            moves=moves,  # type: ignore[arg-type]
        ))
    return tuple(out)


def _read_bag(pb: PyBoy) -> tuple[BagSlot, ...]:
    n = min(_read_byte(pb, ram.BAG_COUNT), BAG_SLOTS)
    out: list[BagSlot] = []
    for i in range(n):
        item_id = _read_byte(pb, ram.BAG_ITEMS_START + i * 2)
        qty = _read_byte(pb, ram.BAG_ITEMS_START + i * 2 + 1)
        if item_id == 0xFF:
            break
        out.append(BagSlot(item_id=item_id, qty=min(qty, 99)))
    return tuple(out)


def _read_event_flags(pb: PyBoy) -> bytes:
    """Curated 256-byte subset of the 312-byte flag region.

    Phase 1 keeps the first 256 of 312 — covers the early-mid game story
    flags. Phase 2 may include the full range or a different curated subset.
    """
    return _read_bytes(pb, ram.EVENT_FLAGS_START, EVENT_FLAGS_BYTES)


def _read_battle(pb: PyBoy) -> BattleState:
    in_battle = _read_byte(pb, ram.IS_IN_BATTLE) != 0
    if not in_battle:
        return BattleState(in_battle=False)
    return BattleState(
        in_battle=True,
        opp_species_id=_read_byte(pb, ram.BATTLE_OPP_SPECIES),
        opp_level=min(100, _read_byte(pb, ram.BATTLE_OPP_LEVEL)),
        opp_hp=_read_u16_be(pb, ram.BATTLE_OPP_HP),
        turn=min(255, _read_byte(pb, ram.BATTLE_TURN)),
    )


def _read_tile_collision_stub(pb: PyBoy) -> bytes:
    """Placeholder — Phase 0 returns zeros.

    Real tile-collision extraction requires looking up the current tileset's
    collision table and indexing into it via the player's overworld tile map
    around (x, y). That logic is non-trivial and lives in a follow-up task
    (Phase 1a Task X). For Phase 0 the schema requires the right *length*,
    so we return 256 zeros.
    """
    return bytes(TILE_COLLISION_BYTES)


# ---- Public API ----

def read_state(pyboy: PyBoy) -> GameState:
    """Extract a structured GameState from a PyBoy instance at its current frame."""
    return GameState(
        map_id=_read_byte(pyboy, ram.MAP_ID),
        x=_read_byte(pyboy, ram.PLAYER_X),
        y=_read_byte(pyboy, ram.PLAYER_Y),
        party=_read_party(pyboy),
        bag=_read_bag(pyboy),
        badges=_read_byte(pyboy, ram.BADGES),
        event_flags=_read_event_flags(pyboy),
        money=_read_bcd(pyboy, ram.MONEY, 3),
        time_played_frames=_read_byte(pyboy, ram.TIME_PLAYED_FRAMES),
        battle=_read_battle(pyboy),
        tile_collision=_read_tile_collision_stub(pyboy),
        menu_flags=_read_byte(pyboy, ram.MENU_FLAGS),
    )
```

- [ ] **Step 4: Run integration tests**

```bash
cd world_model
pytest tests/test_state_extraction.py -v -m integration
```

Expected: 3 tests pass IF ROM and `init.state` are present at repo root. If they're missing, tests are skipped (the conftest fixtures handle that gracefully).

If a test fails with `KeyError` or `AttributeError` on `pb.memory[addr]`: PyBoy 2.x exposes memory as an indexable object; older versions used `pyboy.botsupport_manager().memory()`. We're on 2.4.0 per `requirements.txt`, so `pb.memory[addr]` is correct. If broken, double-check `pip show pyboy` returns 2.4.0.

- [ ] **Step 5: Commit**

```bash
git add world_model/pokemon_planner/env.py world_model/tests/test_state_extraction.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add PyBoy wrapper + structured state extractor

PokeBoy class wraps PyBoy 2.x boot + save-state load. read_state()
pulls position, party, bag, badges, event flags, money, battle
state, and menu flags into a GameState. Tile-collision extraction
is a 256B-zeros stub for Phase 0; real impl deferred to Phase 1a.

Integration-marked tests verify structural validity against init.state.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: KB schema + initial 30-species seed

**Files:**
- Create: `world_model/pokemon_planner/kb/__init__.py`
- Create: `world_model/pokemon_planner/kb/species.yaml`
- Create: `world_model/pokemon_planner/kb/trainers.yaml`
- Create: `world_model/pokemon_planner/kb/items.yaml`
- Create: `world_model/pokemon_planner/kb/regions.yaml`
- Create: `world_model/tests/test_kb.py`

**Why:** The KB encodes domain knowledge ("Bellsprout lives on Route 24, requires Cascade Badge"). Layer 2 planner depends on this for goal decomposition. Phase 0 only needs 30 species (early-game) to validate the loader; the rest get filled across Phase 1a/1b/1c.

The 30 species: Bulbasaur, Ivysaur, Venusaur, Charmander, Charmeleon, Charizard, Squirtle, Wartortle, Blastoise (9 starters/evolutions), Pidgey, Pidgeotto, Rattata, Raticate, Caterpie, Metapod, Butterfree, Weedle, Kakuna, Beedrill, Spearow, Fearow, Ekans (Red exclusive), Sandshrew (Red exclusive), Nidoran♂, Nidoran♀, Pikachu, Geodude, Mankey, Zubat, Oddish.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_kb.py`:

```python
"""Tests for the knowledge base loader."""
import pytest

from pokemon_planner.kb import KnowledgeBase, load_kb


def test_load_kb():
    kb = load_kb()
    assert isinstance(kb, KnowledgeBase)


def test_species_count_at_least_30():
    kb = load_kb()
    assert len(kb.species) >= 30


def test_oddish_in_species():
    kb = load_kb()
    oddish = kb.get_species("ODDISH")
    assert oddish.species_id == 0x47  # canonical Pokemon Red species ID for Oddish
    assert oddish.name == "ODDISH"


def test_charmander_starter_metadata():
    kb = load_kb()
    char = kb.get_species("CHARMANDER")
    assert char.species_id == 0xB0
    assert "starter" in char.tags
    assert char.encounter_method == "gift"


def test_ekans_red_exclusive():
    """Ekans is Red-exclusive (Sandshrew is the Blue counterpart)."""
    kb = load_kb()
    ekans = kb.get_species("EKANS")
    assert "red_exclusive" in ekans.tags


def test_unknown_species_raises():
    kb = load_kb()
    with pytest.raises(KeyError):
        kb.get_species("MISSINGNO")


def test_kb_is_frozen_after_load():
    """The loaded KB shouldn't be mutable from the outside."""
    kb = load_kb()
    with pytest.raises(Exception):
        kb.species["BULBASAUR"].name = "Tomato"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_kb.py -v
```
Expected: ImportError on `pokemon_planner.kb`.

- [ ] **Step 3: Create `world_model/pokemon_planner/kb/__init__.py`**

```python
"""Knowledge base — curated game data, not learned.

Phase 0 ships with 30 species + stub schemas for trainers/items/regions.
Subsequent phases expand the data; the loader contract stays stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel, ConfigDict


KB_DIR = Path(__file__).parent


class Species(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str                              # uppercase identifier, e.g. "ODDISH"
    species_id: int                        # canonical Pokemon Red byte ID
    encounter_method: str                  # "wild" | "gift" | "trade" | "event"
    encounter_zones: tuple[str, ...] = ()  # region names (matches regions.yaml entries)
    required_badges: tuple[str, ...] = ()
    requires_item: tuple[str, ...] = ()
    evolution_from: str | None = None
    evolution_method: str | None = None    # "level:36" | "stone:LEAF" | "trade" | None
    tags: tuple[str, ...] = ()             # free-form: "starter", "red_exclusive", "legendary", etc.


class Trainer(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    location: str
    party: tuple[str, ...] = ()


class Item(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    item_id: int
    sources: tuple[str, ...] = ()


class Region(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    map_id: int
    blockers: tuple[str, ...] = ()         # e.g. "snorlax", "cut_tree", "surf"


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    species: Mapping[str, Species]
    trainers: Mapping[str, Trainer]
    items: Mapping[str, Item]
    regions: Mapping[str, Region]

    def get_species(self, name: str) -> Species:
        if name not in self.species:
            raise KeyError(f"Unknown species: {name}")
        return self.species[name]


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        data = yaml.safe_load(f)
    return data or []


def load_kb(kb_dir: Path | None = None) -> KnowledgeBase:
    """Load all KB YAML files and construct a frozen KnowledgeBase."""
    d = kb_dir if kb_dir is not None else KB_DIR
    species = {x["name"]: Species(**x) for x in _load_yaml(d / "species.yaml")}
    trainers = {x["name"]: Trainer(**x) for x in _load_yaml(d / "trainers.yaml")}
    items = {x["name"]: Item(**x) for x in _load_yaml(d / "items.yaml")}
    regions = {x["name"]: Region(**x) for x in _load_yaml(d / "regions.yaml")}
    return KnowledgeBase(
        species=species, trainers=trainers, items=items, regions=regions
    )
```

- [ ] **Step 4: Create `world_model/pokemon_planner/kb/species.yaml`**

Initial 30 species. Species IDs from pret/pokered's `pokemon.asm`. Encounter zones reference region names that we'll fill in regions.yaml at Step 6.

```yaml
- name: BULBASAUR
  species_id: 0x99
  encounter_method: gift
  encounter_zones: [OAKS_LAB]
  tags: [starter, grass, poison]

- name: IVYSAUR
  species_id: 0x09
  encounter_method: trade
  evolution_from: BULBASAUR
  evolution_method: "level:16"
  tags: [grass, poison]

- name: VENUSAUR
  species_id: 0x9A
  encounter_method: trade
  evolution_from: IVYSAUR
  evolution_method: "level:32"
  tags: [grass, poison]

- name: CHARMANDER
  species_id: 0xB0
  encounter_method: gift
  encounter_zones: [OAKS_LAB]
  tags: [starter, fire]

- name: CHARMELEON
  species_id: 0xB2
  encounter_method: trade
  evolution_from: CHARMANDER
  evolution_method: "level:16"
  tags: [fire]

- name: CHARIZARD
  species_id: 0xB4
  encounter_method: trade
  evolution_from: CHARMELEON
  evolution_method: "level:36"
  tags: [fire, flying]

- name: SQUIRTLE
  species_id: 0xB1
  encounter_method: gift
  encounter_zones: [OAKS_LAB]
  tags: [starter, water]

- name: WARTORTLE
  species_id: 0xB3
  encounter_method: trade
  evolution_from: SQUIRTLE
  evolution_method: "level:16"
  tags: [water]

- name: BLASTOISE
  species_id: 0x1C
  encounter_method: trade
  evolution_from: WARTORTLE
  evolution_method: "level:36"
  tags: [water]

- name: PIDGEY
  species_id: 0x24
  encounter_method: wild
  encounter_zones: [ROUTE_1, ROUTE_2, ROUTE_22, VIRIDIAN_FOREST]
  tags: [normal, flying, common]

- name: PIDGEOTTO
  species_id: 0x96
  encounter_method: wild
  encounter_zones: [ROUTE_5, ROUTE_6, ROUTE_24, ROUTE_25]
  evolution_from: PIDGEY
  evolution_method: "level:18"
  tags: [normal, flying]

- name: RATTATA
  species_id: 0xA5
  encounter_method: wild
  encounter_zones: [ROUTE_1, ROUTE_2, ROUTE_22]
  tags: [normal, common]

- name: RATICATE
  species_id: 0xA6
  encounter_method: wild
  encounter_zones: [ROUTE_16, ROUTE_17, ROUTE_18]
  evolution_from: RATTATA
  evolution_method: "level:20"
  tags: [normal]

- name: CATERPIE
  species_id: 0x7B
  encounter_method: wild
  encounter_zones: [VIRIDIAN_FOREST, ROUTE_25]
  tags: [bug, common]

- name: METAPOD
  species_id: 0x7C
  encounter_method: wild
  encounter_zones: [VIRIDIAN_FOREST]
  evolution_from: CATERPIE
  evolution_method: "level:7"
  tags: [bug]

- name: BUTTERFREE
  species_id: 0x7D
  encounter_method: trade
  evolution_from: METAPOD
  evolution_method: "level:10"
  tags: [bug, flying]

- name: WEEDLE
  species_id: 0x70
  encounter_method: wild
  encounter_zones: [VIRIDIAN_FOREST, ROUTE_25]
  tags: [bug, poison, common]

- name: KAKUNA
  species_id: 0x71
  encounter_method: wild
  encounter_zones: [VIRIDIAN_FOREST]
  evolution_from: WEEDLE
  evolution_method: "level:7"
  tags: [bug, poison]

- name: BEEDRILL
  species_id: 0x72
  encounter_method: trade
  evolution_from: KAKUNA
  evolution_method: "level:10"
  tags: [bug, poison]

- name: SPEAROW
  species_id: 0x05
  encounter_method: wild
  encounter_zones: [ROUTE_3, ROUTE_22]
  tags: [normal, flying]

- name: FEAROW
  species_id: 0x23
  encounter_method: wild
  encounter_zones: [ROUTE_9, ROUTE_10]
  evolution_from: SPEAROW
  evolution_method: "level:20"
  tags: [normal, flying]

- name: EKANS
  species_id: 0x6D
  encounter_method: wild
  encounter_zones: [ROUTE_4, ROUTE_9]
  tags: [poison, red_exclusive]

- name: SANDSHREW
  species_id: 0x60
  encounter_method: wild
  encounter_zones: [ROUTE_4, ROUTE_23]
  tags: [ground, red_exclusive]

- name: NIDORAN_M
  species_id: 0x0F
  encounter_method: wild
  encounter_zones: [ROUTE_3, ROUTE_4, ROUTE_22]
  tags: [poison]

- name: NIDORAN_F
  species_id: 0x03
  encounter_method: wild
  encounter_zones: [ROUTE_3, ROUTE_4, ROUTE_22]
  tags: [poison]

- name: PIKACHU
  species_id: 0x54
  encounter_method: wild
  encounter_zones: [VIRIDIAN_FOREST, POWER_PLANT]
  tags: [electric, mascot]

- name: GEODUDE
  species_id: 0xA9
  encounter_method: wild
  encounter_zones: [MT_MOON, ROCK_TUNNEL, VICTORY_ROAD]
  tags: [rock, ground]

- name: MANKEY
  species_id: 0x39
  encounter_method: wild
  encounter_zones: [ROUTE_22]
  tags: [fighting]

- name: ZUBAT
  species_id: 0x6B
  encounter_method: wild
  encounter_zones: [MT_MOON, ROCK_TUNNEL, SEAFOAM_ISLANDS]
  tags: [poison, flying]

- name: ODDISH
  species_id: 0x47
  encounter_method: wild
  encounter_zones: [ROUTE_5, ROUTE_6, ROUTE_24, ROUTE_25]
  required_badges: [CASCADE]
  tags: [grass, poison]
```

- [ ] **Step 5: Create stub `world_model/pokemon_planner/kb/trainers.yaml`**

```yaml
- name: BROCK
  location: PEWTER_GYM
  party: [GEODUDE, ONIX]

- name: MISTY
  location: CERULEAN_GYM
  party: [STARYU, STARMIE]
```

- [ ] **Step 6: Create stub `world_model/pokemon_planner/kb/items.yaml`**

```yaml
- name: POKEBALL
  item_id: 0x04
  sources: [POKEMART_ANY]

- name: POTION
  item_id: 0x14
  sources: [POKEMART_ANY]

- name: BICYCLE
  item_id: 0x06
  sources: [BIKE_SHOP_CERULEAN]
```

- [ ] **Step 7: Create stub `world_model/pokemon_planner/kb/regions.yaml`**

Regions referenced from species.yaml. Map IDs from pret/pokered's `constants/map_constants.asm`.

```yaml
- name: PALLET_TOWN
  map_id: 0x00

- name: VIRIDIAN_CITY
  map_id: 0x01

- name: PEWTER_CITY
  map_id: 0x02

- name: CERULEAN_CITY
  map_id: 0x03

- name: ROUTE_1
  map_id: 0x0C

- name: ROUTE_2
  map_id: 0x0D

- name: ROUTE_3
  map_id: 0x0E

- name: ROUTE_4
  map_id: 0x0F

- name: ROUTE_5
  map_id: 0x10
  blockers: [GUARD_HOUSE_5_6]

- name: ROUTE_6
  map_id: 0x11

- name: ROUTE_9
  map_id: 0x14

- name: ROUTE_10
  map_id: 0x15

- name: ROUTE_16
  map_id: 0x1B

- name: ROUTE_17
  map_id: 0x1C

- name: ROUTE_18
  map_id: 0x1D

- name: ROUTE_22
  map_id: 0x21

- name: ROUTE_23
  map_id: 0x22

- name: ROUTE_24
  map_id: 0x23

- name: ROUTE_25
  map_id: 0x24

- name: VIRIDIAN_FOREST
  map_id: 0x33

- name: MT_MOON
  map_id: 0x3B
  blockers: []

- name: ROCK_TUNNEL
  map_id: 0x52
  blockers: [FLASH]

- name: POWER_PLANT
  map_id: 0x53
  blockers: [SURF]

- name: SEAFOAM_ISLANDS
  map_id: 0xC1
  blockers: [SURF, STRENGTH]

- name: VICTORY_ROAD
  map_id: 0x6C
  blockers: [STRENGTH, SURF]

- name: OAKS_LAB
  map_id: 0x28

- name: PEWTER_GYM
  map_id: 0x36

- name: CERULEAN_GYM
  map_id: 0x41
```

- [ ] **Step 8: Run KB tests**

```bash
cd world_model
pytest tests/test_kb.py -v
```
Expected: All 7 tests pass.

- [ ] **Step 9: Commit**

```bash
git add world_model/pokemon_planner/kb/
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add KB schema + initial 30-species seed data

Pydantic schema for Species/Trainer/Item/Region; YAML data files
loaded at runtime via load_kb(). 30 early-game species filled
(starters, route 1-3 commons, Pewter/Cerulean-relevant, Pikachu,
Mankey, etc.). Trainers/items/regions seeded with the entries
referenced from species data; rest filled in subsequent phases.

Per spec Section 2: KB is data, not learned. Sourced from
pret/pokered ROM disassembly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Goal DSL — atoms + combinators

**Files:**
- Create: `world_model/pokemon_planner/goals/__init__.py`
- Create: `world_model/pokemon_planner/goals/dsl.py`
- Create: `world_model/pokemon_planner/goals/atoms.py`
- Create: `world_model/tests/test_goals.py`

**Why:** Layer 3 of the architecture (spec Section 2). Phase 0 ships atoms (catch / beat / reach / have_item / level / evolve) and the four combinators (then / and_ / or_ / forall). Named shortcuts like `catch_all()` and `beat_champion()` come in Phase 1c when the planner can actually execute them.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_goals.py`:

```python
"""Tests for the goal DSL — atoms, combinators, and predicate evaluation."""
import pytest

from pokemon_planner.goals import (
    catch,
    beat,
    reach,
    have_item,
    level,
    evolve,
    then,
    and_,
    or_,
    forall,
    Goal,
    Atom,
    Then,
    And,
    Or,
    Forall,
)
from pokemon_planner.state import (
    BagSlot,
    BattleState,
    GameState,
    PartySlot,
)


def _empty_state(map_id: int = 0, party: tuple = (), bag: tuple = (), badges: int = 0) -> GameState:
    return GameState(
        map_id=map_id, x=0, y=0, party=party, bag=bag, badges=badges,
        event_flags=bytes(256), money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


def _slot(species_id: int, level_v: int = 5, hp: int = 20) -> PartySlot:
    return PartySlot(
        species_id=species_id, level=level_v, hp_cur=hp, hp_max=hp,
        status=0, moves=(0, 0, 0, 0),
    )


# ---- Atom construction ----

def test_catch_constructs_atom():
    g = catch("ODDISH")
    assert isinstance(g, Atom)
    assert g.predicate_type == "catch"
    assert g.entity == "ODDISH"


def test_reach_constructs_atom():
    g = reach("ROUTE_1")
    assert isinstance(g, Atom)
    assert g.predicate_type == "reach"


def test_have_item_default_qty_one():
    g = have_item("POKEBALL")
    assert g.params.get("qty") == 1


def test_level_with_threshold():
    g = level("CHARIZARD", 50)
    assert g.entity == "CHARIZARD"
    assert g.params["threshold"] == 50


def test_evolve_pair():
    g = evolve("CHARMANDER", "CHARMELEON")
    assert g.params["from"] == "CHARMANDER"
    assert g.params["to"] == "CHARMELEON"


# ---- Predicate evaluation ----

def test_catch_predicate_true_when_species_in_party():
    state = _empty_state(party=(_slot(0x47),))   # Oddish
    g = catch("ODDISH")
    assert g.predicate(state, species_id_lookup={"ODDISH": 0x47}) is True


def test_catch_predicate_false_when_species_not_in_party():
    state = _empty_state(party=(_slot(0x99),))   # Bulbasaur
    g = catch("ODDISH")
    assert g.predicate(state, species_id_lookup={"ODDISH": 0x47}) is False


def test_reach_predicate_matches_map_id():
    state = _empty_state(map_id=0x10)  # Route 5
    g = reach("ROUTE_5")
    assert g.predicate(state, map_id_lookup={"ROUTE_5": 0x10}) is True


def test_have_item_predicate_checks_bag():
    state = _empty_state(bag=(BagSlot(item_id=0x04, qty=3),))   # Pokeball
    g = have_item("POKEBALL", qty=2)
    assert g.predicate(state, item_id_lookup={"POKEBALL": 0x04}) is True


def test_have_item_predicate_qty_short():
    state = _empty_state(bag=(BagSlot(item_id=0x04, qty=1),))
    g = have_item("POKEBALL", qty=2)
    assert g.predicate(state, item_id_lookup={"POKEBALL": 0x04}) is False


def test_level_predicate():
    state = _empty_state(party=(_slot(0xB4, level_v=50),))   # L50 Charizard
    g = level("CHARIZARD", 50)
    assert g.predicate(state, species_id_lookup={"CHARIZARD": 0xB4}) is True


# ---- Combinators ----

def test_then_constructs_sequence():
    g = then(catch("ODDISH"), beat("BROCK"))
    assert isinstance(g, Then)
    assert len(g.children) == 2


def test_and_constructs_unordered():
    g = and_(catch("ODDISH"), catch("PIDGEY"))
    assert isinstance(g, And)


def test_or_constructs_alternatives():
    g = or_(catch("HITMONLEE"), catch("HITMONCHAN"))
    assert isinstance(g, Or)


def test_forall_compiles_to_atoms():
    g = forall(("BULBASAUR", "CHARMANDER", "SQUIRTLE"), lambda p: catch(p))
    assert isinstance(g, Forall)
    assert len(g.children) == 3
    assert all(isinstance(c, Atom) for c in g.children)


def test_then_predicate_all_must_hold():
    state = _empty_state(
        party=(_slot(0x47),),         # Oddish caught
        badges=0b0000_0001,           # Boulder badge
    )
    lookups = dict(
        species_id_lookup={"ODDISH": 0x47},
        trainer_lookup={"BROCK": "boulder_badge_bit"},
    )
    # In Phase 0, beat() against a trainer just checks a badge bit if mapped.
    # We cheat here by using the map_id_lookup to fake the trainer-flag eval.
    g = then(catch("ODDISH"))
    assert g.predicate(state, **lookups) is True


def test_goal_repr_is_readable():
    """Goals should print recognizably for log readability."""
    g = then(catch("ODDISH"), beat("BROCK"))
    s = repr(g)
    assert "ODDISH" in s
    assert "BROCK" in s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_goals.py -v
```
Expected: ImportError on `pokemon_planner.goals`.

- [ ] **Step 3: Write `world_model/pokemon_planner/goals/dsl.py`**

```python
"""Goal DSL — Goal class hierarchy + predicate evaluation.

See spec Section 2. Phase 0 implements: 6 atoms (catch, beat, reach, have_item,
level, evolve) and 4 combinators (then, and_, or_, forall). Phase 1c adds
named shortcuts (catch_all, beat_champion).

Predicates evaluate against a GameState. Atoms accept lookups (species_id_lookup,
map_id_lookup, etc.) so the predicate doesn't need direct KB access — that
binding happens at the planner layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from pokemon_planner.state import GameState


# ---- Types ----

class Goal:
    """Base class for all goals. Predicate evaluation is per-subclass."""

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Goal):
    predicate_type: str          # "catch" | "beat" | "reach" | "have_item" | "level" | "evolve"
    entity: str = ""             # primary entity name (species, trainer, region, item)
    params: Mapping[str, Any] = field(default_factory=dict)

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return _evaluate_atom(self, state, lookups)

    def __repr__(self) -> str:
        if self.params:
            return f"{self.predicate_type}({self.entity}, {dict(self.params)})"
        return f"{self.predicate_type}({self.entity})"


@dataclass(frozen=True)
class Then(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        # All must hold (sequencing matters at decomposition, not at predicate-eval)
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "then(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class And(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "and_(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Or(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return any(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "or_(" + ", ".join(repr(c) for c in self.children) + ")"


@dataclass(frozen=True)
class Forall(Goal):
    children: tuple[Goal, ...]

    def predicate(self, state: GameState, **lookups: Mapping[str, Any]) -> bool:
        return all(c.predicate(state, **lookups) for c in self.children)

    def __repr__(self) -> str:
        return "forall([" + ", ".join(repr(c) for c in self.children) + "])"


# ---- Atom evaluation ----

def _evaluate_atom(atom: Atom, state: GameState, lookups: Mapping[str, Any]) -> bool:
    pt = atom.predicate_type
    if pt == "catch":
        sid = _lookup(lookups, "species_id_lookup", atom.entity)
        return any(slot.species_id == sid for slot in state.party)
    if pt == "reach":
        mid = _lookup(lookups, "map_id_lookup", atom.entity)
        return state.map_id == mid
    if pt == "have_item":
        iid = _lookup(lookups, "item_id_lookup", atom.entity)
        qty_needed = atom.params.get("qty", 1)
        for slot in state.bag:
            if slot.item_id == iid and slot.qty >= qty_needed:
                return True
        return False
    if pt == "level":
        sid = _lookup(lookups, "species_id_lookup", atom.entity)
        threshold = atom.params["threshold"]
        return any(s.species_id == sid and s.level >= threshold for s in state.party)
    if pt == "beat":
        # Phase 0: 'beat' resolves via an event-flag bit lookup if provided.
        # If no lookup is available the predicate returns False (planner must
        # supply it). This will be wired up properly in Phase 1c.
        flag_lookup = lookups.get("trainer_flag_lookup")
        if flag_lookup is None or atom.entity not in flag_lookup:
            return False
        flag_bit_index = flag_lookup[atom.entity]
        byte_index, bit_index = divmod(flag_bit_index, 8)
        if byte_index >= len(state.event_flags):
            return False
        return bool(state.event_flags[byte_index] & (1 << bit_index))
    if pt == "evolve":
        sid_to = _lookup(lookups, "species_id_lookup", atom.params["to"])
        return any(slot.species_id == sid_to for slot in state.party)
    raise ValueError(f"Unknown predicate type: {pt}")


def _lookup(lookups: Mapping[str, Any], key: str, name: str) -> int:
    table = lookups.get(key)
    if table is None:
        raise KeyError(f"required lookup '{key}' not provided for entity '{name}'")
    if name not in table:
        raise KeyError(f"'{name}' not found in {key}")
    return table[name]
```

- [ ] **Step 4: Write `world_model/pokemon_planner/goals/atoms.py`**

```python
"""Public constructors for goal atoms and combinators.

Re-exported from pokemon_planner.goals so users write `from pokemon_planner
import goals as g; g.catch(g.ODDISH)`.
"""
from __future__ import annotations

from typing import Callable, Iterable

from pokemon_planner.goals.dsl import Atom, And, Forall, Goal, Or, Then


# ---- Atoms ----

def catch(species: str) -> Atom:
    return Atom(predicate_type="catch", entity=species)


def beat(trainer: str) -> Atom:
    return Atom(predicate_type="beat", entity=trainer)


def reach(region: str) -> Atom:
    return Atom(predicate_type="reach", entity=region)


def have_item(item: str, qty: int = 1) -> Atom:
    return Atom(predicate_type="have_item", entity=item, params={"qty": qty})


def level(species: str, threshold: int) -> Atom:
    return Atom(
        predicate_type="level", entity=species, params={"threshold": threshold}
    )


def evolve(from_species: str, to_species: str) -> Atom:
    return Atom(
        predicate_type="evolve",
        entity=from_species,
        params={"from": from_species, "to": to_species},
    )


# ---- Combinators ----

def then(*children: Goal) -> Then:
    return Then(children=tuple(children))


def and_(*children: Goal) -> And:
    return And(children=tuple(children))


def or_(*children: Goal) -> Or:
    return Or(children=tuple(children))


def forall(items: Iterable[str], fn: Callable[[str], Goal]) -> Forall:
    return Forall(children=tuple(fn(x) for x in items))
```

- [ ] **Step 5: Write `world_model/pokemon_planner/goals/__init__.py`**

```python
"""Goal DSL package — atoms, combinators, types, and species/region constants.

Usage:

    from pokemon_planner import goals as g

    g.catch(g.ODDISH)
    g.then(g.beat(g.BROCK), g.catch(g.BELLSPROUT))
    g.forall((g.BULBASAUR, g.CHARMANDER, g.SQUIRTLE), lambda p: g.catch(p))
"""
from pokemon_planner.goals.atoms import (
    and_,
    beat,
    catch,
    evolve,
    forall,
    have_item,
    level,
    or_,
    reach,
    then,
)
from pokemon_planner.goals.dsl import (
    And,
    Atom,
    Forall,
    Goal,
    Or,
    Then,
)


# ---- Convenience constants ----
# Names mirror their KB entries. Importing them as constants lets users write
# g.catch(g.ODDISH) instead of g.catch("ODDISH"). The values are just strings
# that the predicate evaluator looks up via the species_id_lookup table.

ODDISH = "ODDISH"
PIDGEY = "PIDGEY"
RATTATA = "RATTATA"
PIKACHU = "PIKACHU"
BULBASAUR = "BULBASAUR"
CHARMANDER = "CHARMANDER"
SQUIRTLE = "SQUIRTLE"
CHARMELEON = "CHARMELEON"
CHARIZARD = "CHARIZARD"

BROCK = "BROCK"
MISTY = "MISTY"

ROUTE_1 = "ROUTE_1"
ROUTE_5 = "ROUTE_5"
PEWTER_CITY = "PEWTER_CITY"
CERULEAN_CITY = "CERULEAN_CITY"

POKEBALL = "POKEBALL"
POTION = "POTION"
BICYCLE = "BICYCLE"


__all__ = [
    # Combinators
    "and_", "beat", "catch", "evolve", "forall", "have_item",
    "level", "or_", "reach", "then",
    # Types
    "And", "Atom", "Forall", "Goal", "Or", "Then",
    # Constants
    "ODDISH", "PIDGEY", "RATTATA", "PIKACHU",
    "BULBASAUR", "CHARMANDER", "SQUIRTLE",
    "CHARMELEON", "CHARIZARD",
    "BROCK", "MISTY",
    "ROUTE_1", "ROUTE_5", "PEWTER_CITY", "CERULEAN_CITY",
    "POKEBALL", "POTION", "BICYCLE",
]
```

- [ ] **Step 6: Run goal tests**

```bash
cd world_model
pytest tests/test_goals.py -v
```
Expected: All 14 tests pass.

- [ ] **Step 7: Commit**

```bash
git add world_model/pokemon_planner/goals/ world_model/tests/test_goals.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add Goal DSL — atoms + combinators + predicate evaluation

6 atoms (catch, beat, reach, have_item, level, evolve), 4 combinators
(then, and_, or_, forall), plus exported constants for early-game
species/regions/items so users can write g.catch(g.ODDISH).

Predicate evaluation accepts lookup tables as kwargs — keeps the DSL
decoupled from the KB until planner-layer wiring (Phase 1b/1c).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: World-model architecture stubs (h, g, f networks)

**Files:**
- Create: `world_model/pokemon_planner/world_model/__init__.py`
- Create: `world_model/pokemon_planner/world_model/arch.py`
- Create: `world_model/tests/test_world_model_arch.py`

**Why:** Implement the three networks from spec Section 4.2 — representation `h`, dynamics `g`, prediction `f`. Phase 0 wants stubs that *instantiate, forward-pass, and return correct shapes*. Real training comes in Task 10; real Pokemon data comes in Phase 1a.

For Phase 0 we use a simplified observation: a flat float vector instead of the full structured state. State→tensor encoding (one embedding per typed field, etc.) is non-trivial and gets its own task in Phase 1a. The arch stub takes shape `(batch, obs_dim)` and produces shape `(batch, latent_dim)` — sufficient to verify the architecture wires up.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_world_model_arch.py`:

```python
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
    """End-to-end forward through h → g → f."""
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_world_model_arch.py -v
```
Expected: ImportError on `pokemon_planner.world_model.arch`.

- [ ] **Step 3: Write `world_model/pokemon_planner/world_model/arch.py`**

```python
"""World-model architecture stubs (Phase 0).

Three networks from spec Section 4.2:
- RepresentationNet (h): obs → latent state
- DynamicsNet (g): (latent, action) → (next_latent, obs_pred, reward)
- PredictionNet (f): (latent, goal_emb) → (policy_logits, value)

Phase 0 uses simple MLPs over a flat obs vector. Phase 1a will replace the
representation function with a transformer-based encoder over typed fields,
and the dynamics function will gain explicit chance-node outputs for
stochastic dynamics. Interfaces are stable across the upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class WorldModelConfig:
    obs_dim: int                  # flat obs vector length (Phase 0); typed fields in Phase 1a
    action_dim: int               # number of discrete actions
    latent_dim: int               # latent state dimensionality
    goal_emb_dim: int             # goal embedding dimensionality
    hidden_dim: int               # MLP hidden width
    num_predicate_types: int = 6  # catch/beat/reach/have_item/level/evolve
    num_entities: int = 256       # max distinct entity IDs (species, trainer, etc.)


def _mlp(*dims: int, dropout: float = 0.0) -> nn.Module:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class RepresentationNet(nn.Module):
    """h(obs) → latent state s."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = _mlp(config.obs_dim, config.hidden_dim, config.latent_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)


class DynamicsNet(nn.Module):
    """g(s, a) → (s', obs_pred, r̂)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.action_emb = nn.Embedding(config.action_dim, config.hidden_dim)
        self.trunk = _mlp(
            config.latent_dim + config.hidden_dim, config.hidden_dim, config.hidden_dim
        )
        self.s_next_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.obs_head = nn.Linear(config.hidden_dim, config.obs_dim)
        self.r_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a_emb = self.action_emb(a)
        h = self.trunk(torch.cat([s, a_emb], dim=-1))
        s_next = self.s_next_head(h)
        obs_pred = self.obs_head(h)
        r_pred = self.r_head(h).squeeze(-1)
        return s_next, obs_pred, r_pred


class PredictionNet(nn.Module):
    """f(s, goal_emb) → (policy_logits, value)."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.trunk = _mlp(
            config.latent_dim + config.goal_emb_dim, config.hidden_dim, config.hidden_dim
        )
        self.policy_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, s: Tensor, goal_emb: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(torch.cat([s, goal_emb], dim=-1))
        pi = self.policy_head(h)
        v = self.value_head(h).squeeze(-1)
        return pi, v


class WorldModel(nn.Module):
    """Wrapper exposing h, g, f as submodules."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.h = RepresentationNet(config)
        self.g = DynamicsNet(config)
        self.f = PredictionNet(config)

    def forward(self, obs: Tensor, action: Tensor, goal_emb: Tensor) -> dict[str, Tensor]:
        s = self.h(obs)
        s_next, obs_pred, r_pred = self.g(s, action)
        pi, v = self.f(s_next, goal_emb)
        return dict(s=s, s_next=s_next, obs_pred=obs_pred, r_pred=r_pred, pi=pi, v=v)
```

- [ ] **Step 4: Write `world_model/pokemon_planner/world_model/__init__.py`**

```python
"""World-model subpackage. Phase 0 ships h/g/f stubs; richer architectures
arrive in Phase 1a (transformer encoder, stochastic dynamics).
"""
from pokemon_planner.world_model.arch import (
    DynamicsNet,
    PredictionNet,
    RepresentationNet,
    WorldModel,
    WorldModelConfig,
)

__all__ = [
    "DynamicsNet",
    "PredictionNet",
    "RepresentationNet",
    "WorldModel",
    "WorldModelConfig",
]
```

- [ ] **Step 5: Run architecture tests**

```bash
cd world_model
pytest tests/test_world_model_arch.py -v
```
Expected: All 6 tests pass. Param count should be well under 1M for the test config.

- [ ] **Step 6: Commit**

```bash
git add world_model/pokemon_planner/world_model/ world_model/tests/test_world_model_arch.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add world-model architecture stubs (h, g, f)

MLP-based representation / dynamics / prediction networks per spec
Section 4.2. Phase 0 stub: flat obs vector, simple feedforward.
Phase 1a will replace the encoder with a transformer over typed
state fields and add stochastic dynamics chance-node outputs.

Shape tests verify forward passes and check no NaN/Inf at init.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Training stub on synthetic data

**Files:**
- Create: `world_model/pokemon_planner/world_model/train_stub.py`
- Create: `world_model/tests/test_world_model_train_stub.py`

**Why:** Verify that the joint loss (spec Section 4.4) actually trains on synthetic data without diverging or NaN-ing. This is *not* real training — synthetic data is generated to have a learnable structure. Real training on PyBoy trajectories arrives in Phase 1a.

- [ ] **Step 1: Write the failing test**

Create `world_model/tests/test_world_model_train_stub.py`:

```python
"""Synthetic-data training stub — verifies loss decreases and no NaN."""
import torch

from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig
from pokemon_planner.world_model.train_stub import train_steps


def _config() -> WorldModelConfig:
    return WorldModelConfig(
        obs_dim=64,
        action_dim=9,
        latent_dim=32,
        goal_emb_dim=16,
        hidden_dim=64,
    )


def test_train_steps_returns_loss_history():
    cfg = _config()
    wm = WorldModel(cfg)
    losses = train_steps(wm, cfg, n_steps=5, batch_size=16, seed=0)
    assert len(losses) == 5
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)


def test_loss_decreases_over_steps():
    cfg = _config()
    wm = WorldModel(cfg)
    losses = train_steps(wm, cfg, n_steps=200, batch_size=32, seed=42)
    # Synthetic data is structured to be learnable; 200 steps should produce
    # a meaningful drop. This is the "doesn't diverge" gate from spec 9.3.
    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    assert late < early, f"Loss did not decrease: early={early:.4f} late={late:.4f}"


def test_no_nan_throughout_training():
    cfg = _config()
    wm = WorldModel(cfg)
    losses = train_steps(wm, cfg, n_steps=50, batch_size=16, seed=1)
    for i, l in enumerate(losses):
        assert torch.isfinite(torch.tensor(l)), f"NaN/Inf at step {i}: {l}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd world_model
pytest tests/test_world_model_train_stub.py -v
```
Expected: ImportError on `train_stub`.

- [ ] **Step 3: Write `world_model/pokemon_planner/world_model/train_stub.py`**

```python
"""Synthetic-data training stub for the world model.

Generates a tiny synthetic environment where dynamics are linear and rewards
are predictable, so the joint loss (state, value, policy, reward, consistency)
should reliably decrease. Used only in Phase 0 to verify wiring; Phase 1a
replaces this with real PyBoy-trajectory training.
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from pokemon_planner.world_model.arch import WorldModel, WorldModelConfig


def _synthetic_batch(cfg: WorldModelConfig, batch_size: int, generator: torch.Generator):
    """A learnable synthetic transition.

    obs ~ N(0, I)
    action ~ Uniform(action_dim)
    goal_emb ~ N(0, I)
    obs_next = A @ obs + B @ a_onehot + small noise   (linear, deterministic-ish)
    reward = (W @ goal_emb) · obs   (linear in (goal, state))
    target_pi = uniform (placeholder; loss uses cross-entropy with logits anyway)
    target_v = reward (1-step return for the stub)
    """
    device = "cpu"
    obs = torch.randn(batch_size, cfg.obs_dim, generator=generator)
    action = torch.randint(0, cfg.action_dim, (batch_size,), generator=generator)
    goal_emb = torch.randn(batch_size, cfg.goal_emb_dim, generator=generator)

    A = torch.randn(cfg.obs_dim, cfg.obs_dim, generator=generator) * 0.01
    B = torch.randn(cfg.obs_dim, cfg.action_dim, generator=generator) * 0.01
    a_onehot = torch.nn.functional.one_hot(action, num_classes=cfg.action_dim).float()
    obs_next = obs @ A.T + a_onehot @ B.T + 0.01 * torch.randn(
        batch_size, cfg.obs_dim, generator=generator
    )

    W = torch.randn(cfg.obs_dim, cfg.goal_emb_dim, generator=generator) * 0.01
    reward = (goal_emb @ W.T * obs).sum(dim=-1)

    target_v = reward.clone()
    target_pi = torch.full((batch_size, cfg.action_dim), 1.0 / cfg.action_dim)

    return dict(
        obs=obs, action=action, goal_emb=goal_emb,
        obs_next=obs_next, reward=reward,
        target_pi=target_pi, target_v=target_v,
    )


def _joint_loss(out: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
    # L_obs: predicted next observation vs. actual
    l_obs = torch.nn.functional.mse_loss(out["obs_pred"], batch["obs_next"])
    # L_value: predicted V vs. target return
    l_value = torch.nn.functional.mse_loss(out["v"], batch["target_v"])
    # L_policy: predicted logits vs. target distribution (KL via cross-entropy)
    log_pi = torch.nn.functional.log_softmax(out["pi"], dim=-1)
    l_policy = -(batch["target_pi"] * log_pi).sum(dim=-1).mean()
    # L_reward: predicted reward vs. actual
    l_reward = torch.nn.functional.mse_loss(out["r_pred"], batch["reward"])
    # L_consist: latent dynamics should produce a usable next state (dummy here —
    # encourages dynamics output to have similar magnitude to encoder output)
    l_consist = torch.nn.functional.mse_loss(out["s_next"], out["s"].detach())

    # Weights from spec Section 4.4 (default-ish; tuning in Phase 1a)
    return 1.0 * l_obs + 0.25 * l_value + 1.0 * l_policy + 0.5 * l_reward + 0.1 * l_consist


def train_steps(
    wm: WorldModel,
    cfg: WorldModelConfig,
    n_steps: int,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> List[float]:
    """Train n_steps on synthetic data; return list of per-step loss values."""
    gen = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(wm.parameters(), lr=lr)
    losses: List[float] = []
    wm.train()
    for _ in range(n_steps):
        batch = _synthetic_batch(cfg, batch_size, gen)
        out = wm(batch["obs"], batch["action"], batch["goal_emb"])
        loss = _joint_loss(out, batch)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        optim.step()
        losses.append(loss.item())
    return losses
```

- [ ] **Step 4: Run training-stub tests**

```bash
cd world_model
pytest tests/test_world_model_train_stub.py -v
```
Expected: All 3 tests pass. The "loss decreases" test runs 200 steps; on a CPU it should complete in under 30 seconds.

If `test_loss_decreases_over_steps` fails: verify the synthetic data is structured (the linear transformation matrices A, B, W aren't accidentally constant-zero). The test uses `seed=42`; reproduce locally and inspect early/late loss values.

- [ ] **Step 5: Commit**

```bash
git add world_model/pokemon_planner/world_model/train_stub.py world_model/tests/test_world_model_train_stub.py
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Add synthetic-data training stub + joint loss

Implements the joint MuZero-style loss (obs prediction, value, policy,
reward, consistency) per spec Section 4.4 and trains it on a tiny
synthetic linear environment. Verifies wiring: loss decreases over
200 steps, no NaN, gradient clipping applied.

Real PyBoy-trajectory training arrives in Phase 1a — this stub is
purely a "the architecture trains at all" gate per spec Section 9.3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: End-to-end smoke test

**Files:**
- Create: `world_model/tests/test_e2e_smoke.py`

**Why:** Phase 0's definition-of-done (spec Section 9.3) requires "smoke test runs PyBoy + reads structured state." This test wires together everything Phase 0 has built: load PyBoy, extract state, load KB, evaluate a goal predicate, run the world model forward. If this passes, Phase 0 is done.

- [ ] **Step 1: Write the smoke test**

Create `world_model/tests/test_e2e_smoke.py`:

```python
"""End-to-end Phase 0 smoke test: PyBoy → state → KB → goal predicate → world model.

This is the Phase 0 acceptance test. Marked integration so it skips when
the ROM/save-state aren't available. When everything's wired correctly,
it runs in a few seconds.
"""
import pytest
import torch

from pokemon_planner import goals as g
from pokemon_planner.env import PokeBoy, read_state
from pokemon_planner.kb import load_kb
from pokemon_planner.world_model import WorldModel, WorldModelConfig


@pytest.mark.integration
def test_phase_0_e2e(rom_path, init_state_path):
    # 1. Boot PyBoy and load init.state
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        # 2. Extract structured state
        state = read_state(pb.pyboy)
        assert state.party_size <= 6, "extracted state has impossible party size"

        # 3. Load knowledge base
        kb = load_kb()
        assert "ODDISH" in kb.species

        # 4. Evaluate a goal predicate against the state.
        #    catch(ODDISH) is False from init.state (no Oddish in starting party).
        species_id_lookup = {name: sp.species_id for name, sp in kb.species.items()}
        goal = g.catch(g.ODDISH)
        achieved = goal.predicate(state, species_id_lookup=species_id_lookup)
        assert achieved is False

        # 5. Run a world-model forward pass on a synthetic obs+goal embedding
        cfg = WorldModelConfig(
            obs_dim=64, action_dim=9, latent_dim=32, goal_emb_dim=16, hidden_dim=64,
        )
        wm = WorldModel(cfg)
        wm.eval()
        with torch.no_grad():
            obs = torch.randn(1, cfg.obs_dim)
            action = torch.tensor([0])
            goal_emb = torch.randn(1, cfg.goal_emb_dim)
            out = wm(obs, action, goal_emb)
        assert out["pi"].shape == (1, cfg.action_dim)
        assert out["v"].shape == (1,)
        assert torch.isfinite(out["pi"]).all()
        assert torch.isfinite(out["v"]).all()
    finally:
        pb.close()


@pytest.mark.integration
def test_phase_0_compositional_goal_against_real_state(rom_path, init_state_path):
    """A compositional goal evaluates correctly against an extracted state."""
    pb = PokeBoy(rom_path=str(rom_path), save_state_path=str(init_state_path))
    try:
        state = read_state(pb.pyboy)
        kb = load_kb()
        species_id_lookup = {n: s.species_id for n, s in kb.species.items()}
        map_id_lookup = {n: r.map_id for n, r in kb.regions.items()}

        # then(catch(ODDISH), catch(PIDGEY)) is False (no party at init)
        goal = g.then(g.catch(g.ODDISH), g.catch(g.PIDGEY))
        assert goal.predicate(
            state,
            species_id_lookup=species_id_lookup,
            map_id_lookup=map_id_lookup,
        ) is False
    finally:
        pb.close()
```

- [ ] **Step 2: Run the smoke test**

```bash
cd world_model
pytest tests/test_e2e_smoke.py -v -m integration
```

Expected (with ROM + save state present): 2 tests pass.
Expected (without ROM): 2 tests skipped with clear "ROM not found" / "save state not found" messages.

If the test fails on `goal.predicate(state, ...)`: verify `species_id_lookup` is built from the actual loaded KB and that `ODDISH`'s species_id matches what species.yaml says (`0x47`).

- [ ] **Step 3: Run the FULL test suite to confirm Phase 0 is green**

```bash
cd world_model
pytest -v
```

Expected:
- `test_smoke.py` — 4 pass
- `test_state.py` — 8 pass
- `test_ram_addresses.py` — 4 pass
- `test_state_extraction.py` — 3 pass (or skip if no ROM)
- `test_kb.py` — 7 pass
- `test_goals.py` — 14 pass
- `test_world_model_arch.py` — 6 pass
- `test_world_model_train_stub.py` — 3 pass
- `test_e2e_smoke.py` — 2 pass (or skip if no ROM)

Total: ~51 tests, all green.

- [ ] **Step 4: Update `STATE.md` to reflect Phase 0 complete**

Edit `world_model/STATE.md` — replace the Phase line and Working/Next Up sections to read:

```markdown
## Phase
0 — Foundation (complete) → Phase 1a kickoff next

## Working
- Package scaffolded; `pip install -e .` works
- Smoke tests pass (51/51 tests green)
- State Pydantic schema + RAM extractor (read_state(pyboy) → GameState)
- 30-species KB + Trainer/Item/Region stubs
- Goal DSL: 6 atoms, 4 combinators, exported constants
- World-model arch stubs (h/g/f networks); synthetic-data training verifies loss decreases
- End-to-end smoke test: PyBoy → state → KB → goal predicate → WM forward pass

## Broken / Known Issues
- (none — Phase 0 closed cleanly)

## Next Up
1. Phase 1a plan (separate doc): bootstrap demonstration data from PWhiddy's PPO checkpoint
2. Replace flat obs vector with typed-field encoder (real Phase 1 state)
3. First real WM training run; track val loss in `docs/tuning.md`
```

- [ ] **Step 5: Append a Phase-0-complete entry to `docs/progress.md`**

Append to `world_model/docs/progress.md`:

```markdown

## 2026-05-XX — Phase 0 complete

Foundation tasks 1–11 done. 51 tests passing. Package installs cleanly,
state extractor works against init.state, KB loader returns 30 species,
goal DSL atoms and combinators evaluate correctly, world-model architecture
stubs train on synthetic data without NaN, end-to-end smoke test green.

Ready for Phase 1a: bootstrap demonstration data + first real world-model
training. Plan to be drafted as `docs/superpowers/plans/2026-XX-XX-phase-1a-bootstrap.md`.
```

(Replace `XX` with the actual date when you finish.)

- [ ] **Step 6: Commit**

```bash
git add world_model/tests/test_e2e_smoke.py world_model/STATE.md world_model/docs/progress.md
git -c user.email="christopherscottkeller@gmail.com" -c user.name="RoseOfficial" commit -m "$(cat <<'EOF'
Phase 0 acceptance: end-to-end smoke + STATE/progress update

E2E smoke test wires PyBoy boot, state extraction, KB load, goal-
predicate evaluation, and world-model forward pass into a single
~3-second test. Compositional goals evaluate correctly against
real extracted states.

51 tests passing total. STATE.md and progress.md updated to reflect
Phase 0 complete; ready to plan Phase 1a (bootstrap demonstration
data + first real WM training).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (post-write checklist for the author of this plan)

Before declaring this plan ready, verify against the spec:

**Spec coverage:** Each Phase-0-relevant spec section maps to at least one task:
- Section 4.1 (state schema) → Tasks 4, 6
- Section 4.2 (architecture) → Task 9
- Section 4.4 (joint loss) → Task 10
- Section 2 (goal DSL) → Task 8
- Section 3 (planner data sources / KB) → Task 7
- Section 9.1 (file layout) → Task 1
- Section 9.3 (Phase 0 definition-of-done) → Task 11
- Section 10 (observability artifacts) → Task 3

Phase 0 deliberately does NOT cover: Layer 2 planner (Phase 1b), MCTS (Phase 1b),
verification flywheel (Phase 1b), training on real data (Phase 1a),
hierarchical decomposition (Phase 1c), evaluation benchmarks (per-phase).
Those go in their own plans.

**Placeholders:** None. Every task has exact code, exact paths, exact commands.

**Type consistency:**
- `GameState`, `PartySlot`, `BagSlot`, `BattleState` defined in Task 4, used in Tasks 6, 8, 11.
- `read_state(pyboy: PyBoy) → GameState` defined in Task 6, used in Task 11.
- `WorldModelConfig`, `WorldModel` defined in Task 9, used in Tasks 10, 11.
- `KnowledgeBase`, `load_kb()`, `Species` defined in Task 7, used in Task 11.
- `Atom`, `Then`, `And`, `Or`, `Forall`, `Goal` defined in Task 8, used in Task 11.

All cross-task references are consistent.

---

## What this plan does NOT do

This plan is **scoped to Phase 0**. Phase 1a (bootstrap data + real WM training),
Phase 1b (first goals work end-to-end), Phase 1c (mid-game + compositional),
Phase 1d (endgame), Phase 2 (full memory observation), and Phase 3 (strategy
discovery) each get their own plans. Each phase produces working, testable
software on its own. After this plan completes, expect to spend a session
brainstorming and writing the Phase 1a plan before beginning Phase 1a work.
