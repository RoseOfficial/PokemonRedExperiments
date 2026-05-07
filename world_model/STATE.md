# Project State — 2026-05-07

## Phase
0 — Foundation (COMPLETE) → Phase 1a kickoff next

## Working
- Package scaffolded; `pip install -e .` works
- Full test suite green: **54/54 passing** (Python 3.11.9, Windows)
- State Pydantic schema (`pokemon_planner.state.GameState` + PartySlot/BagSlot/BattleState)
- RAM address constants (`pokemon_planner._ram_addresses`) sourced from pret/pokered
- PyBoy wrapper + structured state extractor (`read_state(pyboy) → GameState`) verified against `init.state`
- 30-species KB + Trainer/Item/Region stubs (`load_kb()` returns frozen KnowledgeBase)
- Goal DSL: 6 atoms (catch/beat/reach/have_item/level/evolve) + 4 combinators (then/and_/or_/forall) + exported constants
- World-model architecture stubs (h/g/f networks); ~370K params at test config
- Synthetic-data training stub: joint loss decreases over 200 steps, no NaN
- E2E smoke: PyBoy → state → KB → goal predicate → world-model forward pass, all wired
- Observability scaffolding (this file, ADR-0001, architecture.md, progress.md, tuning.md, CLAUDE_BOOTSTRAP.md)

## Broken / Known Issues
- Tile-collision extraction in `env.py` is a 256B-zeros stub — real implementation deferred to Phase 1a
- `pyproject.toml` doesn't list runtime deps; users must install `requirements_windows.txt` separately. Will fix in Phase 1a if deps shift.

## Next Up
1. **Phase 1a plan** (separate doc): bootstrap demonstration data from PWhiddy's PPO checkpoint + existing v2 checkpoints
2. Replace flat obs vector with typed-field encoder over actual `GameState` fields
3. First real world-model training run; track val loss in `docs/tuning.md`
4. Implement real tile-collision extraction (currently zeros)

## Recent Changes (last 7 days)
- 2026-05-07: Project kicked off; spec committed at `docs/superpowers/specs/2026-05-07-pokemon-world-model-search-design.md`
- 2026-05-07: ADR-0001 captures the MuZero-vs-PPO decision
- 2026-05-07: Phase 0 scaffolding tasks 1–3 complete
- 2026-05-07: Tasks 4–10 complete: state schema, RAM addresses, extractor, KB+30 species, goal DSL, WM arch stubs, training stub
- 2026-05-07: Task 11 (E2E smoke) complete; full suite 54/54 green; **Phase 0 closed**

## Where To Look
- Architecture: `docs/architecture.md`
- Recent decisions: `docs/adr/`
- Progress: `docs/progress.md`
- Tuning experiments: `docs/tuning.md`
- Phase 0 plan: `../docs/superpowers/plans/2026-05-07-phase-0-foundation.md`
