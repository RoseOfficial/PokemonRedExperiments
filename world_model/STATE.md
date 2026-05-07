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
