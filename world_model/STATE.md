# Project State — 2026-05-08

## Phase
1a — code complete; awaiting production training run

## Working
- All Phase 0 deliverables (carried over)
- ReplayBuffer with Parquet persistence (Task 2)
- Real tile-collision extraction in env.py with 8 covered tilesets (Tasks 3-4)
- Bootstrap demonstration data extraction script (Task 5)
- Goal embedding (DummyGoalEmbedding for Phase 1a placeholder; GoalEmbedder for Phase 1b) (Task 6)
- Typed-field tokenizer (44 tokens × 384d) (Task 7)
- Full transformer h/g/f architecture (~57.5M params) (Task 8)
- Joint loss with placeholder value/policy targets (Task 9)
- Full-state checkpoint module (Task 10)
- W&B logger (no-op when WANDB_API_KEY unset) (Task 11)
- Eval-during-training + DoD gate (Task 12)
- Training loop function with k-step unroll + AMP + checkpointing (Task 13)
- CLI + go_forever (bash + ps1) + ADR-0002 (Task 14)
- split_eval.py + smoke-test config (Task 15)
- Full test suite: 109 passing, 4 xfailed (Phase 0 stubs)

## Broken / Known Issues
- f-head trained on placeholder targets (BC on PPO + MC of shaped reward). Will be retrained from scratch in Phase 1b once MCTS exists. ADR-0002.
- 16 tilesets uncovered (Phase 1c fills the rest); affected late-game maps fall through to DEFAULT_TABLE (all walkable, suboptimal)
- Phase 0 train_stub tests xfail — interface changed in Task 8

## Next Up
1. **Run bootstrap extraction** end-to-end on all 4 v2 PPO checkpoints (~24h compute):
   ```
   cd world_model
   python scripts/bootstrap_demos.py
   ```
2. **Run split_eval** to hold out 5% of episodes for validation:
   ```
   python scripts/split_eval.py
   ```
3. **Smoke-test the training pipeline** with phase_1a_smoke.yaml (100 steps, tiny model):
   ```
   python scripts/train.py --config configs/phase_1a_smoke.yaml --checkpoint-dir runs/smoke
   ```
4. **Launch production training** via go_forever:
   ```
   ./scripts/go_forever.sh                 # bash
   .\scripts\go_forever.ps1                # PowerShell
   ```
5. **Validate** against the DoD gate (>=80% on acc/map_id, acc/x, acc/y, acc/party_species_slot_0). Record per-field accuracies in docs/tuning.md.

## Recent Changes (last 7 days)
- 2026-05-07: Phase 1a spec + plan + ADR-0002 committed
- 2026-05-07-08: Tasks 1-15 implementation complete (15 tasks across 2 days)
- 2026-05-08: Phase 1a structure complete; awaiting production training run

## Where To Look
- Architecture: `docs/architecture.md`
- Recent decisions: `docs/adr/`
- Progress: `docs/progress.md`
- Tuning experiments: `docs/tuning.md`
- Phase 1a plan: `../docs/superpowers/plans/2026-05-07-phase-1a-bootstrap-and-first-training.md`
