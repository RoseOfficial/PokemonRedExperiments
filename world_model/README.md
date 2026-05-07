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
