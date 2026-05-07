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
