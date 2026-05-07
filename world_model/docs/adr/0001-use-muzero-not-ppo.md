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
