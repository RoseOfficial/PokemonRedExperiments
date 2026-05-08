# ADR-0002: Full joint loss with placeholder value/policy targets

Status: Accepted
Date: 2026-05-07
Supersedes: none

## Context

MuZero's full joint loss has 5 components: state prediction (dynamics), value, policy, reward, consistency. Value and policy targets normally come from MCTS visit counts and Monte-Carlo simulated returns. Phase 1a does not have MCTS yet — that's Phase 1b.

For Phase 1a's first real training run, three options for the loss:

1. **Dynamics-only:** Train h and g rigorously on next-state prediction + consistency. Leave f untrained until Phase 1b.
2. **Dynamics + behavioral cloning on demo actions:** Train policy head π via BC on the PPO demo's action distribution. Value head still untrained.
3. **Full joint loss with placeholder targets:** All 5 loss components active. Value targets = MC returns over PPO's shaped reward. Policy targets = BC on demo actions. Single dummy goal embedding throughout.

## Decision

Adopt option 3 — **full joint loss with placeholder value/policy targets**. The architecture is correct; the targets are temporary.

### Placeholder-target details

- **Policy targets:** `target_pi[t] = one_hot(actions[t], num_classes=9)` — BC on PPO's actions.
- **Value targets:** `target_v[t] = sum(γ^i * rewards[t+i] for i in range(remaining_episode_length))` — Monte-Carlo returns over PPO's shaped reward, γ=0.997. Computed once at extraction time, stored in the replay buffer.
- **Reward targets:** `target_r[t+1] = rewards[t+1]` — v2's shaped reward.
- **Goal embedding:** A single learned 384d vector (`DummyGoalEmbedding`) used for all training samples. PPO bootstrap data has no goal labels.

### Phase 1b retraining contract

At Phase 1b's start, MCTS is wired in and produces real goal-conditioned targets:
- Policy targets ← MCTS visit-distribution
- Value targets ← MCTS simulated returns across actual goals
- Goal embedding ← real `GoalEmbedder` over sampled goals

The f-head is **retrained from scratch** in Phase 1b. The h and g networks (representation + dynamics) carry over.

## Consequences

Positive:
- End-to-end training pipeline working from day one.
- Surface area exercised across all heads, catching bugs (shape mismatches, NaN issues) on real data immediately.

Negative:
- f-head learns "imitate PPO under a dummy goal" — a worse-than-random init for true goal-conditioned planning. f-head must retrain from scratch in Phase 1b.
- ~7M parameters of f-head training compute is partially wasted in Phase 1a.
- Risk of "convergence to wrong objective" at the f-head, masking issues that only Phase 1b's MCTS would surface.

The 7M wasted compute is an acceptable cost: f-head is small, retraining converges in ~10K steps under MCTS targets.

## Alternatives considered

- **Dynamics-only:** Cleanest but leaves the f-head architecturally untested in Phase 1a. Discovered bugs around f-head shapes / goal-emb wiring would surface in Phase 1b instead, when MCTS is also new.
- **Dynamics + BC on policy only:** Avoids the value-head waste but doesn't materially simplify the pipeline.
