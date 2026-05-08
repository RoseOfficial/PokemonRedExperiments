"""Bootstrap data extraction — run v2 PPO checkpoints, record demonstration trajectories.

Per spec Section 3:
- Run each of the 4 v2/runs/poke_*_steps.zip checkpoints
- ~120 episodes per checkpoint, ~2,083 decisions per episode -> ~250K per source
- Total target: ~1M demonstration decisions
- v2 PPO acts in v2's env (its native obs); we record OUR typed GameState in parallel

Usage:
    python scripts/bootstrap_demos.py [--source v2_1310720] [--max-episodes 120] [--max-steps 50000]

Environment requirements:
    - ../PokemonRed.gb (ROM)
    - ../init.state (PyBoy save state past title)
    - ../v2/runs/poke_<N>_steps.zip (PPO checkpoints)
    - v2 env imports must work (uses v2/red_gym_env_v2.py via sys.path manipulation)

Resumable: idempotent on (source, episode_id) -- re-running picks up from where it left off.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Allow importing v2's env from world_model/scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2"))

from pokemon_planner.data.replay import ReplayBuffer  # noqa: E402
from pokemon_planner.data.trajectory import Trajectory, TrajectoryStep  # noqa: E402
from pokemon_planner.env import read_state  # noqa: E402


# v2's action space (from v2/red_gym_env_v2.py): 0=DOWN, 1=LEFT, 2=RIGHT,
# 3=UP, 4=A, 5=B, 6=START. Our 9-action space adopts the same order, with
# 7=SELECT and 8=NO-OP appended (these never appear in v2 PPO output but
# are reserved for Phase 1b/1c agents that exercise the full button range).
V2_ACTION_TO_OURS: dict[int, int] = {
    0: 0,  # DOWN
    1: 1,  # LEFT
    2: 2,  # RIGHT
    3: 3,  # UP
    4: 4,  # A
    5: 5,  # B
    6: 6,  # START
}


# Source key -> checkpoint filename (under v2/runs/)
SOURCES: dict[str, str] = {
    "v2_1310720": "poke_1310720_steps.zip",
    "v2_2621440": "poke_2621440_steps.zip",
    "v2_3932160": "poke_3932160_steps.zip",
    "v2_5242880": "poke_5242880_steps.zip",
}


@dataclass
class ExtractionConfig:
    sources: list[str]
    max_episodes_per_source: int = 120
    max_steps_per_episode: int = 2500
    buffer_root: Path = Path("data/replay_buffer")


def _infer_source_name(ckpt_path: Path) -> str:
    """Infer source key from the checkpoint filename."""
    for k, v in SOURCES.items():
        if v == ckpt_path.name:
            return k
    raise ValueError(
        f"Unknown checkpoint filename: {ckpt_path.name!r}. "
        f"Expected one of: {list(SOURCES.values())}"
    )


def extract_one_episode(
    *,
    ckpt_path: Path,
    rom_path: Path,
    init_state_path: Path,
    episode_id: int,
    seed: int,
    max_steps: int,
    buffer: ReplayBuffer,
    source_name: str | None = None,
    session_root: Path | None = None,
) -> Trajectory | None:
    """Run one episode of v2 PPO and capture our typed states. Idempotent.

    Args:
        ckpt_path: Path to a v2 PPO .zip checkpoint.
        rom_path: Path to PokemonRed.gb.
        init_state_path: Path to init.state (PyBoy save state).
        episode_id: Integer episode index (used for idempotency).
        seed: RNG seed passed to env.reset().
        max_steps: Maximum number of env steps to run.
        buffer: ReplayBuffer instance to persist to.
        source_name: If None, inferred from ckpt_path.name via SOURCES.
        session_root: Root directory for per-episode PyBoy session subdirs.
            If None, a fresh OS temp dir is created. Pass tmp_path in tests.

    Returns:
        The extracted Trajectory, or None if (source_name, episode_id) was
        already in the buffer (idempotent skip).
    """
    from stable_baselines3 import PPO
    from red_gym_env_v2 import RedGymEnv  # type: ignore  # from v2/

    if source_name is None:
        source_name = _infer_source_name(ckpt_path)

    # Skip if already extracted (idempotency)
    if (source_name, episode_id) in buffer._index.seen_episodes:
        return None

    # Per-episode session subdir — avoids cross-episode state accumulation and
    # works on Windows (Path("/tmp/...") resolves to C:\tmp which may not exist).
    if session_root is None:
        session_root = Path(tempfile.mkdtemp(prefix="poke_bootstrap_"))
    session_path = session_root / source_name / f"ep_{episode_id}"
    session_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state_path),
        "max_steps": max_steps,
        "print_rewards": False,  # Must be False on Windows (deadlocks SubprocVecEnv pipes)
        "save_video": False,
        "fast_video": False,
        "session_path": session_path,
        "gb_path": str(rom_path),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 0.5,    # matches v2/baseline_fast_v2.py training config
        "extra_buttons": False,
        "explore_weight": 3,
    }
    # RedGymEnv.__init__ opens "events.json" with a bare relative path, so it
    # must be constructed with v2/ as the working directory.
    _orig_cwd = os.getcwd()
    os.chdir(str(REPO_ROOT / "v2"))
    try:
        env = RedGymEnv(env_config)
    finally:
        os.chdir(_orig_cwd)
    obs, _ = env.reset(seed=seed)

    model = PPO.load(
        str(ckpt_path), env=env, device="cpu",
        custom_objects={"lr_schedule": 0, "clip_range": 0},
    )

    steps: list[TrajectoryStep] = []
    done = False
    step_idx = 0
    cumulative_reward = 0.0
    while not done and step_idx < max_steps:
        action, _ = model.predict(obs, deterministic=False)
        v2_action = int(action)
        our_action = V2_ACTION_TO_OURS.get(v2_action, 8)  # fall back to NO-OP (8)

        # Capture our typed state BEFORE stepping -- represents state agent acts on
        state = read_state(env.pyboy)

        obs, reward, terminated, truncated, info = env.step(v2_action)
        done = bool(terminated or truncated)
        cumulative_reward += float(reward)

        steps.append(TrajectoryStep(
            state=state,
            action=our_action,
            reward=float(reward),
            done=done,
            info={
                "source": source_name,
                "episode_id": episode_id,
                "step_index": step_idx,
                "cumulative_reward": cumulative_reward,
            },
        ))
        step_idx += 1

    env.close()

    traj = Trajectory(
        steps=tuple(steps),
        source=source_name,
        episode_id=episode_id,
        seed=seed,
    )
    buffer.add(traj, priority=0.3)
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run v2 PPO checkpoints and record demonstration trajectories."
    )
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()) + ["all"],
        default="all",
        help="Which checkpoint(s) to run (default: all 4).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=120,
        help="Episodes per checkpoint (default: 120).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2500,
        help="Max env steps per episode (default: 2500).",
    )
    parser.add_argument(
        "--buffer-root",
        type=Path,
        default=Path("data/replay_buffer"),
        help="Root directory for the Parquet replay buffer.",
    )
    args = parser.parse_args()

    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]
    buffer = ReplayBuffer(root=args.buffer_root)

    rom = REPO_ROOT / "PokemonRed.gb"
    init_state = REPO_ROOT / "init.state"
    if not rom.exists():
        sys.exit(f"ROM not found: {rom}")
    if not init_state.exists():
        sys.exit(f"Save state not found: {init_state}")

    for source in sources:
        ckpt = REPO_ROOT / "v2" / "runs" / SOURCES[source]
        if not ckpt.exists():
            print(f"[bootstrap] SKIP {source}: checkpoint not found at {ckpt}")
            continue
        print(f"[bootstrap] === Source: {source} ===")
        session_root = Path(tempfile.mkdtemp(prefix="poke_bootstrap_"))
        for episode_id in range(args.max_episodes):
            seed = (hash(source) ^ episode_id) & 0x7FFF_FFFF
            try:
                result = extract_one_episode(
                    ckpt_path=ckpt,
                    rom_path=rom,
                    init_state_path=init_state,
                    episode_id=episode_id,
                    seed=seed,
                    max_steps=args.max_steps,
                    buffer=buffer,
                    source_name=source,
                    session_root=session_root,
                )
                if result is None:
                    print(f"[bootstrap] {source} ep {episode_id}: skipped (already extracted)")
                else:
                    print(f"[bootstrap] {source} ep {episode_id}: {result.length} steps")
            except Exception as exc:
                print(f"[bootstrap] {source} ep {episode_id} FAILED: {exc}")
                continue

    print(f"[bootstrap] DONE. Buffer size: {buffer.size} steps total.")


if __name__ == "__main__":
    main()
