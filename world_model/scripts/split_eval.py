"""One-off: move every 20th episode shard to data/replay_buffer/eval/.

Run after bootstrap_demos.py has populated the replay buffer. Holds out 5%
of episodes for validation by `eval/` subdirectory convention.
"""
import json
import shutil
from pathlib import Path

REPLAY_ROOT = Path("data/replay_buffer")
EVAL_ROOT = REPLAY_ROOT / "eval"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    meta_path = REPLAY_ROOT / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"meta.json not found at {meta_path}. "
            f"Run bootstrap_demos.py first to populate the replay buffer."
        )
    meta = json.loads(meta_path.read_text())

    eval_episode_ids = []
    moved = 0
    for shard in sorted(REPLAY_ROOT.glob("traj_*.parquet")):
        # Filename format: traj_{source}_{episode_id}.parquet
        parts = shard.stem.split("_")
        try:
            episode_id = int(parts[-1])
        except ValueError:
            continue
        if episode_id % 20 == 0:    # 5% held out
            shutil.move(str(shard), str(EVAL_ROOT / shard.name))
            eval_episode_ids.append(episode_id)
            moved += 1

    meta["eval_split_episodes"] = eval_episode_ids
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Moved {moved} shards to eval/. Eval episode IDs (first 10): {eval_episode_ids[:10]}")


if __name__ == "__main__":
    main()
