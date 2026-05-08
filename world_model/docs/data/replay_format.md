# Replay Buffer On-Disk Format

The replay buffer persists demonstration trajectories to Parquet files under `data/replay_buffer/`. This format is shared by Phase 1a's bootstrap extractor and read by the training loop.

## Layout

```
data/replay_buffer/
├── meta.json                            # global metadata
├── traj_v2_1310720_<batch_id>.parquet   # per-source episode batches
├── traj_v2_2621440_<batch_id>.parquet
├── traj_v2_3932160_<batch_id>.parquet
├── traj_v2_5242880_<batch_id>.parquet
└── eval/
    └── traj_eval_<source>_<batch_id>.parquet   # held-out 5%
```

## Parquet schema

| Column | Type | Notes |
|---|---|---|
| `episode_id` | int64 | Globally unique across all sources |
| `step_index` | int32 | 0-based within the episode |
| `state_bytes` | binary | msgpack-serialized GameState |
| `action` | int8 | 0–8 |
| `reward` | float32 | v2's shaped reward |
| `done` | bool | true on last step of episode |
| `mc_return` | float32 | Computed at extraction time, gamma=0.997 |
| `source` | string | Checkpoint name (e.g. "v2_1310720") |
| `seed` | int64 | RNG seed for the episode (reproducibility) |

## meta.json

```json
{
  "version": "0.1",
  "schema_version": "phase_1a",
  "sources": {
    "v2_1310720": {"path": "../v2/runs/poke_1310720_steps.zip", "n_episodes": 120, "n_steps": 250000},
    "v2_2621440": {"path": "...", "n_episodes": 120, "n_steps": 250000},
    "v2_3932160": {"path": "...", "n_episodes": 120, "n_steps": 250000},
    "v2_5242880": {"path": "...", "n_episodes": 120, "n_steps": 250000}
  },
  "total_episodes": 480,
  "total_steps": 1000000,
  "eval_split_episodes": []
}
```

## Resumability

`bootstrap_demos.py` writes one Parquet shard per episode atomically. After each episode it updates `meta.json`. On resume, the script reads `meta.json` and skips already-extracted (source, episode_id) pairs.

## Disk budget

~720 bytes per row × 1M rows ≈ 720 MB raw. Parquet snappy compression brings this to ~300–400 MB total.
