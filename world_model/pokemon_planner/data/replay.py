"""Priority-weighted replay buffer with Parquet on-disk persistence.

Phase 1a uses a single source ("demo", priority 0.3). Phase 1b adds
divergence/success/exploration sources without changing this interface.

See docs/data/replay_format.md for the on-disk format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pokemon_planner.data.trajectory import (
    Trajectory,
    deserialize_state,
    serialize_state,
)
from pokemon_planner.state import GameState


# Parquet schema (matches docs/data/replay_format.md)
_PARQUET_SCHEMA = pa.schema([
    ("episode_id", pa.int64()),
    ("step_index", pa.int32()),
    ("state_bytes", pa.binary()),
    ("action", pa.int8()),
    ("reward", pa.float32()),
    ("done", pa.bool_()),
    ("mc_return", pa.float32()),
    ("source", pa.string()),
    ("seed", pa.int64()),
    ("priority", pa.float32()),
])


@dataclass
class Batch:
    """A sampled batch of (k+1)-length trajectory windows."""

    states: list[list[GameState]]    # outer=batch, inner=k+1 states
    actions: list[list[int]]          # outer=batch, inner=k actions
    rewards: list[list[float]]        # outer=batch, inner=k rewards
    mc_returns: list[list[float]]    # outer=batch, inner=k mc_returns
    sources: list[str]                # one per batch element


class _RowEntry(NamedTuple):
    """One row in the in-memory replay index."""

    file_idx: int
    row_in_file: int
    episode_id: int
    step_index_in_episode: int
    episode_length: int
    priority: float


@dataclass
class _Index:
    """In-memory index over all rows in the Parquet shards."""

    file_paths: list[Path] = field(default_factory=list)
    rows: list[_RowEntry] = field(default_factory=list)
    seen_episodes: set[tuple[str, int]] = field(default_factory=set)
    _cached_weights: np.ndarray | None = None  # normalized priorities, lazy


class ReplayBuffer:
    """Priority-weighted sampler over Parquet shards.

    Usage::

        buf = ReplayBuffer(root=Path("data/replay_buffer"))
        buf.add(traj, priority=0.3)               # writes a Parquet shard
        batch = buf.sample_batch(64, k_unroll=5)  # returns a Batch
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root / "meta.json"
        self._meta = self._load_or_create_meta()
        self._index = _Index()
        self._rebuild_index()

    @property
    def size(self) -> int:
        """Total number of steps across all loaded shards."""
        return len(self._index.rows)

    def add(self, traj: Trajectory, priority: float) -> None:
        """Append one trajectory as a single Parquet shard. Idempotent on (source, ep)."""
        key = (traj.source, traj.episode_id)
        if key in self._index.seen_episodes:
            return  # already extracted; skip

        mc_returns = traj.mc_returns(gamma=self._meta.get("gamma", 0.997))
        rows: list[dict] = []
        for i, step in enumerate(traj.steps):
            rows.append({
                "episode_id": int(traj.episode_id),
                "step_index": int(i),
                "state_bytes": serialize_state(step.state),
                "action": int(step.action),
                "reward": float(step.reward),
                "done": bool(step.done),
                "mc_return": float(mc_returns[i]),
                "source": str(traj.source),
                "seed": int(traj.seed),
                "priority": float(priority),
            })

        table = pa.Table.from_pylist(rows, schema=_PARQUET_SCHEMA)
        shard_name = f"traj_{traj.source}_{traj.episode_id}.parquet"
        shard_path = self.root / shard_name
        pq.write_table(table, shard_path, compression="snappy")

        # Update meta
        src = self._meta["sources"].setdefault(traj.source, {"n_episodes": 0, "n_steps": 0})
        src["n_episodes"] += 1
        src["n_steps"] += traj.length
        self._meta["total_episodes"] += 1
        self._meta["total_steps"] += traj.length
        self._save_meta()

        # Update index
        file_idx = len(self._index.file_paths)
        self._index.file_paths.append(shard_path)
        for i in range(traj.length):
            self._index.rows.append(
                _RowEntry(
                    file_idx=file_idx,
                    row_in_file=i,
                    episode_id=traj.episode_id,
                    step_index_in_episode=i,
                    episode_length=traj.length,
                    priority=priority,
                )
            )
        self._index.seen_episodes.add(key)
        self._index._cached_weights = None  # invalidate on new data

    def sample_batch(self, batch_size: int, k_unroll: int) -> Batch:
        """Sample B trajectory windows of length k+1.

        Windows that span episode boundaries are rejected and resampled.

        Args:
            batch_size: Number of windows to return.
            k_unroll: Unroll depth; each window has k+1 states and k actions/rewards.

        Returns:
            A :class:`Batch` with lists of length ``batch_size``.

        Raises:
            RuntimeError: If the buffer is empty or too small to fill the batch.
        """
        states: list[list[GameState]] = []
        actions: list[list[int]] = []
        rewards: list[list[float]] = []
        mc_returns_out: list[list[float]] = []
        sources: list[str] = []

        attempts = 0
        max_attempts = batch_size * 10
        while len(states) < batch_size and attempts < max_attempts:
            idx = self._weighted_sample_index()
            window = self._read_window(idx, k_unroll + 1)
            attempts += 1
            if window is None:
                continue
            ws, wa, wr, wm, wsrc = window
            states.append(ws)
            actions.append(wa)
            rewards.append(wr)
            mc_returns_out.append(wm)
            sources.append(wsrc)

        if len(states) < batch_size:
            raise RuntimeError(
                f"Could not sample {batch_size} valid windows in {max_attempts} attempts. "
                f"Buffer may be too small or k_unroll too large."
            )
        return Batch(
            states=states,
            actions=actions,
            rewards=rewards,
            mc_returns=mc_returns_out,
            sources=sources,
        )

    # ---- Internals ----

    def _load_or_create_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text())
        meta: dict = {
            "version": "0.1",
            "schema_version": "phase_1a",
            "sources": {},
            "total_episodes": 0,
            "total_steps": 0,
            "eval_split_episodes": [],
            "gamma": 0.997,
        }
        self._save_meta(meta)
        return meta

    def _save_meta(self, meta: dict | None = None) -> None:
        m = meta if meta is not None else self._meta
        self._meta_path.write_text(json.dumps(m, indent=2))

    def _rebuild_index(self) -> None:
        """Scan all traj_*.parquet shards and populate the in-memory index."""
        for shard in sorted(self.root.glob("traj_*.parquet")):
            file_idx = len(self._index.file_paths)
            self._index.file_paths.append(shard)
            t = pq.read_table(shard, columns=["episode_id", "source", "priority"])
            ep_ids = t.column("episode_id").to_pylist()
            sources = t.column("source").to_pylist()
            priorities = t.column("priority").to_pylist()
            ep_length = len(ep_ids)
            for i, (eid, _src, prio) in enumerate(zip(ep_ids, sources, priorities)):
                self._index.rows.append(
                    _RowEntry(
                        file_idx=file_idx,
                        row_in_file=i,
                        episode_id=eid,
                        step_index_in_episode=i,
                        episode_length=ep_length,
                        priority=prio,
                    )
                )
            if ep_ids:
                self._index.seen_episodes.add((sources[0], ep_ids[0]))

    def _weighted_sample_index(self) -> int:
        """Draw a row index proportional to per-row priority weights.

        Normalizes once and caches; cache is invalidated by :meth:`add`.
        """
        if not self._index.rows:
            raise RuntimeError("ReplayBuffer is empty")
        if self._index._cached_weights is None:
            priorities = np.array([r.priority for r in self._index.rows], dtype=np.float64)
            total = priorities.sum()
            if total <= 0:
                raise RuntimeError(
                    "ReplayBuffer has total priority 0. All entries have priority=0; "
                    "cannot sample. Use uniform priority or assign non-zero weights."
                )
            self._index._cached_weights = priorities / total
        return int(np.random.choice(len(self._index.rows), p=self._index._cached_weights))

    def _read_window(
        self,
        idx: int,
        length: int,
    ) -> tuple[list[GameState], list[int], list[float], list[float], str] | None:
        """Read ``length`` consecutive rows starting at ``idx``.

        Returns ``None`` if the window would cross an episode boundary or exceed
        the buffer.
        """
        if idx + length > len(self._index.rows):
            return None
        # All rows must be in the same file and same episode
        start_entry = self._index.rows[idx]
        file_idx = start_entry.file_idx
        start_row_in_file = start_entry.row_in_file
        if start_entry.step_index_in_episode + length > start_entry.episode_length:
            return None
        # Verify every row in the window belongs to the same shard
        for offset in range(1, length):
            if self._index.rows[idx + offset].file_idx != file_idx:
                return None

        # Read the shard once for the whole window
        shard = self._index.file_paths[file_idx]
        table = pq.read_table(shard)
        states: list[GameState] = []
        actions: list[int] = []
        rewards: list[float] = []
        mc_returns: list[float] = []
        for offset in range(length):
            step_in_file = self._index.rows[idx + offset].row_in_file
            states.append(
                deserialize_state(table.column("state_bytes")[step_in_file].as_py())
            )
            if offset < length - 1:
                actions.append(int(table.column("action")[step_in_file].as_py()))
                rewards.append(float(table.column("reward")[step_in_file].as_py()))
                mc_returns.append(float(table.column("mc_return")[step_in_file].as_py()))
        source = table.column("source")[start_row_in_file].as_py()
        return states, actions, rewards, mc_returns, source
