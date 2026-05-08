"""Tests for the priority-weighted replay buffer with Parquet persistence."""
import json
from pathlib import Path

import pytest

from pokemon_planner.data.replay import ReplayBuffer
from pokemon_planner.data.trajectory import Trajectory, TrajectoryStep
from pokemon_planner.state import (
    BattleState,
    GameState,
)


def _state(map_id: int = 0) -> GameState:
    return GameState(
        map_id=map_id, x=0, y=0, party=(), bag=(), badges=0,
        event_flags=bytes(256), money=0, time_played_frames=0,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


def _traj(source: str, episode_id: int, length: int = 8) -> Trajectory:
    steps = tuple(
        TrajectoryStep(state=_state(map_id=i), action=i % 9,
                       reward=float(i), done=(i == length - 1), info={})
        for i in range(length)
    )
    return Trajectory(steps=steps, source=source, episode_id=episode_id, seed=0)


def test_buffer_init_empty(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    assert buf.size == 0
    assert (tmp_path / "meta.json").exists()


def test_buffer_add_persists_parquet(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    traj = _traj("v2_1310720", episode_id=1, length=10)
    buf.add(traj, priority=0.3)
    assert buf.size == 10  # 10 steps
    files = list(tmp_path.glob("traj_*.parquet"))
    assert len(files) == 1


def test_buffer_meta_records_sources(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    buf.add(_traj("v2_1310720", episode_id=0, length=4), priority=0.3)
    buf.add(_traj("v2_2621440", episode_id=0, length=4), priority=0.3)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert "v2_1310720" in meta["sources"]
    assert "v2_2621440" in meta["sources"]
    assert meta["total_steps"] == 8


def test_buffer_reload_from_disk(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    buf.add(_traj("v2_1310720", episode_id=0, length=5), priority=0.3)
    del buf
    buf2 = ReplayBuffer(root=tmp_path)
    assert buf2.size == 5


def test_sample_batch_returns_correct_shape(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    for i in range(5):
        buf.add(_traj("v2_1310720", episode_id=i, length=20), priority=0.3)
    batch = buf.sample_batch(batch_size=4, k_unroll=3)
    # batch should have B=4 trajectories of length k+1=4
    assert len(batch.states) == 4
    assert all(len(t) == 4 for t in batch.states)


def test_sample_batch_rejects_episode_boundary_crossing(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path)
    buf.add(_traj("v2_1310720", episode_id=0, length=3), priority=0.3)  # too short for k=5
    buf.add(_traj("v2_1310720", episode_id=1, length=20), priority=0.3)
    # k_unroll=5 needs windows of length 6; only the second episode supports it
    batch = buf.sample_batch(batch_size=8, k_unroll=5)
    assert len(batch.states) == 8  # should fill via resampling


def test_buffer_skip_already_extracted_episode(tmp_path: Path):
    """add() should be idempotent on (source, episode_id)."""
    buf = ReplayBuffer(root=tmp_path)
    buf.add(_traj("v2_1310720", episode_id=42, length=5), priority=0.3)
    buf.add(_traj("v2_1310720", episode_id=42, length=5), priority=0.3)  # same!
    assert buf.size == 5  # not 10
