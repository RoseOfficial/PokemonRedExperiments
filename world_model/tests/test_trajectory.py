"""Tests for trajectory data structures and serialization."""
import pytest

from pokemon_planner.data.trajectory import (
    TrajectoryStep,
    Trajectory,
    serialize_state,
    deserialize_state,
)
from pokemon_planner.state import (
    BattleState,
    BagSlot,
    GameState,
    PartySlot,
)


def _sample_state(map_id: int = 5) -> GameState:
    return GameState(
        map_id=map_id, x=10, y=12,
        party=(PartySlot(species_id=0xB0, level=12, hp_cur=20, hp_max=24,
                         status=0, moves=(0x21, 0, 0, 0)),),
        bag=(BagSlot(item_id=0x04, qty=5),),
        badges=0b0000_0001,
        event_flags=bytes(256),
        money=300, time_played_frames=42,
        battle=BattleState(in_battle=False),
        tile_collision=bytes(256), menu_flags=0,
    )


def test_trajectorystep_construct():
    step = TrajectoryStep(
        state=_sample_state(),
        action=4,
        reward=0.5,
        done=False,
        info={"source": "v2_1310720", "episode_id": 7, "step_index": 42},
    )
    assert step.action == 4
    assert step.reward == pytest.approx(0.5)
    assert step.info["source"] == "v2_1310720"


def test_state_msgpack_roundtrip():
    state = _sample_state(map_id=0x12)
    blob = serialize_state(state)
    assert isinstance(blob, bytes)
    assert 600 < len(blob) < 1500   # rough envelope per spec Section 3.5
    restored = deserialize_state(blob)
    assert restored == state


def test_trajectory_with_multiple_steps():
    steps = [
        TrajectoryStep(state=_sample_state(map_id=i), action=i % 9,
                       reward=float(i), done=(i == 4), info={})
        for i in range(5)
    ]
    traj = Trajectory(
        steps=tuple(steps),
        source="v2_1310720",
        episode_id=42,
        seed=12345,
    )
    assert len(traj.steps) == 5
    assert traj.steps[-1].done is True
    assert traj.length == 5


def test_trajectory_compute_mc_returns():
    """MC return: discounted sum of rewards from each step to episode end."""
    steps = (
        TrajectoryStep(state=_sample_state(), action=0, reward=1.0, done=False, info={}),
        TrajectoryStep(state=_sample_state(), action=0, reward=2.0, done=False, info={}),
        TrajectoryStep(state=_sample_state(), action=0, reward=3.0, done=True, info={}),
    )
    traj = Trajectory(steps=steps, source="x", episode_id=0, seed=0)
    mc = traj.mc_returns(gamma=0.5)
    # step[0]: 1 + 0.5*2 + 0.25*3 = 1 + 1 + 0.75 = 2.75
    # step[1]: 2 + 0.5*3 = 3.5
    # step[2]: 3
    assert mc[0] == pytest.approx(2.75)
    assert mc[1] == pytest.approx(3.5)
    assert mc[2] == pytest.approx(3.0)
