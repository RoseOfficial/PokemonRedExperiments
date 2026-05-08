"""Integration test for the bootstrap extraction script.

Runs a tiny extraction (1 episode, ~50 steps) to verify wiring end-to-end.
Skipped if any v2 checkpoint is missing.
"""
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_extract_one_episode(tmp_path: Path, repo_root: Path):
    """Run extraction on one v2 checkpoint for ~50 steps; verify Parquet shard exists."""
    ckpt = repo_root / "v2" / "runs" / "poke_1310720_steps.zip"
    if not ckpt.exists():
        pytest.skip(f"v2 checkpoint not found: {ckpt}")
    rom = repo_root / "PokemonRed.gb"
    if not rom.exists():
        pytest.skip(f"ROM not found: {rom}")

    from pokemon_planner.data.replay import ReplayBuffer
    from scripts.bootstrap_demos import extract_one_episode

    buf = ReplayBuffer(root=tmp_path)
    ep = extract_one_episode(
        ckpt_path=ckpt,
        rom_path=rom,
        init_state_path=repo_root / "init.state",
        episode_id=0,
        seed=12345,
        max_steps=50,
        buffer=buf,
    )

    assert ep.length > 0, "Extraction produced no steps"
    assert buf.size == ep.length
    parquets = list(tmp_path.glob("traj_*.parquet"))
    assert len(parquets) == 1


@pytest.mark.integration
def test_extract_idempotent(tmp_path: Path, repo_root: Path):
    """Re-running extraction with same (source, episode_id) is a no-op."""
    ckpt = repo_root / "v2" / "runs" / "poke_1310720_steps.zip"
    if not ckpt.exists():
        pytest.skip(f"v2 checkpoint not found: {ckpt}")
    rom = repo_root / "PokemonRed.gb"
    if not rom.exists():
        pytest.skip(f"ROM not found: {rom}")

    from pokemon_planner.data.replay import ReplayBuffer
    from scripts.bootstrap_demos import extract_one_episode

    buf = ReplayBuffer(root=tmp_path)
    extract_one_episode(
        ckpt_path=ckpt, rom_path=rom,
        init_state_path=repo_root / "init.state",
        episode_id=0, seed=12345, max_steps=20, buffer=buf,
    )
    size_after_first = buf.size

    # Same episode_id — should be skipped
    extract_one_episode(
        ckpt_path=ckpt, rom_path=rom,
        init_state_path=repo_root / "init.state",
        episode_id=0, seed=12345, max_steps=20, buffer=buf,
    )
    assert buf.size == size_after_first  # not double
