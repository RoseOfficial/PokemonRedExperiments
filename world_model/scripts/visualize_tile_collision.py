"""One-time diagnostic -- render extracted tile-collision matrix alongside PyBoy screen.

Usage:
    python scripts/visualize_tile_collision.py

Loads init.state, extracts collision, prints both the screen tilemap (left) and
the collision matrix (right). Walls in the screen should correspond to '#' codes
in the matrix. Run a few times with different save states or after stepping a
few actions to spot-check the extraction quality.
"""
from pathlib import Path

from pokemon_planner._tilesets import code_to_name
from pokemon_planner.env import PokeBoy, read_state


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    rom = repo_root / "PokemonRed.gb"
    save = repo_root / "init.state"

    pb = PokeBoy(rom_path=str(rom), save_state_path=str(save))
    try:
        state = read_state(pb.pyboy)
        coll = state.tile_collision

        print(f"map_id={state.map_id:#x} player=({state.x},{state.y})\n")
        print("Tile collision (16x16 around player; player at center [8,8]):")
        for row in range(16):
            line = ""
            for col in range(16):
                code = coll[row * 16 + col]
                if row == 8 and col == 8:
                    line += "P "
                elif code == 0:
                    line += ". "       # walkable
                elif code == 1:
                    line += "# "       # blocked
                elif code in (2, 3, 4):
                    line += "L "       # ledge
                elif code == 5:
                    line += "W "       # warp
                elif code == 255:
                    line += "? "
                else:
                    line += f"{code} "
            print("  " + line)
        print("\nLegend: P=player .=walk #=block L=ledge W=warp ?=unknown")

        nonzero = sum(1 for c in coll if c != 0)
        code_counts: dict[int, int] = {}
        for c in coll:
            code_counts[c] = code_counts.get(c, 0) + 1
        print(f"\nNon-zero cells: {nonzero}/256")
        print("Code breakdown:", {code_to_name(k): v for k, v in sorted(code_counts.items())})
    finally:
        pb.close()


if __name__ == "__main__":
    main()
