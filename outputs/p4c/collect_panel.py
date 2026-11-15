import argparse
from pathlib import Path
from cr3_lab.demonstrations.collect import collect
from cr3_lab.tasks.motor_scene import SpawnRange

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--from-seed", type=int, default=5500)
parser.add_argument("--through-seed", type=int, default=5518)
args = parser.parse_args()
# 18 distinct random positions, with both orientations grouped by seed.
configs = [(SpawnRange.FIXED, 5500, False, "rgb_expert", "test")]
configs += [(SpawnRange.LARGE, seed, False, "rgb_expert",
             "train" if seed <= 5512 else "validation" if seed <= 5515 else "test")
            for seed in range(5501, 5519)]
collect(args.output, Path("outputs"), configs=[c for c in configs if args.from_seed <= c[1] <= args.through_seed], check_parity=False)
