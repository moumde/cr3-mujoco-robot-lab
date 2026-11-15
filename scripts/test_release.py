"""Run one release test tier with explicit prerequisites."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from demo import ROOT, verify

PUBLIC = (
    "tests/test_release_demo.py", "tests/test_action_chunks.py", "tests/test_p2b1.py",
    "tests/test_p0.py", "tests/test_presentation.py",
)
HISTORICAL_INPUTS = (
    "outputs/p3c-7/results.json", "outputs/p3c-10.4/data/manifest.json",
    "outputs/p3c-10.7/run/online.jsonl",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("public-core", "optional-research", "historical-full"))
    parser.add_argument("--output", type=Path, help="New directory for the Optional Research rollouts")
    args = parser.parse_args()
    if args.tier == "optional-research" and not args.output:
        parser.error("optional-research requires --output")
    if args.tier == "public-core":
        verify({"quickstart"})
        paths = list(PUBLIC)
    else:
        verify({"quickstart", "research"})
        if args.tier == "historical-full":
            missing = [p for p in HISTORICAL_INPUTS if not (ROOT / p).is_file()]
            if missing:
                raise SystemExit("Restore private historical inputs:\n" + "\n".join(missing))
            paths = ["tests", *sorted(str(p.relative_to(ROOT)) for p in ROOT.glob("outputs/*/test_*.py"))]
        else:
            paths = ["tests/test_p4a.py", "tests/test_residual_budget.py",
                     "outputs/p5a/test_ppo_dist.py", "outputs/p5a-r3/test_final.py"]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], cwd=ROOT, env=env, check=True)
    if args.tier == "optional-research":
        subprocess.run([sys.executable, "scripts/demo.py", "research", str(args.output.resolve())],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
