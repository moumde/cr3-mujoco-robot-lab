"""Plot validation learning curves for the three-seed P3b comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    fields = [("success_rate", "Success rate"), ("distance_mm", "Final distance (mm)"),
              ("mean_reward", "Episode reward")]
    for method, color in (("ppo", "#2368a2"), ("sac", "#b65327")):
        runs = []
        for folder in sorted(args.directory.glob(f"formal-{method}-*")):
            if not folder.is_dir():
                continue
            records = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
            runs.append({row["steps"]: row for row in records})
        if not runs:
            raise ValueError(f"No formal runs for {method}")
        steps = sorted(set.intersection(*(set(run) for run in runs)))
        for axis, (key, label) in zip(axes, fields):
            values = np.asarray([[run[step][key] for step in steps] for run in runs])
            axis.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=0.18)
            axis.plot(steps, values.mean(axis=0), color=color, label=method.upper())
            axis.set(xlabel="Training environment steps", ylabel=label)
            axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Reach validation | mean and range across training seeds")
    fig.savefig(args.directory / "learning-curves.png", dpi=160)


if __name__ == "__main__":
    main()
