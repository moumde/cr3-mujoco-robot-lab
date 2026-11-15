"""Render per-seed validation curves for Grasp & Lift."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    metrics = {"bilateral": "Bilateral contact rate", "grasped": "Confirmed grasp rate",
               "held": "Stable lift success rate", "mean_reward": "Mean episode reward"}
    for method, color in (("ppo", "#24619b"), ("sac", "#b75628")):
        for seed, style in ((11, "-"), (22, "--")):
            path = args.directory / f"{method}-{seed}" / "validation.jsonl"
            rows = {r["steps"]: r for r in map(json.loads, path.read_text().splitlines())}
            for axis, (field, label) in zip(axes.flat, metrics.items()):
                axis.plot(rows.keys(), [r[field] / r["episodes"] if field != "mean_reward" else r[field]
                                        for r in rows.values()], style, color=color,
                          label=f"{method.upper()} {seed}")
                axis.set(xlabel="Environment steps", ylabel=label)
                axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(args.directory / "validation-curves.png", dpi=150)


if __name__ == "__main__":
    main()
