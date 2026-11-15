"""Plot paired development results and online loss for both smoke seeds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
for seed, color in ((11, "#245e93"), (22, "#ac4c2b")):
    run = root / f"seed-{seed}"
    steps = list(range(0, 5001, 1000))
    rows = [json.loads((run / f"eval-{step}.json").read_text())["summary"] for step in steps]
    for key, axis in (("failed", axes[0, 0]), ("nominal", axes[0, 1])):
        for method, style in (("bc", ":"), ("deterministic", "-"), ("stochastic", "--")):
            axis.plot(steps, [r[key][method]["held"] / r[key][method]["episodes"] for r in rows],
                      style, color=color, label=f"{seed} {method}")
        axis.set(xlabel="Online steps", ylabel=f"{key.capitalize()} success rate", ylim=(-.05, 1.05))
    metrics = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
    for key, axis in (("actor_loss", axes[1, 0]), ("critic_loss", axes[1, 1])):
        axis.plot([r["steps"] for r in metrics], [r["metrics"].get(f"train/{key}") for r in metrics],
                  color=color, label=str(seed))
        axis.set(xlabel="Online steps", ylabel=key.replace("_", " ").capitalize())
for axis in axes.flat:
    axis.legend(fontsize=7)
    axis.grid(alpha=.2)
fig.savefig(root / "curves.png", dpi=160)
