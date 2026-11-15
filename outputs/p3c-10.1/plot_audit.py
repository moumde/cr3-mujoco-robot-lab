"""Check pairing and plot deterministic milestone rates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
base = json.loads((root / "eval-0-0.json").read_text())
paired = {}
fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
for seed in (11, 22):
    steps = list(range(0, 5001, 1000))
    reports = [json.loads((root / f"eval-{seed}-{step}.json").read_text()) for step in steps]
    paired[seed] = {}
    for step, report in zip(steps, reports):
        np.testing.assert_array_equal([r["initial_obs"] for r in report["rows"]],
                                      [r["initial_obs"] for r in base["rows"]])
        assert [(r["seed"], r["orientation"], r["xy"]) for r in report["rows"]] == [
            (r["seed"], r["orientation"], r["xy"]) for r in base["rows"]]
        new = [r for b, r in zip(base["rows"], report["rows"]) if not b["is_success"] and r["is_success"]]
        lost = [r for b, r in zip(base["rows"], report["rows"]) if b["is_success"] and not r["is_success"]]
        paired[seed][step] = {"recovered": [[r["seed"], r["orientation"]] for r in new],
                              "lost": [[r["seed"], r["orientation"]] for r in lost]}
        if step == 0:
            assert report["summary"] == base["summary"]
    for axis, field in zip(axes.flat, ("bilateral", "grasped", "lifted", "held")):
        axis.plot(steps, [r["summary"]["all35"][field] / 40 for r in reports], "o-", label=f"SAC {seed}")
        axis.axhline(base["summary"]["all35"][field] / 40, linestyle=":", color="gray", label="BC" if seed == 11 else None)
        axis.set(xlabel="Historical checkpoint step", ylabel=field, ylim=(-.05, 1.05))
        axis.legend()
fig.suptitle("P3c-7 ±35 mm development manifest | deterministic")
fig.savefig(root / "milestones.png", dpi=160)

(root / "paired-changes.json").write_text(json.dumps(paired, indent=2) + "\n")
