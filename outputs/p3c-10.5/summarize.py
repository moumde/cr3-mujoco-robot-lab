"""Summarize paired milestones and create the offline loss chart."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(rows):
    steps = sum(r["steps"] for r in rows)
    requested = np.sum([r["requested_xyz"] for r in rows], axis=0)
    effective = np.sum([r["effective_xyz"] for r in rows], axis=0)
    return dict(
        bilateral=sum(r["first_bilateral_step"] is not None for r in rows),
        grasp=sum(r["first_lift_step"] is not None for r in rows),
        lift=sum(r["first_100mm_step"] is not None for r in rows),
        success=sum(r["is_success"] for r in rows), contact=sum(bool(r["collision"]) for r in rows),
        ik=sum(sum(r["ik_rejections"].values()) for r in rows),
        mean_steps=steps / len(rows), median_absolute_mm=(
            1000 * np.median([r["absolute_m"] for r in rows], axis=0)).tolist(),
        max_absolute_mm=(1000 * np.max([r["absolute_m"] for r in rows], axis=0)).tolist(),
        saturation=(np.sum([r["saturated_xyz"] for r in rows], axis=0) / steps).tolist(),
        effective_per_requested=np.divide(effective, requested, where=requested > 0,
                                          out=np.zeros(3)).tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    root = parser.parse_args().root
    fig, ax = plt.subplots(figsize=(7, 4))
    report = {}
    for mode in ("uniform", "phase_balanced"):
        evaluation = json.loads((root / mode / "evaluation.json").read_text())
        training = json.loads((root / mode / "training.json").read_text())
        ax.semilogy([r["epoch"] for r in training["history"]],
                    [r["validation"]["loss"] for r in training["history"]], label=mode)
        rows = evaluation["rows"]
        report[mode] = dict(gate=evaluation["gate"], best=training["best"],
                            train_time_s=training["wall_time_s"], eval_time_s=evaluation["wall_time_s"],
                            **{method: summarize([r[method] for r in rows])
                               for method in ("frozen", "teacher", "residual")})
    ax.set(xlabel="Epoch", ylabel="Validation MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "validation-loss.png", dpi=140)
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
