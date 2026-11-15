"""Report paired milestone/efficiency changes and training curves."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def paired(rows, previous):
    both = [(r, p) for r, p in zip(rows, previous, strict=True) if r["is_success"] and p["is_success"]]
    delta = np.array([r["steps"] - p["steps"] for r, p in both])
    lift_delta = [r["first_100mm_step"] - p["first_100mm_step"] for r, p in both]
    ik = [sum(r["ik_rejections"].values()) - sum(p["ik_rejections"].values())
          for r, p in zip(rows, previous, strict=True)]
    return dict(n=len(rows), success=sum(r["is_success"] for r in rows),
                previous_success=sum(r["is_success"] for r in previous),
                bilateral=sum(r["first_bilateral_step"] is not None for r in rows),
                confirmed=sum(r["first_lift_step"] is not None for r in rows),
                lift100=sum(r["first_100mm_step"] is not None for r in rows),
                paired_success=len(both), steps_delta_mean=float(delta.mean()) if len(delta) else None,
                lift_step_delta_mean=float(np.mean(lift_delta)) if lift_delta else None,
                faster=int((delta < 0).sum()), slower=int((delta > 0).sum()), tied=int((delta == 0).sum()),
                ik_delta=sum(ik), cases_with_more_ik=sum(i > 0 for i in ik),
                contacts=sum(bool(r["collision"]) for r in rows),
                max_absolute_mm=(1000 * np.max([r["absolute_m"] for r in rows], axis=0)).tolist(),
                max_signed_mm=(1000 * np.max([r["peak_signed_m"] for r in rows], axis=0)).tolist(),
                executed_saturation=(np.sum([r["saturated_xyz"] for r in rows], axis=0) /
                                     sum(r["steps"] for r in rows)).tolist())


def report(root):
    baseline = json.loads((root / "residual-bc.json").read_text())
    status = json.loads((root / "status.json").read_text())
    results = {}
    for check in status["checkpoints"]:
        step = check["step"]
        path = root / f"step-{step}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        groups = {}
        for group, selection in (
                ("history", lambda r: r["profile"] != "validation"),
                ("nominal", lambda r: r["profile"] == "fixed"),
                ("validation", lambda r: r["profile"] == "validation"),
                ("history_protrusion", lambda r: r["profile"] != "validation" and
                 r["orientation"] == "protrusion_facing_camera"),
                ("history_flat", lambda r: r["profile"] != "validation" and
                 r["orientation"] == "flat_facing_camera"),
                ("validation_protrusion", lambda r: r["profile"] == "validation" and
                 r["orientation"] == "protrusion_facing_camera"),
                ("validation_flat", lambda r: r["profile"] == "validation" and
                 r["orientation"] == "flat_facing_camera")):
            ids = [i for i, r in enumerate(rows) if selection(r)]
            groups[group] = paired([rows[i] for i in ids], [baseline[i] for i in ids])
        results[step] = groups
    updates = [json.loads(line) for line in (root / "updates.jsonl").read_text().splitlines()]
    online = [json.loads(line) for line in (root / "online.jsonl").read_text().splitlines()]
    actor_abs = np.abs([r["actor_residual"] for r in online])
    online_stats = dict(
        scenarios=sorted({r["scene_seed"] for r in online}),
        actual_actor_abs_quantiles=np.quantile(actor_abs, [.5, .95, 1], axis=0).tolist(),
        actual_actor_saturation=(actor_abs >= .95).mean(axis=0).tolist(),
        completed=[{k: r[k] for k in ("step", "scene_seed", "orientation", "reason")}
                   for r in online if r["reason"]],
    )
    (root / "summary.json").write_text(json.dumps(dict(status=status, paired=results,
                                                      online=online_stats), indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for ax, metric in zip(axes.flat, ("actor", "critic", "alpha", "anchor"), strict=True):
        ax.plot([r["step"] for r in updates], [r[metric] for r in updates])
        ax.set(xlabel="Online step", ylabel=metric)
    fig.tight_layout()
    fig.savefig(root / "training.png", dpi=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
