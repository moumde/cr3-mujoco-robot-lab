"""Summarize R1 gates and compare baseline methods on the same manifest."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from r1 import ROOT, assess, load_policy, panels
from experiment import ZERO
from validate import feedback
from conservative_sac import rollout
import gzip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stats(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95))) if len(values) else None


def paired(rows, base):
    success = [(a, b) for a, b in zip(rows, base, strict=True) if a["is_success"] and b["is_success"]]
    delta = [a["steps"] - b["steps"] for a, b in success]
    return dict(
        success=sum(r["is_success"] for r in rows), count=len(rows),
        steps=stats([a["steps"] for a, _ in success]), delta_steps=stats(delta),
        faster=sum(d < 0 for d in delta), slower=sum(d > 0 for d in delta),
        lift100=stats([a["first_100mm_step"] for a, _ in success]),
        delta_lift100=stats([a["first_100mm_step"] - b["first_100mm_step"] for a, b in success]),
        ik=sum(sum(r["ik_rejections"].values()) for r in rows),
        contacts=sum(bool(r["collision"]) for r in rows),
        absolute_mm_xyz=[stats([1000*r["absolute_m"][i] for r in rows]) for i in range(3)])


def analyze(out):
    torch.set_num_threads(1)
    base = json.loads((out / "residual-bc.json").read_text())
    status = json.loads((out / "status.json").read_text())
    online = [json.loads(line) for line in (out / "online.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (out / "updates.jsonl").read_text().splitlines()]
    groups = dict(history=list(range(42)), validation=list(range(42, 62)))
    for split, ids in list(groups.items()):
        for direction in ("protrusion", "flat"):
            groups[f"{split}_{direction}"] = [i for i in ids if base[i]["orientation"].startswith(direction)]
    panels_by_strategy = {"residual-bc": base}
    for ckpt in status["checkpoints"]:
        step = ckpt["step"]
        panels_by_strategy[f"step-{step}"] = json.loads((out / f"step-{step}.json").read_text())
    history = json.loads((ROOT.parent / "p3c-10.5/uniform/evaluation.json").read_text())["rows"]
    bc = load_policy()
    extra = {}
    for method in ("frozen", "teacher"):
        with gzip.open(out / f"validation-{method}.jsonl.gz", "xt") as stream:
            extra[method] = []
            for index, case in enumerate(panels()[42:], 42):
                obs, result = rollout(case, bc, method, lambda _: ZERO, stream)
                np.testing.assert_array_equal(obs, np.asarray(base[index]["initial"], dtype=np.float32))
                extra[method].append(result)
        panels_by_strategy[method] = [row[method] for row in history] + extra[method]
    (out / "validation-comparators.json").write_text(json.dumps(extra, indent=2) + "\n")
    summary = {method: {group: paired([rows[i] for i in ids], [base[i] for i in ids])
                        for group, ids in groups.items()}
               for method, rows in panels_by_strategy.items()}
    diagnostics = {}
    for name, entries in (("warmup", online[:500]), ("post_warmup", online[500:])):
        if entries:
            diagnostics[name] = dict(steps=len(entries),
                max_abs_drift=np.abs([r["policy_delta"] for r in entries]).max(axis=0).tolist(),
                projected_drift_p95_mm=np.quantile(1000*np.abs([r["target_delta_m"] for r in entries]),
                                                     .95, axis=0).tolist(),
                sampled_saturation=(np.abs([r["stochastic"] for r in entries]) >= .95).mean(axis=0).tolist(),
                deterministic_saturation=(np.abs([r["deterministic"] for r in entries]) >= .95).mean(axis=0).tolist(),
                max_absolute_mm=(1000*np.max([r["residual_budget"]["absolute_budget_m"] for r in entries],
                                             axis=0)).tolist())
    (out / "summary.json").write_text(json.dumps(dict(paired=summary, online=diagnostics,
        online_episodes=[{k: r[k] for k in ("step", "scene_seed", "orientation", "reason")}
                         for r in online if r["terminated"] or r["truncated"]],
        comparator_steps=sum(r["steps"] for rows in extra.values() for r in rows)), indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for ax, metric in zip(axes.flat, ("actor", "critic", "alpha", "anchor"), strict=True):
        ax.plot([r["step"] for r in updates], [r[metric] for r in updates])
        ax.set(xlabel="Online step", ylabel=metric)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=140)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
