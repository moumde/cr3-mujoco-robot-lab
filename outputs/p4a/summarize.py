"""Summarize paired P4a results and audit historic SAC ledger travel."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


def report(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['method']}/{row['profile']}"].append(row)
    result = {}
    for name, group in groups.items():
        summary = {key: sum(bool(r[key]) for r in group) for key in
                   ("orientation_correct", "grasped", "stable", "completed", "success",
                    "unexpected_collisions", "dropped", "observed_bilateral")}
        summary["episodes"] = len(group)
        summary["lift_100mm"] = sum(r["max_lift_m"] >= .1 for r in group)
        summary["hold_20_ticks"] = sum(r.get("learned_grasp", {}).get(
            "hold_steps", r["observed_carry_hold_ticks"]) >= 20 for r in group)
        summary["phase_completed"] = {phase: sum(successor in r["phases"] for r in group)
                                      for phase, successor in (("Reorient", "Transfer"),
                                                                ("Transfer", "Place"),
                                                                ("Place", "Release"))}
        for key, scale in (("rgb_xy_error_m", 1000), ("position_error_m", 1000),
                           ("angle_error_deg", 1), ("simulation_s", 1), ("wall_s", 1)):
            values = np.array([r[key] for r in group if r[key] is not None]) * scale
            summary[key.removesuffix("_m") + "_mm" if scale == 1000 else key] = dict(mean=float(values.mean()), max=float(values.max()))
        handoff = [r for r in group if "handoff" in r]
        if handoff:
            for key in ("tcp_speed_m_s", "ctrl_jump_rad"):
                summary[f"first_carry_max_{key}"] = max(r["motion_transitions"][0][key]
                                                       for r in handoff)
            summary["max_passive_grip_offset_error_mm"] = max(
                1000 * r["handoff"]["passive_grip_offset_error_m"] for r in handoff)
        result[name] = summary
    return result


def historical_audit():
    selected = {17208, 17211, 18093}
    path = Path("outputs/p3c-10.7-r1/run/step-750.jsonl.gz")
    phases = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    net = defaultdict(float)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            seed = row["scene_seed"]
            if seed not in selected or row["orientation"] != "flat_facing_camera":
                continue
            obs = row["obs"]
            phase = ("hold" if obs[33] > 0 else "lift" if obs[34] else
                     "bilateral" if obs[31] else "precontact")
            delta = 1000 * row["residual_budget"]["accepted_residual_target_delta_m"][1]
            phases[seed][phase][0 if delta >= 0 else 1] += abs(delta)
            net[seed] += delta
    return {seed: dict(net_y_mm=net[seed], direction_travel_mm_by_post_step_phase={
        phase: values.tolist() for phase, values in group.items()})
        for seed, group in phases.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = json.loads((args.output / "results.json").read_text())
    (args.output / "summary.json").write_text(json.dumps(report(results), indent=2) + "\n")
    (args.output / "sac-ledger-audit.json").write_text(json.dumps(historical_audit(), indent=2) + "\n")
