"""Extract paired earliest deviations and time-aligned regression milestones."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def read(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def details(row):
    trace = row["residual_budget"]
    return dict(step=row["steps"], phase=row["phase"], next_phase=row["next_phase"],
                grasp_error_mm=(1000 * np.linalg.norm(row["next_obs"][12:15])),
                object_speed_m_s=np.linalg.norm(row["next_obs"][21:24]),
                angular_speed_rad_s=np.linalg.norm(row["next_obs"][24:27]),
                height_mm=1000 * row["next_obs"][27], hold_steps=row["hold_steps"],
                signed_mm=(1000 * np.asarray(trace["signed_budget_m"])).tolist(),
                absolute_mm=(1000 * np.asarray(trace["absolute_budget_m"])).tolist(),
                teacher=row["teacher"], student=row["student"],
                ik=trace["ik_status"], collision=row["collision"])


def analyze(root):
    results = {}
    for seed, student in ((17207, "uniform"), (17218, "phase_balanced"), (17201, "phase_balanced")):
        rows = read(root / f"{seed}-{student}.jsonl.gz")
        teacher = read(root / f"{seed}-teacher.jsonl.gz")
        first = {}
        for a, b in zip(rows, teacher):
            trace_a, trace_b = (r["residual_budget"] for r in (a, b))
            distance = 1000 * np.linalg.norm(np.subtract(trace_a["tcp_world_m"], trace_b["tcp_world_m"]))
            comparisons = {
                "exact_action_difference": a["student"] != a["teacher"],
                "action_difference_ge_005": np.max(abs(np.subtract(a["student"], a["teacher"]))) >= .05,
                "tcp_distance_ge_1mm": distance >= 1,
                "object_distance_ge_1mm": np.linalg.norm(np.subtract(a["next_obs"][15:18],
                                                                     b["next_obs"][15:18])) >= .001,
                "contact_difference": a["next_obs"][31] != b["next_obs"][31],
                "hold_difference": a["hold_steps"] != b["hold_steps"],
            }
            for event, occurred in comparisons.items():
                if occurred and event not in first:
                    first[event] = dict(student=details(a), teacher=details(b), tcp_distance_mm=distance)
        methods = {}
        for mode in ("frozen", "teacher", "uniform", "phase_balanced"):
            trajectory = read(root / f"{seed}-{mode}.jsonl.gz")
            end = trajectory[-1]
            methods[mode] = dict(end=details(end), reason=end["reason"],
                                 bilateral=end["first_bilateral_step"], confirmation=end["first_lift_step"],
                                 lift_100mm=end["first_100mm_step"],
                                 max_hold=max(r["hold_steps"] for r in trajectory),
                                 first_ik_reject=next((details(r) for r in trajectory
                                                       if r["ik_rejected"]), None))
        results[seed] = dict(first=first, methods=methods)
    (root / "diagnostic-summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
