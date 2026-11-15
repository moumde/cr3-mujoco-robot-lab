"""Audit correction authority on frozen failure traces and initial exploration."""
import gzip
import json
from pathlib import Path

import numpy as np
from cr3_lab.rl.reach import TCP_DELTA_M

root = Path(__file__).resolve().parent
report = {}
for path in sorted((root.parent / "p3c-8").glob("*-bc.jsonl.gz")):
    with gzip.open(path, "rt") as stream:
        rows = list(map(json.loads, stream))
    action = np.asarray([r["raw_action"][:3] for r in rows])
    near = np.array([r["grasp_distance_m"] <= .065 for r in rows])
    down = action[:, 2] < 0
    mask = near & down
    tcp = np.asarray([r["tcp_world_m"] for r in rows])
    report[path.stem] = {
        "xyz_min": action.min(0).tolist(), "xyz_max": action.max(0).tolist(),
        "measured_tcp_delta_mm_abs_max": (1000 * abs(np.diff(tcp, axis=0)).max(0)).tolist(),
        "final_grasp_distance_mm": 1000 * rows[-1]["grasp_distance_m"],
        "near_descending_steps": int(mask.sum()),
        "can_cancel_at_scale_half": int((mask & (abs(action[:, 2]) <= .5)).sum()),
        "first_rejected": next(({"step": r["step"], "xyz": r["raw_action"][:3], **r["ik"]}
                                 for r in rows if r["ik"]["status"] != "accepted"), None),
    }
rng = np.random.default_rng(11)
for bias in (-3, -2):
    a = abs(np.tanh(rng.normal(0, np.exp(bias), (100000, 3))))
    p = np.percentile(a, [50, 95, 100], axis=0)
    report[f"log_std_{bias}"] = {"abs_residual": p.tolist(),
                                "nominal_correction_mm": (p * .5 * TCP_DELTA_M * 1000).tolist()}
(root / "calibration.json").write_text(json.dumps(report, indent=2))
