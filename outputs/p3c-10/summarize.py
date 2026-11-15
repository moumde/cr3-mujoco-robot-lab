"""Compact development checkpoint comparison, with aggregate motion percentiles."""
from collections import Counter
import gzip
import json
from pathlib import Path

from smoke import distributions

root = Path(__file__).resolve().parent
result = {}
for seed in (11, 22):
    run = root / f"seed-{seed}"
    checkpoints = {}
    for step in range(0, 5001, 1000):
        report = json.loads((run / f"eval-{step}.json").read_text())
        pairs = report["pairs"]
        checkpoints[step] = {"summary": report["summary"]}
        for method in ("bc", "deterministic", "stochastic"):
            rows = [p[method] for p in pairs]
            ik = Counter()
            for row in rows:
                ik.update(row["distribution"]["ik"])
            checkpoints[step][method] = {
                "ik": dict(ik), "collision": dict(Counter(r["collision"] for r in rows if r["collision"])),
                "lost_grip": sum(r["reason"] == "lost_grip" for r in rows),
                "mean_min_grasp_mm": 1000 * sum(r["min_grasp_distance_m"] for r in rows) / len(rows),
                "closure_half_steps": [r["closure_half_step"] for r in rows],
            }
    with gzip.open(run / "online.jsonl.gz", "rt") as stream:
        rows = [json.loads(line)["residual_control"] for line in stream]
    result[seed] = {"checkpoints": checkpoints, "warmup": distributions(rows[:500]),
                    "online": distributions(rows), "steps": len(rows)}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
