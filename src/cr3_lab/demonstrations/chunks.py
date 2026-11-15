"""Construct five contiguous actuator targets starting at each camera observation."""
import json

import numpy as np

PHASES = ("Initialize", "Approach", "Pre-Grasp", "Grasp", "Lift", "Reorient",
          "Transfer", "Place", "Release", "Retreat")
CHUNK = 5
FPS = 10


def labels(control, indices):
    for k in indices:
        segment = control[k:k+CHUNK]
        if len(segment) != CHUNK:
            continue
        for i, row in enumerate(segment):
            np.testing.assert_allclose(
                [row["time_s"], row["observed_time_s"], row["action_end_s"]],
                [(k+i)/50, (k+i)/50, (k+i+1)/50], atol=1e-12, rtol=0)
        yield k, np.asarray([r["action"] for r in segment], np.float32), PHASES.index(
            segment[0].get("phase", "Initialize"))


def control_rows(folder):
    return [json.loads(line) for line in (folder / "control.jsonl").read_text().splitlines()]
