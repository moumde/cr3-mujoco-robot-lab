"""Check all exported action labels against original control intervals."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cr3_lab.demonstrations.chunks import control_rows, labels


def verify(raw, dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    for split in ("train", "validation", "test"):
        observed = []
        for path in sorted((dataset / split / "data").rglob("*.parquet")):
            observed.extend(pq.read_table(path, columns=["episode_index", "action",
                "observation.state", "phase_index", "timestamp"]).to_pylist())
        expected = []
        for episode in (e for e in manifest if e["split"] == split and e["success"]):
            folder = raw / episode["id"]
            control = control_rows(folder)
            for k, action, phase in labels(control, range(0, len(control), 5)):
                expected.append((episode["dataset_episode"], action.reshape(35), control[k]["state"],
                                 phase, control[k]["time_s"]))
        for actual, (index, action, state, phase, timestamp) in zip(observed, expected, strict=True):
            assert actual["episode_index"] == index and actual["phase_index"] == phase
            np.testing.assert_array_equal(actual["action"], action)
            np.testing.assert_array_equal(actual["observation.state"], np.asarray(state, np.float32))
            assert abs(actual["timestamp"] - timestamp) < 2e-6
        print(split, len(observed), "chunks verified")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    verify(args.raw, args.dataset)
