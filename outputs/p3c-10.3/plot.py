"""Plot budget use for each diagnostic from the recorded JSONL."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    cases = json.loads((output / "results.json").read_text())["probes"]
    paths = [output / f"{r['scene_seed']}-{r['orientation']}-{r['mode']}.jsonl.gz" for r in cases]
    fig, axes = plt.subplots(len(paths), 2, figsize=(11, 1.6 * len(paths)), squeeze=False)
    for path, (signed, absolute) in zip(paths, axes, strict=True):
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for axis, key in ((signed, "signed_budget_m"), (absolute, "absolute_budget_m")):
            values = 1000 * np.array([r["residual_budget"][key] for r in rows])
            axis.plot(np.arange(1, len(rows) + 1), values, label=["X", "Y", "Z"])
            axis.grid(alpha=.25)
            axis.set_ylabel("mm")
        signed.set_title(path.name.removesuffix(".jsonl.gz"), fontsize=8)
        absolute.set_title(rows[-1]["reason"], fontsize=8)
    axes[0, 0].legend(loc="upper right", ncols=3)
    axes[-1, 0].set_xlabel("Step · signed cumulative")
    axes[-1, 1].set_xlabel("Step · absolute cumulative")
    fig.tight_layout()
    fig.savefig(output / "budget.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
