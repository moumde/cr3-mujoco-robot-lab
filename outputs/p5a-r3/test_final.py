from pathlib import Path
import numpy as np

from final_test import ROOT, read, digest, sample_xy
from report_final import summarize


def test_frozen_protocol_and_panel():
    manifest = read(ROOT / "test_manifest.json")
    cases = manifest["cases"]
    assert len(cases) == 80 and len({tuple(c["xy"]) for c in cases}) == 40
    assert sorted({c["seed"] for c in cases}) == list(range(17700,17740))
    for c in cases:
        np.testing.assert_array_equal(c["xy"], sample_xy(.035,c["seed"]))
    frozen = read(ROOT / "frozen_candidates.json")
    assert len(frozen["candidates"]) == 7
    for c in frozen["candidates"]:
        assert digest(Path(c["path"])) == c["sha256"]


def test_paired_efficiency_excludes_failures():
    row = dict(steps=330,success=True,bilateral=1,grasp=2,lift=3,hold=20,collision=None,
               reason="success",final_stage="hold",review_exceeded=False,ik={},absolute_mm=[1,2,3],
               drift_p95=[0,0,0],phase_steps={},tcp_mm=200)
    fast = dict(row, steps=320)
    failure = dict(row,steps=150,success=False,reason="unexpected_contact",collision="pad/table")
    stats = summarize([fast,failure],[row,row])
    assert stats["paired_mean"] == -10 and stats["all_mean"] == 235
    assert not stats["eligible"] and stats["lost_baseline"] == 1
