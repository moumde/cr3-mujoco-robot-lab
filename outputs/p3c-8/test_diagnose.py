import gzip
import json

import numpy as np
import pytest
import torch

import diagnose


@pytest.mark.parametrize("seed, orientation, rejected", [(17208, diagnose.Orientation.PROTRUSION, 256),
                                                         (17211, diagnose.Orientation.FLAT, 0)])
def test_instrumentation_reproduces_expected_result(tmp_path, monkeypatch, seed, orientation, rejected):
    monkeypatch.setattr(diagnose, "ROOT", tmp_path)
    torch.set_num_threads(1)
    env = diagnose.ChallengeEnv()
    env.xy = diagnose.sample_xy(0.035, seed)
    cp = torch.load(diagnose.CHECKPOINT, weights_only=True)
    center, spread = (cp[k].numpy() for k in ("center", "spread"))
    _, info = diagnose.record(env, seed, orientation, "bc", diagnose.load_policy(),
                              center, spread, np.full(40, -np.inf), np.full(40, np.inf))
    with gzip.open(next(tmp_path.glob("*.gz")), "rt") as stream:
        rows = list(map(json.loads, stream))
    assert sum(r["ik"]["status"] != "accepted" for r in rows) == rejected
    assert rows[-1]["info"] == info
    assert all(r["applied_tcp_delta_m"] == [0, 0, 0] for r in rows if r["info"]["ik_rejected"])
    env.close()
