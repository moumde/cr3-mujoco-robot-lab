from collections import Counter
import gzip
import json

import numpy as np
import pytest

from dataset import digest, feedback, phase, plan, read_episode


def test_split_and_orientation_balance():
    cases = list(plan())
    assert [seed for seed, _, _ in cases] == list(range(18000, 18100))
    counts = Counter((split, orientation) for _, orientation, split in cases)
    assert sorted(counts.values()) == [10, 10, 40, 40]


def test_feedback_uses_only_observable_fields():
    obs = np.zeros(43, dtype=np.float32)
    obs[12:15] = [.002, -.003, -.02]
    base = np.array([.13, -.27, .35, -1], dtype=np.float32)
    expected = feedback(obs, base)
    changed = np.full(43, .77, dtype=np.float32)
    changed[12:15] = obs[12:15]
    changed[[27, 34]] = obs[[27, 34]]
    np.testing.assert_array_equal(feedback(changed, base), expected)
    assert expected.shape == (3,)
    assert np.max(np.abs(expected)) <= 1
    assert phase(obs) == "near"


def test_reader_verifies_hash(tmp_path):
    path = tmp_path / "episode.jsonl.gz"
    row = {"teacher_residual_raw_3": [0, .5, -1]}
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps(row) + "\n")
    assert read_episode(tmp_path, {"file": path.name, "sha256": digest(path)}) == [row]
    with pytest.raises(ValueError, match="Checksum mismatch"):
        read_episode(tmp_path, {"file": path.name, "sha256": "incorrect"})
