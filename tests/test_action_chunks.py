import numpy as np
import pytest

from cr3_lab.demonstrations.chunks import labels


def control(count):
    return [{"time_s": i/50, "observed_time_s": i/50, "action_end_s": (i+1)/50,
             "action": [i] * 7, "phase": "Lift"} for i in range(count)]


def test_chunk_starts_at_observation_and_drops_partial_tail():
    frames = list(labels(control(12), [0, 5, 10]))
    assert [entry[0] for entry in frames] == [0, 5]
    np.testing.assert_array_equal(frames[1][1][:, 0], np.arange(5, 10))
    assert frames[1][1].shape == (5, 7)


def test_chunk_rejects_misaligned_observation():
    records = control(5)
    records[2]["observed_time_s"] = 1
    with pytest.raises(AssertionError):
        list(labels(records, [0]))
