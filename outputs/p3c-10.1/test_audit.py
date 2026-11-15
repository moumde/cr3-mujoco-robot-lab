import numpy as np

from audit import manifest, stage


def test_manifest_is_paired_and_held_out():
    cases = manifest()
    assert len({tuple(row["spawn_xy_m"]) for row in cases}) == 20
    assert {row["seed"] for row in cases}.isdisjoint(range(17700, 17740))


def test_stage_partition():
    obs = np.zeros(40)
    assert stage(obs) == "before_bilateral"
    obs[31] = 1
    assert stage(obs) == "bilateral"
    obs[34] = 1
    assert stage(obs) == "confirmed"
    obs[27] = 0.10
    assert stage(obs) == "100mm"
    obs[33] = .1
    assert stage(obs) == "holding"
