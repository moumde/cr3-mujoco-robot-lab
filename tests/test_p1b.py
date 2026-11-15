from dataclasses import replace

import mujoco
import numpy as np
import pytest

from cr3_lab.tasks.motor_expert import MotorExpert
from cr3_lab.tasks.motor_scene import (
    DEFAULT_SCENE, Orientation, SpawnRange, load_scene, oracle_pose, reset_scene,
    sample_initial_xy,
)


@pytest.mark.parametrize("spawn_range", list(SpawnRange))
def test_spawn_reproducible_and_fits_workcell(spawn_range):
    scene = DEFAULT_SCENE
    seeds = range(17, 37)
    samples = np.array([sample_initial_xy(scene, seed, spawn_range) for seed in seeds])
    np.testing.assert_array_equal(
        samples, [sample_initial_xy(scene, seed, spawn_range) for seed in seeds])
    assert len(np.unique(samples, axis=0)) == (1 if spawn_range is SpawnRange.FIXED else len(seeds))
    assert np.all(abs(samples - scene.initial_xy_m) <= spawn_range.half_width_m)
    assert np.all(samples[:, 1] > scene.motor.length_m / 2 + scene.motor.boss_length_m)
    with pytest.raises(ValueError, match="left half"):
        sample_initial_xy(replace(scene, initial_xy_m=(0.45, 0.04)), 17, spawn_range)


def test_random_reset_and_oracle_replanning():
    model = load_scene()
    data = mujoco.MjData(model)
    expert = MotorExpert(model)
    grasp_targets = []
    for seed, orientation in zip((17, 21), Orientation):
        xy = sample_initial_xy(DEFAULT_SCENE, seed, SpawnRange.LARGE)
        reset_scene(model, data, orientation, seed, spawn_range=SpawnRange.LARGE)
        center, rotation = oracle_pose(data)
        np.testing.assert_array_equal(center[:2], xy)
        direction = 1 if orientation is Orientation.PROTRUSION else -1
        np.testing.assert_allclose(rotation[:, 2], [0, direction, 0], atol=1e-12)
        result = expert.run(data, orientation, seed, spawn_range=SpawnRange.LARGE)
        assert result.success, result
        np.testing.assert_array_equal(result.initial_xy_m, xy)
        offset = np.array(result.planned_grasp_tcp_m[:2]) - xy
        assert np.linalg.norm(offset) < 0.01
        grasp_targets.append(np.array(result.planned_grasp_tcp_m[:2]))
    assert np.linalg.norm(grasp_targets[0] - grasp_targets[1]) > 0.01
