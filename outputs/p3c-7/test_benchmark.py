import numpy as np

from benchmark import ChallengeEnv, DEV, TEST, clearance, sample_xy
from cr3_lab.tasks.motor_scene import MotorDimensions


def test_spawn_bounds_splits_and_nominal_arm():
    seeds = [seed for cases in (DEV, TEST) for group in cases.values() for seed in group]
    assert len(seeds) == len(set(seeds))
    assert min(clearance(0.035)) > 0
    env = ChallengeEnv()
    env.xy = sample_xy(0.035, DEV[35][0])
    first, _ = env.reset(seed=1)
    env.xy = sample_xy(0.035, DEV[35][1])
    second, _ = env.reset(seed=1)
    np.testing.assert_array_equal(first[:12], second[:12])
    np.testing.assert_array_equal(first[37:40], second[37:40])
    assert not np.array_equal(first[15:17], second[15:17])
    motor = MotorDimensions()
    assert env.model.body("motor").mass[0] == motor.mass_kg + motor.boss_mass_kg
    env.close()
