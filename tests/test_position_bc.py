import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from cr3_lab.rl.grasp_benchmark import execute
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.position_bc import TEST, TRAIN, VALIDATION, check_splits, xy
from cr3_lab.tasks.motor_scene import Orientation


def test_xy_splits_are_disjoint_and_cover_edges():
    check_splits()
    for split in (TRAIN, VALIDATION, TEST):
        points = np.array([xy(seed) for seed in split])
        offsets = points - [0.45, 0.09]
        assert np.all(abs(offsets) <= 0.025)
        assert np.any(np.max(abs(offsets), axis=1) >= 0.020)


def test_closed_loop_policy_receives_current_observation():
    env = MotorGraspLiftEnv(max_steps=2)
    seen = []
    actions = []

    def policy(obs):
        seen.append(obs.copy())
        return np.array([0, 0, -0.5, -1], dtype=np.float32)

    initial, result = execute(env, 16000, Orientation.FLAT, policy=policy,
                              on_action=lambda obs, action: actions.append(action.copy()))
    np.testing.assert_array_equal(seen[0], initial)
    assert not np.array_equal(seen[0], seen[1])
    assert len(actions) == 2 and result["truncated"]
