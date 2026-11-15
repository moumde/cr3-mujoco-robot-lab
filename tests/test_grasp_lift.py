import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from cr3_lab.rl.check_grasp_lift import expert_actions
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.tasks.motor_scene import SpawnRange


def test_seed_and_ramp():
    env = MotorGraspLiftEnv(SpawnRange.LARGE)
    first, info = env.reset(seed=12)
    after, *_ = env.step([0, 0, 0, 1])
    assert after[30] == pytest.approx(0.05)
    assert env.observation_space.contains(after)
    repeated, repeated_info = env.reset(seed=12)
    np.testing.assert_array_equal(first, repeated)
    assert info == repeated_info
    _, toward, *_ = env.step([0, 0, -1, -1])
    env.reset(seed=12)
    _, away, *_ = env.step([0, 0, 1, -1])
    assert toward > away


@pytest.mark.parametrize("start", ["grasp_ready", "pregrasp"])
def test_physical_expert_and_drop(start):
    env = MotorGraspLiftEnv(start=start)
    env.reset(seed=10000)
    for action in expert_actions(env):
        _, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["is_success"], info
    assert reward > 0 and info["max_lift_m"] >= 0.10
    assert info["first_bilateral_step"] < info["first_lift_step"] < info["steps"]
    assert env.model.neq == 1  # The only equality couples the two fingers.

    env.reset(seed=10000)
    for action in expert_actions(env):
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        if info["first_lift_step"] is not None:
            break
    assert info["first_lift_step"] is not None
    for _ in range(12):
        _, reward, terminated, _, info = env.step([0, 0, 0, -1])
        if terminated:
            break
    assert terminated and info["reason"] == "lost_grip" and reward <= -5


def test_collision_and_timeout():
    env = MotorGraspLiftEnv(start="pregrasp")
    env.reset(seed=10000)
    for _ in range(100):
        _, reward, terminated, _, info = env.step([0, 0, -1, -1])
        if terminated:
            break
    assert terminated and info["reason"] == "unexpected_contact" and reward <= -5

    env = MotorGraspLiftEnv(max_steps=2)
    env.reset(seed=10000)
    assert env.step([0, 0, 0, -1])[2:4] == (False, False)
    assert env.step([0, 0, 0, -1])[2:4] == (False, True)


def test_discounted_contact_cycles_and_fault_cannot_pay():
    from cr3_lab.rl.grasp_lift import GAMMA, shaped_reward

    for before, after in ((0.2, 1.0), (1.0, 2.0), (2.0, 2.0)):
        enter = shaped_reward(before, after, False, False, False)
        leave = shaped_reward(after, before, False, False, False)
        assert enter + GAMMA * leave < 0
        fault = shaped_reward(after, after, False, False, True)
        assert enter + GAMMA * fault < 0
