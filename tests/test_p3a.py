import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from cr3_lab.core.control import STEPS_PER_COMMAND
from cr3_lab.core.model import TIMESTEP_S
from cr3_lab.rl.reach import MotorReachEnv, TCP_DELTA_M, TRAIN_SEEDS, TEST_SEEDS
from cr3_lab.tasks.motor_scene import SpawnRange


@pytest.fixture(scope="module")
def env():
    return MotorReachEnv(SpawnRange.LARGE, max_steps=80)


def test_reset_and_seed(env):
    assert not set(TRAIN_SEEDS) & set(TEST_SEEDS)
    first, info = env.reset(seed=1)
    env.step([0, 0, 1])
    second, repeated = env.reset(seed=1)
    np.testing.assert_array_equal(first, second)
    assert info == repeated
    third, _ = env.reset(seed=2)
    assert not np.array_equal(first[15:18], third[15:18])
    assert first.shape == (32,) and env.observation_space.contains(first)


def test_random_actions_and_step_size(env):
    env.reset(seed=5)
    env.action_space.seed(6)
    with pytest.raises(ValueError):
        env.step([0, 0, 2])
    with pytest.raises(ValueError):
        env.step([np.nan, 0, 0])
    for _ in range(160):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        before = env.data.time
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.data.time - before == pytest.approx(TIMESTEP_S * STEPS_PER_COMMAND)
        assert env.observation_space.contains(obs)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            env.reset()


def test_success_and_timeout(env):
    env.reset(seed=9000, options={"scene_seed": 9000})
    for _ in range(env.max_steps):
        delta = np.clip((env.reach_goal - env.tcp_target) / TCP_DELTA_M, -1, 1)
        _, reward, terminated, truncated, info = env.step(delta)
        if terminated or truncated:
            break
    assert terminated and not truncated and info["is_success"] and reward > 0
    timeout = MotorReachEnv(SpawnRange.FIXED, max_steps=2)
    timeout.reset(seed=9000)
    assert timeout.step(np.zeros(3))[2:4] == (False, False)
    _, _, terminated, truncated, info = timeout.step(np.zeros(3))
    assert truncated and not terminated and info["reason"] == "timeout"
    with pytest.raises(gym.error.ResetNeeded, match="Call reset"):
        timeout.step(np.zeros(3))


def test_substep_observer_preserves_step(env):
    import mujoco

    env.reset(seed=9000)
    data = env.data
    reference = mujoco.MjData(env.model)
    mujoco.mj_copyData(reference, env.model, data)
    timestamps = []
    target = env.joint_target + 0.001
    env.robot.arm.step(reference, target)
    env.robot.arm.step(data, target, observe_substep=lambda state: timestamps.append(state.time))
    assert len(timestamps) == STEPS_PER_COMMAND
    np.testing.assert_allclose(np.diff(timestamps), TIMESTEP_S, atol=1e-12)
    np.testing.assert_allclose(data.qpos, reference.qpos, atol=1e-10, rtol=0)
    np.testing.assert_allclose(data.qvel, reference.qvel, atol=1e-10, rtol=0)


def test_reach_reward_progress_and_ik_rejection(env, monkeypatch):
    from cr3_lab.tasks.ik import Unreachable

    env.reset(seed=9001)
    before = env._distance()
    _, approaching_reward, _, _, info = env.step([0, 0, -1])
    assert info["distance_m"] < before and approaching_reward > -0.005
    assert env.data.ctrl[env.robot.gripper.actuator] == 0

    env.reset(seed=9001)
    def reject(*args):
        raise Unreachable("test rejection")

    monkeypatch.setattr(env.robot.ik, "solve", reject)
    held = env.tcp_target.copy()
    for _ in range(5):
        _, reward, terminated, _, info = env.step([0, 0, -1])
        assert info["ik_rejected"] and not terminated and reward == pytest.approx(-0.015, abs=1e-6)
        np.testing.assert_array_equal(env.tcp_target, held)


def test_gripper_closure_modes(env):
    env.reset(seed=1)
    gripper = env.robot.gripper
    for tick in range(1, 21):
        gripper.command_closure(env.data, 1, "gradual")
        assert env.data.ctrl[gripper.actuator] / gripper.travel_m == pytest.approx(tick / 20)
    gripper.command_closure(env.data, 0, "gradual")
    assert env.data.ctrl[gripper.actuator] == 0
    gripper.command_closure(env.data, 1, "fast")
    assert env.data.ctrl[gripper.actuator] == gripper.travel_m
