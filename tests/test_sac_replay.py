"""Audit SB3 timeout bootstrapping and terminal observations on CR3 Reach."""
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from cr3_lab.rl.evaluate_reach import RelativeReach
from cr3_lab.rl.reach import MotorReachEnv


@pytest.mark.parametrize("success", [False, True])
def test_replay_preserves_final_transition(success):
    env = MotorReachEnv(max_steps=150 if success else 1)
    wrapped = RelativeReach(env)
    vec = DummyVecEnv([lambda: wrapped])
    model = SAC("MlpPolicy", vec, buffer_size=200, learning_starts=200, seed=11)
    model._last_obs = vec.reset()
    if success:
        env.reach_goal = env.tcp_target.copy()
        model._last_obs = wrapped.observation(env._observation())[None]
    previous = model._last_obs.copy()
    action = np.array([[0, 0, -0.5]], dtype=np.float32)
    obs, reward, done, infos = vec.step(action)
    assert done[0] and infos[0]["is_success"] == success
    assert infos[0]["TimeLimit.truncated"] == (not success)
    model._store_transition(model.replay_buffer, action, obs, reward, done, infos)
    buffer = model.replay_buffer
    np.testing.assert_array_equal(buffer.observations[0], previous)
    np.testing.assert_array_equal(buffer.actions[0], action)
    np.testing.assert_array_equal(buffer.rewards[0], reward)
    np.testing.assert_array_equal(buffer.next_observations[0, 0], infos[0]["terminal_observation"])
    assert not np.array_equal(buffer.next_observations[0, 0], obs[0])
    assert buffer.dones[0, 0] == 1
    assert buffer.timeouts[0, 0] == int(not success)
    assert buffer.sample(1).dones.item() == int(success)
    vec.close()


def test_r1_test_seeds_are_held_out():
    from cr3_lab.rl.evaluate_reach import R1_TEST_SEEDS
    from cr3_lab.rl.reach import TEST_SEEDS, TRAIN_SEEDS
    from cr3_lab.rl.train_reach import VALIDATION_SEEDS

    assert set(R1_TEST_SEEDS).isdisjoint((*TRAIN_SEEDS, *VALIDATION_SEEDS, *TEST_SEEDS))
