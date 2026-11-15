import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from torch import nn

from residual import ResidualEnv, TrainingChallenge, initialize_sac_actor, load_policy
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv


def test_training_reset_and_residual_masks():
    env = ResidualEnv(TrainingChallenge(), load_policy(), (0.5, 0.5, 0.5, 0.05))
    obs, _ = env.reset(seed=9)
    repeated, _ = env.reset(seed=9)
    np.testing.assert_array_equal(obs, repeated)
    base = env.base_action()
    _, _, _, _, info = env.step([0, 0, 0, 1])
    np.testing.assert_array_equal(info["residual_control"]["final_action"], base)
    with pytest.raises(ValueError, match="training pool"):
        env.reset(options={"scene_seed": 17700})
    env.close()


def test_sac_residual_probability_and_replay_timeout():
    env = ResidualEnv(MotorGraspLiftEnv(max_steps=1), load_policy())
    model = SAC("MlpPolicy", env, buffer_size=10, seed=11, device="cpu",
                policy_kwargs={"net_arch": [64, 64], "activation_fn": nn.Tanh})
    initialize_sac_actor(model.actor)
    vec = model.get_env()
    obs = vec.reset()
    with torch.no_grad():
        deterministic, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(deterministic, np.zeros((1, 4)))
        action, log_prob = model.actor.action_log_prob(torch.as_tensor(obs))
        assert torch.isfinite(log_prob).all()
        assert torch.all(abs(action) <= 1)
    normalized = action.numpy()
    model._last_obs = obs.copy()
    following, reward, done, infos = vec.step(normalized)
    assert done[0] and infos[0]["TimeLimit.truncated"]
    model._store_transition(model.replay_buffer, normalized, following, reward, done, infos)
    buf = model.replay_buffer
    np.testing.assert_array_equal(buf.actions[0], normalized)
    np.testing.assert_array_equal(buf.observations[0], obs)
    np.testing.assert_array_equal(buf.next_observations[0, 0], infos[0]["terminal_observation"])
    np.testing.assert_array_equal(buf.rewards[0], reward)
    assert not np.allclose(buf.actions[0, 0], infos[0]["residual_control"]["final_action"])
    assert buf.sample(1).dones.item() == 0
    assert model._n_updates == 0
    vec.close()


def test_composition_bounds_and_independent_gripper():
    from residual import compose

    base = np.array([0.9, -0.9, 0.2, -0.9], dtype=np.float32)
    residual = np.array([1, -1, 0, -1], dtype=np.float32)
    np.testing.assert_allclose(compose(base, residual, [0.5, 0.5, 0.5, 0]), [1, -1, 0.2, -0.9])
    np.testing.assert_allclose(compose(base, residual, [0, 0, 0, 0.2]), [0.9, -0.9, 0.2, -1])
