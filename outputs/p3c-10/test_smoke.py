import numpy as np
import torch

from smoke import CASES, XYZResidual, create
from residual import TRAIN_SCENES, load_policy
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv


def test_three_dimensional_sac_warmup_and_timeout():
    assert not {seed for _, seed, _ in CASES}.intersection(TRAIN_SCENES)
    assert not {seed for _, seed, _ in CASES}.intersection(range(17700, 17740))
    env = XYZResidual(MotorGraspLiftEnv(max_steps=1), load_policy(), [.1, .2, .5])
    model = create(env, 11, -3)
    vec = model.get_env()
    obs = model._last_obs = vec.reset()
    assert obs.shape == (1, 40) and model.target_entropy == -3
    action, _ = model.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action, np.zeros((1, 3)))
    np.testing.assert_allclose(env.action([1, -1, 0.5]), [1, -1, 0.5, 0])
    with torch.no_grad():
        samples, log_prob = model.actor.action_log_prob(torch.as_tensor(obs).repeat(2048, 1))
        assert torch.isfinite(log_prob).all() and samples.shape == (2048, 3)
        assert 0.08 < torch.quantile(abs(samples), .95).item() < .11
    action, residual = model._sample_action(500)
    assert max(abs(residual[0])) < .25
    next_obs, rewards, done, infos = vec.step(action)
    assert done[0] and infos[0]["TimeLimit.truncated"]
    ctrl = infos[0]["residual_control"]
    assert ctrl["final_action"][3] == ctrl["bc_action"][3]
    model._store_transition(model.replay_buffer, residual, next_obs, rewards, done, infos)
    replay = model.replay_buffer.sample(1)
    np.testing.assert_allclose(replay.actions.numpy(), residual)
    np.testing.assert_array_equal(replay.next_observations[0].numpy(), infos[0]["terminal_observation"])
    assert replay.dones.item() == 0
    assert model._n_updates == 0
    vec.close()
