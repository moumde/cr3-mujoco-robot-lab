import numpy as np
import torch
from ppo import (ROOT, BudgetedResidualEnv, DistilledPPOPolicy, MonitoredGain,
                 ResidualPPO, TrainPlant, action, load_policy, model_at)


def test_probability_init_and_checkpoint(tmp_path):
    env = MonitoredGain(BudgetedResidualEnv(TrainPlant(), load_policy()))
    try:
        student = model_at(ROOT.parent / "p3c-10.5/uniform/model.pt")
        m = ResidualPPO(DistilledPPOPolicy, env, n_steps=250, batch_size=125,
                        policy_kwargs=dict(ortho_init=False))
        m.policy.initialize(student)
        obs, _ = env.reset(seed=11)
        np.testing.assert_array_equal(m.predict(obs, deterministic=True)[0], action(student, obs, 1.))
        tensor = torch.from_numpy(obs)[None].repeat(32, 1)
        sampled, _, old = m.policy(tensor)
        _, new, entropy = m.policy.evaluate_actions(tensor, sampled)
        torch.testing.assert_close((new-old).exp(), torch.ones_like(old))
        assert entropy is None and sampled.abs().max() < 1
        dist = m.policy.get_distribution(tensor).distribution
        expected = (dist.log_prob(sampled.atanh()) - (1-sampled.square()+1e-6).log()).sum(-1)
        torch.testing.assert_close(new, expected, atol=1e-3, rtol=1e-4)
        m.save(tmp_path/"model")
        loaded = ResidualPPO.load(tmp_path/"model")
        np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0],
                                      m.predict(obs, deterministic=True)[0])
    finally:
        env.close()
