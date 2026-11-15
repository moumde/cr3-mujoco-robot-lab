import numpy as np
import torch

from r1 import (ROOT, BudgetedResidualEnv, DistilledPolicy, Gain, R1SAC, TrainPlant,
                action, calibration, load_policy, model_at, snapshot)


def test_calibration_is_below_preregistered_drift():
    reference = model_at(ROOT.parent / "p3c-10.5/uniform/model.pt")
    result = calibration(reference, load_policy())
    assert result["steps"] == 211
    assert max(result["exploration_abs"][2]) < .25
    assert max(result["exploration_projected_mm"][2]) < .046875


def test_distribution_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    env = Gain(BudgetedResidualEnv(TrainPlant(), load_policy()))
    try:
        reference = model_at(ROOT.parent / "p3c-10.5/uniform/model.pt")
        model = R1SAC(DistilledPolicy, env, seed=11, ent_coef=.005, device="cpu",
                      buffer_size=2000, policy_kwargs={"net_arch": [64, 64],
                                                       "activation_fn": torch.nn.Tanh})
        model.actor.initialize(reference)
        obs, _ = env.reset(seed=11)
        np.testing.assert_allclose(model.predict(obs, deterministic=True)[0], action(reference, obs, 1.),
                                   atol=5e-8, rtol=0)
        inputs = torch.from_numpy(obs)[None]
        mean, logstd, _ = model.actor.get_action_dist_params(inputs)
        sampled, logp = model.actor.action_log_prob(inputs)
        latent = sampled.atanh()
        normal = torch.distributions.Normal(mean, logstd.exp())
        expected = (normal.log_prob(latent) - (1 - sampled.square() + 1e-6).log()).sum(-1)
        torch.testing.assert_close(logp, expected, atol=1e-4, rtol=1e-4)
        initial = {key: module.state_dict() for key, module in
                   (("actor", model.actor), ("critic", model.critic), ("reference", model.actor.reference))}
        assert snapshot(tmp_path, model, 0, initial) == dict(actor=0, critic=0)
        assert {id(p) for p in model.actor.reference.parameters()}.isdisjoint(
            id(p) for group in model.actor.optimizer.param_groups for p in group["params"])
    finally:
        env.close()
