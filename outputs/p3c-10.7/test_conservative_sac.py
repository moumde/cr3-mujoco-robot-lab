import io

import numpy as np
import pytest
import torch

from conservative_sac import (ROOT, ConservativeSAC, DistilledPolicy, Gain, TrainPlant, BudgetedResidualEnv,
                   action, load_policy, model_at, panels, violations)


def test_actor_copy_reload_and_training_seed_guard(tmp_path):
    torch.set_num_threads(1)
    env = Gain(BudgetedResidualEnv(TrainPlant(), load_policy()))
    try:
        model = ConservativeSAC(DistilledPolicy, env, device="cpu", ent_coef=.005,
                                policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh))
        reference = model_at(ROOT.parent / "p3c-10.5/uniform/model.pt")
        model.actor.initialize(reference)
        obs, _ = env.reset(seed=11)
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0] * .25,
                                      action(reference, obs, .25))
        _, log_std, _ = model.actor.get_action_dist_params(torch.from_numpy(obs)[None])
        np.testing.assert_array_equal(log_std.detach().numpy(), [[-3, -3, -3]])
        assert all(not p.requires_grad for p in model.actor.reference.parameters())
        assert len(model.actor.optimizer.param_groups[0]["params"]) == 8
        model.update_stream = io.StringIO()
        model.save(tmp_path / "policy")
        restored = ConservativeSAC.load(tmp_path / "policy", device="cpu")
        np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0],
                                      model.predict(obs, deterministic=True)[0])
        with pytest.raises(ValueError, match="outside training"):
            env.reset(options={"scene_seed": 18080})
        assert len(panels()) == 62
    finally:
        env.close()


def test_gate_halts_on_paired_regression_or_drift():
    base = dict(scene_seed=17208, orientation="flat", is_success=True, collision=None,
                reason="success", absolute_m=[0, 0, 0])
    assert not violations([base], [base])
    assert violations([{**base, "is_success": False}], [base])
    assert violations([{**base, "collision": "table"}], [base])
    assert violations([{**base, "absolute_m": [.025, 0, 0]}], [base])
