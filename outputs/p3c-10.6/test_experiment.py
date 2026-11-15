import numpy as np
import torch

from experiment import GAINS, ResidualMLP, action, passes, plan


def test_gate_requires_all_criteria():
    valid = dict(preserved=34, nominal=2, rescued=1, new_contact=0)
    assert passes(valid)
    for field in valid:
        assert not passes({**valid, field: (1 if field == "new_contact" else 0)})
    assert GAINS == (.25, .5, .75, 1.)


def test_gain_multiplies_normalized_model_output():
    model = ResidualMLP(torch.zeros(43), torch.ones(43)).eval()
    obs = np.ones(43, dtype=np.float32)
    np.testing.assert_array_equal(action(model, obs, .25), action(model, obs, 1.) * .25)
    assert np.max(np.abs(action(model, obs, .25))) <= .25


def test_dagger_seed_range_uses_train_only():
    assert [seed for seed, _, split in plan() if split == "train"] == list(range(18000, 18080))
