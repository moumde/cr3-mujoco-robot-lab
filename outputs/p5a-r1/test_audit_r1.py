import numpy as np
from ppo_update_audit import advantage


def test_gae_bootstraps_timeout_without_cross_episode_leakage():
    reward = np.array([1., 2., 3.])
    values = np.array([.1, .2, .3])
    next_values = np.array([.2, .9, 12.])
    result = advantage(reward, values, next_values,
                       terminal=[False, False, True], boundary=[False, True, True])
    np.testing.assert_allclose(result, [1 + .99*.2 - .1 + .99*.95*(2 + .99*.9 - .2),
                                       2 + .99*.9 - .2, 3 - .3])
