import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from cr3_lab.rl.behavior_clone import ambiguity, features, model_for


@pytest.mark.parametrize("clock", [True, False])
def test_policy_features_and_action_range(clock):
    obs = np.arange(40, dtype=np.float32)
    x = features(obs, clock)
    np.testing.assert_array_equal(x, obs if clock else np.r_[obs[:36], obs[37:]])
    action = model_for(clock)(torch.from_numpy(x)).detach().numpy()
    assert action.shape == (4,)
    assert np.isfinite(action).all() and np.max(abs(action)) <= 1


def test_ambiguity_excludes_matching_trajectory_times():
    states = np.array([[0.], [0.003], [0.], [0.003]])
    actions = np.array([[-1.], [1.], [-1.], [1.]])
    report = ambiguity(states, actions, [0, 2, 4])
    assert report["near_pairs"] == report["conflicts"] == 4
    assert report["examples"][0]["steps"] == [0, 1]


def test_robustness_offsets_keep_nominal_pregrasp():
    from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv

    env = MotorGraspLiftEnv(start="pregrasp")
    nominal, _ = env.reset(seed=14000)
    shifted, _ = env.reset(seed=14000, options={"fixed_pregrasp_offset_m": [0.01, -0.005]})
    np.testing.assert_allclose(shifted[37:40], nominal[37:40], atol=1e-7)
    np.testing.assert_allclose(shifted[15:17] - nominal[15:17], [0.01, -0.005], atol=1e-5)
    for offset in ([0.011, 0], [np.nan, 0], [0]):
        with pytest.raises(ValueError, match="XY offset"):
            env.reset(seed=14000, options={"fixed_pregrasp_offset_m": offset})


def test_redundancy_compares_aligned_timesteps():
    from cr3_lab.rl.bc_robustness import redundancy

    data = {"offsets": [0, 2, 4], "observation": np.zeros((4, 40)),
            "action": np.array([[0] * 4, [1] * 4, [0] * 4, [0] * 4])}
    report = redundancy(data)
    assert report["aligned_pairs"] == 2
    assert report["observation_exact"] == 1
    assert report["action_exact"] == 0.5
