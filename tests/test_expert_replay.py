import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import SAC

from cr3_lab.rl.expert_replay import collect, prefill, verify
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.train_grasp_lift import SCALE, ScaledGrasp


def test_expert_replay_round_trip_and_prefill():
    data = collect()
    verify(data, SCALE)
    env = ScaledGrasp(MotorGraspLiftEnv(start="pregrasp"))
    model = SAC("MlpPolicy", env, buffer_size=4000, seed=11)
    count = prefill(model, data, SCALE)
    buffer = model.replay_buffer
    assert count == buffer.size() and model.num_timesteps == 0
    np.testing.assert_array_equal(buffer.observations[:count, 0], data["observation"] * SCALE)
    np.testing.assert_array_equal(buffer.next_observations[:count, 0], data["next_observation"] * SCALE)
    np.testing.assert_array_equal(buffer.actions[:count, 0], data["action"])
    np.testing.assert_allclose(buffer.rewards[:count, 0], data["reward"], atol=1e-7, rtol=0)
    np.testing.assert_array_equal(np.flatnonzero(buffer.dones[:count, 0]), data["offsets"][1:] - 1)
    assert not buffer.timeouts[:count].any()
    with pytest.raises(ValueError, match="empty buffer"):
        prefill(model, data, SCALE)
    env.close()
