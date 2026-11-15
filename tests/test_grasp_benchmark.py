import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from cr3_lab.rl import grasp_benchmark as benchmark
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.tasks.motor_scene import Orientation, SpawnRange


def test_reference_is_full_ready_trajectory_and_replay_never_replans(monkeypatch):
    actions, original = benchmark.record_reference()
    assert actions.shape == (original["steps"], 4)
    assert original["start"] == "grasp_ready" and original["steps"] > 240

    def forbidden(*args):
        raise AssertionError("Replay asked for expert actions")

    monkeypatch.setattr(benchmark, "expert_actions", forbidden)
    def execute(profile, seed):
        env = MotorGraspLiftEnv(profile)
        return benchmark.execute(env, seed, Orientation.PROTRUSION, lambda: actions)

    fixed, result = execute(SpawnRange.FIXED, benchmark.REFERENCE_SEED)
    random, _ = execute(SpawnRange.LARGE, benchmark.REFERENCE_SEED)
    assert result["is_success"]
    np.testing.assert_array_equal(fixed[37:40], random[37:40])
    assert not np.allclose(fixed[15:17], random[15:17])
