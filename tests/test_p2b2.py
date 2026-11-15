
import mujoco
import numpy as np
import pytest

from cr3_lab.perception.camera import VisualEstimate
from cr3_lab.tasks import motor_expert, p2b2, vision_score
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation, SpawnRange, load_scene, sample_initial_xy


class BlankRenderer:
    def update_scene(self, data, name):
        assert name in p2b2.CAMERAS

    def render(self):
        return np.zeros((p2b2.HEIGHT, p2b2.WIDTH, 3), dtype=np.uint8)


def test_invalid_rgb_stops_before_approach():
    model = load_scene()
    episode = p2b2.VisionEpisode(model, BlankRenderer())
    result = episode.run(mujoco.MjData(model), Orientation.FLAT, 5000, SpawnRange.FIXED)
    assert result["phases"] == ["Initialize"]
    assert result["failure_categories"] == ["perception"]
    assert not result["completed"] and not result["success"]


@pytest.mark.parametrize("profile,orientation", [(SpawnRange.FIXED, Orientation.FLAT),
                                                 (SpawnRange.LARGE, Orientation.PROTRUSION)])
def test_vision_control_is_independent_of_oracle_observer(monkeypatch, profile, orientation):
    model = load_scene()
    episode = p2b2.VisionEpisode(model, BlankRenderer())
    data = mujoco.MjData(model)
    xy = sample_initial_xy(DEFAULT_SCENE, 5000, profile)
    estimate = VisualEstimate(tuple(xy + [0.0003, -0.0002]), orientation, 0.8)
    monkeypatch.setattr(p2b2.RgbMotorEstimator, "predict", lambda self, *rgb: estimate)
    original_step = episode.robot.arm.step
    commands = []

    def record(data, target):
        commands.append((target.copy(), float(data.ctrl[episode.robot.gripper.actuator])))
        original_step(data, target)

    def forbidden(*args):
        raise AssertionError("Oracle expert accessed during vision execution")

    monkeypatch.setattr(episode.robot.arm, "step", record)
    monkeypatch.setattr(motor_expert, "oracle_pose", forbidden)
    monkeypatch.setattr(motor_expert.MotorExpert, "run", forbidden)
    normal = episode.run(data, orientation, 5000, profile)
    assert normal["success"], normal
    expected = np.array([np.r_[q, closure] for q, closure in commands])
    commands.clear()

    # Poison ground truth in both evaluation consumers, including the passive observer.
    original_pose = vision_score.oracle_pose

    def poisoned(data):
        center, rotation = original_pose(data)
        return center + [0.2, -0.2, -0.1], rotation @ np.diag([1, -1, -1])

    monkeypatch.setattr(vision_score, "oracle_pose", poisoned)
    monkeypatch.setattr(p2b2, "oracle_pose", poisoned)
    corrupted = episode.run(data, orientation, 5000, profile)
    assert corrupted["completed"] and not corrupted["success"]
    np.testing.assert_array_equal(np.array([np.r_[q, closure] for q, closure in commands]), expected)
