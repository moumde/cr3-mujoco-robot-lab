import mujoco
import numpy as np
import pytest

from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.tasks.motor_expert import MotorExpert
from cr3_lab.tasks.motor_scene import Orientation, load_scene, reset_scene
from cr3_lab.tasks.p4a import LearnedGrasp


def test_attach_ready_observation_preserves_shared_physics():
    model = load_scene()
    data = mujoco.MjData(model)
    robot = MotorExpert(model)
    reset_scene(model, data, Orientation.FLAT, 19000)
    ready, joints = robot.grasp_ready(data)
    learner = LearnedGrasp(model, lambda obs: np.array([0, 0, 0, -1], dtype=np.float32),
                           lambda obs: np.zeros(3, dtype=np.float32),
                           {"xyz_scale": (.5, .5, .5), "signed_budget_m": (.006, .003, .012)})
    before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time
    obs, _ = learner.plant.attach_ready_state(data, Orientation.FLAT, 19000, ready, joints)
    augmented = learner.env.begin_grasp(obs)
    assert learner.plant.model is model and learner.plant.data is data
    for before_array, after in zip(before, (data.qpos, data.qvel, data.ctrl, data.time), strict=True):
        np.testing.assert_array_equal(before_array, after)
    assert augmented.shape == (43,)
    np.testing.assert_array_equal(augmented[40:], np.zeros(3))
    learner.env.step(np.zeros(3, dtype=np.float32))
    assert data.time == pytest.approx(.02)


@pytest.mark.parametrize("orientation,yaw_sign", [(Orientation.PROTRUSION, 1), (Orientation.FLAT, -1)])
def test_carried_planner_starts_at_actual_tcp(monkeypatch, orientation, yaw_sign):
    from cr3_lab.perception.camera import VisualEstimate
    from cr3_lab.tasks import vision_motion
    from cr3_lab.tasks.motor_expert import DEFAULT_EXPERT
    from cr3_lab.tasks.motor_scene import DEFAULT_SCENE

    calls = []
    q = np.ones(6) * .1
    tcp = np.array([.451, .091, .27])

    def targets(ik, cfg, joints, start, end, a, b):
        calls.append((joints, start, end, a, b))
        return [q]

    sent = []
    monkeypatch.setattr(vision_motion, "cartesian_targets", targets)
    vision_motion.execute(VisualEstimate((.45,.09), orientation, .9), DEFAULT_SCENE,
                          DEFAULT_EXPERT, None, tcp, q, lambda phase, joints, closure: sent.append((phase, closure)),
                          carried=(tcp, np.pi / 2))
    assert sent[0] == ("Reorient", 1)
    assert calls[0][0] is q
    np.testing.assert_array_equal(calls[0][1], tcp)
    np.testing.assert_array_equal(calls[0][2], tcp)
    assert calls[0][4] - calls[0][3] == pytest.approx(yaw_sign * np.pi / 2)
