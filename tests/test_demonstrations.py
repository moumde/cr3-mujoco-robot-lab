import mujoco
import numpy as np

from cr3_lab.demonstrations.collect import FPS, Recorder, scene
from cr3_lab.tasks.motor_expert import MotorExpert
from cr3_lab.tasks.motor_scene import Orientation, reset_scene


def test_recorder_samples_before_command_and_keeps_exact_controls():
    model = scene(False)
    robot = MotorExpert(model)
    data = mujoco.MjData(model)
    reset_scene(model, data, Orientation.FLAT, 5400)
    _, joints = robot.grasp_ready(data)
    recorder = Recorder(model, None)
    capture_times = []

    def capture(data):
        capture_times.append(data.time)
        return None

    recorder.images = capture
    with recorder.intercept(robot):
        for i in range(6):
            before = robot.arm.state(data).position_rad
            robot.gripper.command_closure(data, 1., "gradual")
            robot.arm.step(data, joints)
            np.testing.assert_array_equal(recorder.rows[-1]["state"][:6], before)
            np.testing.assert_allclose(recorder.rows[-1]["action"], np.r_[joints, .05 * (i + 1)])
    np.testing.assert_allclose(capture_times, [0, 1 / FPS], atol=1e-12)
    np.testing.assert_allclose([r["time_s"] for r in recorder.rows], np.arange(6) / 50)
