from dataclasses import replace

import mujoco
import numpy as np
import pytest

from cr3_lab.core.model import TCP
from cr3_lab.tasks.ik import Unreachable
from cr3_lab.tasks.motor_expert import DEFAULT_EXPERT, MotorExpert, tool_rotation
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation, load_scene, oracle_pose, reset_scene
from cr3_lab.tasks.p1a import summarize


@pytest.fixture(scope="module")
def model():
    return load_scene()


@pytest.mark.parametrize("orientation, sign", [(Orientation.PROTRUSION, 1), (Orientation.FLAT, -1)])
def test_reset_camera_label_and_fixed_position(model, orientation, sign):
    data = mujoco.MjData(model)
    reset_scene(model, data, orientation, 17)
    center, rotation = oracle_pose(data)
    np.testing.assert_allclose(center, [*DEFAULT_SCENE.initial_xy_m, DEFAULT_SCENE.rest_z_m])
    np.testing.assert_allclose(rotation[:, 2], [0, sign, 0], atol=1e-12)
    assert np.sign(rotation[:, 2] @ (data.cam("inspection").xpos - center)) == sign
    assert model.geom("motor_housing").size[0] * 2 < 0.074
    assert model.nq == 15 and model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT


def test_ik_reachable_and_unreachable(model):
    expert = MotorExpert(model)
    data = mujoco.MjData(model)
    position = [*DEFAULT_SCENE.initial_xy_m, DEFAULT_EXPERT.ready_tcp_z_m]
    rotation = tool_rotation(np.pi / 2)
    q = expert.ik.solve(position, rotation, DEFAULT_EXPERT.ready_ik_seed_rad)
    lower, upper = expert.arm.limits_rad.T
    assert np.all((lower <= q) & (q <= upper))
    expert.ik.data.qpos[expert.arm.qpos] = q
    mujoco.mj_kinematics(model, expert.ik.data)
    np.testing.assert_allclose(expert.ik.data.site(TCP).xpos, position, atol=0.00015)
    np.testing.assert_allclose(expert.ik.data.site(TCP).xmat.reshape(3, 3), rotation, atol=0.001)
    with pytest.raises(Unreachable):
        expert.ik.solve([2., 0., 0.28], rotation, q)
    np.testing.assert_array_equal(data.qpos, model.qpos0)


@pytest.mark.parametrize("orientation", list(Orientation))
def test_complete_episode(model, orientation):
    data = mujoco.MjData(model)
    result = MotorExpert(model).run(data, orientation, 17)
    assert result.success, result
    assert result.phases == ["Initialize", "Approach", "Pre-Grasp", "Grasp", "Lift",
                             "Reorient", "Transfer", "Place", "Release", "Retreat"]
    assert result.grasped and result.stable and not result.dropped
    assert result.max_lift_m >= DEFAULT_EXPERT.min_lift_m
    sign = 1 if orientation is Orientation.PROTRUSION else -1
    assert result.planned_yaw_deg == pytest.approx(sign * 90, abs=2)
    assert result.measured_yaw_deg == pytest.approx(sign * 90, abs=2)
    center, rotation = oracle_pose(data)
    assert center[1] < 0
    np.testing.assert_allclose(center, [*DEFAULT_SCENE.target_xy_m, DEFAULT_SCENE.rest_z_m],
                               atol=DEFAULT_EXPERT.position_tolerance_m)
    assert rotation[:, 2] @ [-1, 0, 0] >= np.cos(np.radians(DEFAULT_EXPERT.angle_tolerance_deg))


@pytest.mark.parametrize("orientation", list(Orientation))
def test_grasp_ready(model, orientation):
    data = mujoco.MjData(model)
    expert = MotorExpert(model)
    reset_scene(model, data, orientation, 17)
    target, joints = expert.grasp_ready(data)
    np.testing.assert_allclose(data.site(TCP).xpos, target, atol=0.00015)
    np.testing.assert_allclose(data.site(TCP).xmat.reshape(3, 3), tool_rotation(np.pi / 2),
                               atol=0.001)
    np.testing.assert_array_equal(data.qpos[expert.arm.qpos], joints)
    np.testing.assert_array_equal(data.qvel, 0)
    np.testing.assert_allclose(data.qacc[expert.arm.dofs], 0, atol=1e-4)
    assert data.ctrl[expert.gripper.actuator] == 0
    for name in ("left_slide", "right_slide"):
        assert data.joint(name).qpos[0] == 0
    assert model.npair > 0
    allowed = {model.geom("table").id, model.geom("motor_housing").id}
    assert all(set(contact.geom) == allowed for contact in data.contact)


def test_unreachable_episode_is_reported(model):
    data = mujoco.MjData(model)
    expert = MotorExpert(model, config=replace(DEFAULT_EXPERT, ready_tcp_z_m=2.0))
    result = expert.run(data, Orientation.FLAT, 17)
    assert not result.completed and not result.success and not result.stable
    assert result.phases[-1] == "Initialize"
    assert "unreachable" in result.failure_reasons[0]
    assert result.failure_categories == ["unreachable_ik"]
    stats = summarize([result])[Orientation.FLAT]
    assert stats["success_rate"] == 0
    assert stats["failure_reasons"] == {result.failure_reasons[0]: 1}
