from math import cos, sin
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from cr3_lab.core.control import JointPositionController, site_pose
from cr3_lab.core.model import DEFAULT_SOURCE, DESCRIPTION, FLANGE, JOINT_NAMES, TCP, load_model
from cr3_lab.gripper.control import ParallelGripper
from cr3_lab.gripper.demo import TARGET_RAD


@pytest.fixture(scope="module")
def urdf():
    return ET.parse(DEFAULT_SOURCE / DESCRIPTION).getroot()


def numbers(text):
    return np.fromstring(text, sep=" ")


def rx(a):
    c, s = cos(a), sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = cos(a), sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = cos(a), sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def urdf_fk(robot, angles):
    """Independent matrix FK walking parents from the upstream Link6."""
    by_child = {j.find("child").attrib["link"]: j for j in robot.findall("joint")}
    positions = dict(zip(JOINT_NAMES, angles))
    chain = []
    link = "Link6"
    while link != "world":
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").attrib["link"]

    pose = np.eye(4)
    for joint in reversed(chain):
        local = np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            roll, pitch, yaw = numbers(origin.attrib["rpy"])
            local[:3, :3] = rz(yaw) @ ry(pitch) @ rx(roll)
            local[:3, 3] = numbers(origin.attrib["xyz"])
        if joint.attrib["type"] == "revolute":
            np.testing.assert_array_equal(numbers(joint.find("axis").attrib["xyz"]), [0, 0, 1])
            local[:3, :3] = local[:3, :3] @ rz(positions[joint.attrib["name"]])
        pose = pose @ local
    return pose


@pytest.mark.parametrize("gripper", [False, True])
def test_model_contract(urdf, gripper):
    model = load_model(gripper=gripper)
    arm = JointPositionController(model)
    assert (model.nq, model.nv, model.nu) == ((8, 8, 7) if gripper else (6, 6, 6))
    np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
    assert model.opt.timestep == 0.002
    assert arm.qpos.tolist() == arm.dofs.tolist() == arm.actuators.tolist() == list(range(6))
    np.testing.assert_array_equal(model.qpos0[:6], np.zeros(6))
    np.testing.assert_allclose(model.body("base_link").pos, [0, 0, 0])
    np.testing.assert_allclose(model.body("dummy_link").pos, [0, 0, 0.03])
    for name in ["base_link", *(f"Link{i}" for i in range(1, 7))]:
        body = model.body(name)
        source = urdf.find(f"link[@name='{name}']/inertial")
        assert body.mass[0] == pytest.approx(float(source.find("mass").attrib["value"]))
        np.testing.assert_allclose(body.ipos, numbers(source.find("origin").attrib["xyz"]))
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        i = {key: float(value) for key, value in source.find("inertia").attrib.items()}
        tensor = [[i["ixx"], i["ixy"], i["ixz"]],
                  [i["ixy"], i["iyy"], i["iyz"]],
                  [i["ixz"], i["iyz"], i["izz"]]]
        np.testing.assert_allclose(rotation @ np.diag(body.inertia) @ rotation.T,
                                   tensor, rtol=0, atol=1e-9)
    for name in JOINT_NAMES:
        joint = model.joint(name)
        actuator = model.actuator(f"{name}_position")
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        assert joint.limited[0] and actuator.ctrllimited[0]
        limit = urdf.find(f"joint[@name='{name}']/limit").attrib
        expected = [float(limit["lower"]), float(limit["upper"])]
        np.testing.assert_array_equal(joint.range, expected)
        np.testing.assert_array_equal(actuator.ctrlrange, expected)
        assert actuator.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT
        assert actuator.trnid[0] == joint.id
        assert actuator.biastype[0] == mujoco.mjtBias.mjBIAS_AFFINE
        assert actuator.gainprm[0] == -actuator.biasprm[1] > 0
        assert actuator.biasprm[2] < 0


@pytest.mark.parametrize("gripper", [False, True])
def test_fk(urdf, gripper):
    model = load_model(gripper=gripper)
    data = mujoco.MjData(model)
    lower, upper = model.jnt_range[:6].T
    cases = [np.zeros(6), *np.eye(6) * 0.3, *(-np.eye(6) * 0.3), lower, upper,
             *np.random.default_rng(17).uniform(lower, upper, (32, 6))]
    errors = []
    for angles in cases:
        data.qpos[:6] = angles
        mujoco.mj_forward(model, data)
        reference = urdf_fk(urdf, angles)
        for name, offset in ((FLANGE, 0), (TCP, 0.10 if gripper else 0)):
            pose = site_pose(data, name)
            expected = reference @ [0, 0, offset, 1]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, pose.quaternion_wxyz)
            rotation = matrix.reshape(3, 3)
            relative = reference[:3, :3].T @ rotation
            antisymmetric = relative - relative.T
            angle = np.arctan2(np.linalg.norm(antisymmetric) / np.sqrt(8),
                               (np.trace(relative) - 1) / 2)
            errors.append((np.linalg.norm(pose.position_m - expected[:3]), angle))
            assert pose.frame == name and pose.parent_frame == "world"
    position_error, angle_error = np.max(errors, axis=0)
    print(f"{len(cases)} poses (gripper={gripper}): {position_error:.3e} m, {angle_error:.3e} rad")
    assert position_error < 1e-10
    assert angle_error < 1e-10


def test_control_and_gripper():
    model = load_model(gripper=True)
    data = mujoco.MjData(model)
    arm = JointPositionController(model)
    gripper = ParallelGripper(model)
    with pytest.raises(ValueError):
        arm.step(data, [np.nan] * 6)
    with pytest.raises(ValueError):
        gripper.command(data, 1.01)
    arm.step(data, [99] * 6)
    np.testing.assert_array_equal(data.ctrl[arm.actuators], arm.limits_rad[:, 1])
    assert data.time == pytest.approx(0.02)
    assert not np.array_equal(arm.state(data).position_rad, data.ctrl[arm.actuators])

    mujoco.mj_resetData(model, data)
    slides = [model.joint(name).qposadr[0] for name in ("left_slide", "right_slide")]
    for command, expected_m in ((gripper.open, 0), (gripper.close, 0.037), (gripper.open, 0)):
        command(data)
        for _ in range(50):
            arm.step(data, TARGET_RAD)
        np.testing.assert_allclose(data.qpos[slides], expected_m, atol=2e-4, rtol=0)
        np.testing.assert_allclose(arm.state(data).position_rad, TARGET_RAD, atol=0.02, rtol=0)
    assert data.time == pytest.approx(3)
    assert all(warning.number == 0 for warning in data.warning)
