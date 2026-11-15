"""Fixed-rate joint position control and named frame observations."""
from dataclasses import dataclass

import mujoco
import numpy as np

from .model import CONTROL_HZ, JOINT_NAMES, TIMESTEP_S

STEPS_PER_COMMAND = round(1 / (CONTROL_HZ * TIMESTEP_S))


@dataclass(frozen=True)
class JointState:
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray


@dataclass(frozen=True)
class Pose:
    frame: str
    parent_frame: str
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray


class JointPositionController:
    """Write actuator setpoints in rad and advance one 20 ms control interval."""

    def __init__(self, model):
        self.model = model
        joints = [model.joint(name) for name in JOINT_NAMES]
        self.qpos = np.array([joint.qposadr[0] for joint in joints])
        self.dofs = np.array([joint.dofadr[0] for joint in joints])
        self.actuators = np.array([model.actuator(f"{name}_position").id for name in JOINT_NAMES])
        self.limits_rad = model.actuator_ctrlrange[self.actuators].copy()
        if not np.isclose(model.opt.timestep, TIMESTEP_S, rtol=0, atol=1e-12):
            raise ValueError("CR3 controller requires a 0.002 s simulation timestep")

    def state(self, data):
        return JointState(data.qpos[self.qpos], data.qvel[self.dofs])

    def step(self, data, target_rad, *, observe_substep=None):
        target = np.asarray(target_rad, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite joint targets in rad")
        data.ctrl[self.actuators] = np.clip(target, *self.limits_rad.T)
        if observe_substep is None:
            mujoco.mj_step(self.model, data, nstep=STEPS_PER_COMMAND)
            mujoco.mj_forward(self.model, data)
            return
        for _ in range(STEPS_PER_COMMAND):
            mujoco.mj_step(self.model, data)
            mujoco.mj_forward(self.model, data)
            observe_substep(data)


def site_pose(data, name):
    """Read a named site pose in MuJoCo world coordinates."""
    site = data.site(name)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, site.xmat)
    return Pose(name, "world", site.xpos.copy(), quaternion)
