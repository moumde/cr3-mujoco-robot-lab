"""Bounded Jacobian IK for the CR3 TCP, using separate kinematics data."""
import mujoco
import numpy as np

from cr3_lab.core.model import TCP


class Unreachable(ValueError):
    pass


class TcpIK:
    def __init__(self, model, arm):
        self.model = model
        self.arm = arm
        self.data = mujoco.MjData(model)
        self.site = model.site(TCP).id
        self.jac_pos = np.empty((3, model.nv))
        self.jac_rot = np.empty_like(self.jac_pos)

    def solve(self, position, rotation, seed):
        lower, upper = self.arm.limits_rad.T
        q = np.clip(seed, lower, upper)
        goal_quat = np.empty(4)
        current_quat = np.empty(4)
        local_error = np.empty(3)
        mujoco.mju_mat2Quat(goal_quat, np.asarray(rotation).ravel())
        for _ in range(120):
            self.data.qpos[self.arm.qpos] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            site = self.data.site(self.site)
            position_error = position - site.xpos
            mujoco.mju_mat2Quat(current_quat, site.xmat)
            mujoco.mju_subQuat(local_error, goal_quat, current_quat)
            rotation_error = site.xmat.reshape(3, 3) @ local_error
            if np.linalg.norm(position_error) < 0.00015 and np.linalg.norm(rotation_error) < 0.001:
                return q
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site)
            jac = np.vstack((self.jac_pos, 0.25 * self.jac_rot))[:, self.arm.dofs]
            error = np.r_[position_error, 0.25 * rotation_error]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.004**2 * np.eye(6), error)
            q = np.clip(q + delta * min(1.0, 0.18 / max(1e-12, np.max(np.abs(delta)))), lower, upper)
        raise Unreachable(f"TCP {np.round(position, 4).tolist()} is unreachable from this IK seed")
