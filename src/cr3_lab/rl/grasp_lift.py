"""Oracle Grasp & Lift using the accepted workcell and physical contacts."""
from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np

from cr3_lab.core.model import CONTROL_HZ, TCP
from cr3_lab.tasks.ik import Unreachable
from cr3_lab.tasks.motor_expert import MotorExpert, tool_rotation
from cr3_lab.tasks.motor_scene import Orientation, SpawnRange, load_scene, oracle_pose, reset_scene
from .reach import TCP_DELTA_M

HOLD_TICKS = 20
LOSS_TICKS = 5
GAMMA = 0.99


def shaped_reward(previous, current, rejected, success, failed):
    if failed:
        return -5.0 - float(previous)
    return float(GAMMA * current - previous - 0.005 - 0.02 * rejected + 3 * success)


class MotorGraspLiftEnv(gym.Env):
    """Start at grasp_ready by default. Pregrasp is a separate curriculum."""

    metadata = {"render_modes": []}

    def __init__(self, spawn_range=SpawnRange.FIXED, *, orientation=Orientation.PROTRUSION,
                 start="grasp_ready", max_steps=400, model=None):
        if start not in {"grasp_ready", "pregrasp"} or max_steps < 1:
            raise ValueError("Expected a supported start and positive horizon")
        self.spawn_range, self.start, self.max_steps = SpawnRange(spawn_range), start, max_steps
        self.default_orientation = Orientation(orientation)
        self.model = load_scene() if model is None else model
        self.data = mujoco.MjData(self.model)
        self.robot = MotorExpert(self.model)
        self.rotation = tool_rotation(np.pi / 2)
        self.slides = [self.model.joint(name).qposadr[0] for name in ("left_slide", "right_slide")]
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        low, high = np.full(40, -np.inf), np.full(40, np.inf)
        low[30:37], high[30:37] = 0, 1
        self.observation_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32))
        self._done = True
        self._twist = np.zeros(6)
        self._force = np.zeros(6)

    def _advance(self, closure):
        robot, data = self.robot, self.data
        robot.gripper.command_closure(data, closure, "gradual")
        robot.arm.step(data, self.joint_target + data.qfrc_bias[robot.arm.dofs] / robot.gains)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.orientation = Orientation(options.get("orientation", self.default_orientation))
        self.scene_seed = int(options["scene_seed"]) if "scene_seed" in options else int(
            self.np_random.integers(2**32))
        robot, data = self.robot, self.data
        scene = robot.scene
        offset = np.zeros(2)
        if "fixed_pregrasp_offset_m" in options:
            offset = np.asarray(options["fixed_pregrasp_offset_m"], dtype=float)
            if (self.start != "pregrasp" or self.spawn_range != SpawnRange.FIXED
                    or offset.shape != (2,) or not np.isfinite(offset).all()
                    or np.any(np.abs(offset) > SpawnRange.SMALL.half_width_m)):
                raise ValueError("Expected a Pre-Grasp fixed-scene XY offset within ±10 mm")
            scene = replace(scene, initial_xy_m=tuple(np.asarray(scene.initial_xy_m) + offset))
        reset_scene(self.model, data, self.orientation, self.scene_seed, scene,
                    spawn_range=self.spawn_range)
        self.tcp_target, self.joint_target = robot.grasp_ready(data)
        if self.start == "pregrasp":
            self.tcp_target = self.grasp_point() + [-offset[0], -offset[1], robot.config.pregrasp_m]
            self.joint_target = robot.ik.solve(self.tcp_target, self.rotation, self.joint_target)
            data.qpos[robot.arm.qpos] = self.joint_target
            mujoco.mj_forward(self.model, data)
        for _ in range(round(robot.config.settle_s * CONTROL_HZ)):
            self._advance(0)
        return self._begin_grasp()

    def attach_ready_state(self, data, orientation, seed, ready, joints):
        """Begin a grasp on the caller's settled grasp_ready state without stepping."""
        if self.start != "grasp_ready":
            raise ValueError("Shared state requires grasp_ready")
        self.data = data
        self.orientation, self.scene_seed = Orientation(orientation), seed
        self.tcp_target, self.joint_target = ready.copy(), joints.copy()
        return self._begin_grasp()

    def _begin_grasp(self):
        data = self.data
        self.steps = self.hold = self.loss = 0
        self.grasped = False
        self.first_bilateral = self.first_lift = self.first_100mm = None
        self.max_lift = self.peak_normal = 0.0
        self._done = False
        self.initial_xy = oracle_pose(data)[0][:2].tolist()
        obs, contact, _ = self._observe()
        if contact:
            raise RuntimeError(f"Unexpected reset contact: {contact}")
        self.potential = self._potential(obs)
        return obs, self._info(False, None, False)

    def grasp_point(self):
        return self.robot.grasp_point(self.data)

    def _observe(self):
        robot, data = self.robot, self.data
        pads, collision, supported, normal = set(), None, False, 0.0
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            a, b = map(int, contact.geom)
            pair = {a, b}
            if pair & robot.motor and robot.table in pair:
                supported = True
            elif pair & robot.motor and pair & robot.pads:
                pads.update(pair & robot.pads)
                mujoco.mj_contactForce(self.model, data, index, self._force)
                normal += self._force[0]
            else:
                collision = f"{robot.names[a]} / {robot.names[b]}"
        center, rotation = oracle_pose(data)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                 self.model.body("motor").id, self._twist, 0)
        state = robot.arm.state(data)
        tcp = data.site(TCP).xpos
        closure = data.ctrl[robot.gripper.actuator] / robot.gripper.travel_m
        obs = np.r_[state.position_rad, state.velocity_rad_s, self.grasp_point() - tcp,
                     center, rotation[:, 2], self._twist[3:], self._twist[:3],
                     center[2] - robot.scene.rest_z_m, data.qpos[self.slides], closure,
                     pads == robot.pads, supported, self.hold / HOLD_TICKS, self.grasped,
                     self.loss / LOSS_TICKS, self.steps / self.max_steps,
                     self.tcp_target].astype(np.float32)
        return obs, collision, float(normal)

    def _potential(self, obs):
        distance = np.linalg.norm(obs[12:15])
        if obs[31] and distance <= 0.035:
            return 1.0 + (0 if obs[32] else 10 * np.clip(obs[27], 0, 0.10)) + 0.5 * obs[33]
        return float(0.5 / (1 + 10 * distance))

    def _info(self, success, reason, rejected):
        return {"is_success": success, "reason": reason, "ik_rejected": rejected,
                "steps": self.steps, "first_bilateral_step": self.first_bilateral,
                "first_lift_step": self.first_lift, "first_100mm_step": self.first_100mm, "hold_steps": self.hold,
                "max_lift_m": self.max_lift, "peak_pad_normal_n": self.peak_normal,
                "scene_seed": self.scene_seed, "orientation": str(self.orientation),
                "observation_mode": "oracle", "start": self.start}

    def tcp_candidates(self, xyz_action):
        """Return the requested and workspace-projected TCP targets in world metres."""
        x, y = self.robot.scene.initial_xy_m
        z = self.robot.scene.rest_z_m
        requested = self.tcp_target + TCP_DELTA_M * xyz_action
        return requested, np.clip(requested, [x - 0.06, y - 0.06, z - 0.04],
                                  [x + 0.06, y + 0.06, z + 0.18])

    def step(self, action):
        if self._done:
            raise gym.error.ResetNeeded("Call reset before stepping")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Expected a finite 4D action in [-1, 1]")
        _, candidate = self.tcp_candidates(action[:3])
        return self._step_target(candidate, action[3])

    def _step_target(self, candidate, gripper_action, *, ik_diagnostics=None):
        """Shared physics step for normal actions and bounded TCP commands."""
        if self._done:
            raise gym.error.ResetNeeded("Call reset before stepping")
        robot = self.robot
        try:
            joints = robot.ik.solve(candidate, self.rotation, self.joint_target)
        except Unreachable:
            joints = None
        rejected = joints is None or bool(np.max(abs(joints - self.joint_target)) > 0.03)
        if ik_diagnostics is not None:
            ik_diagnostics["ik_status"] = (
                "solver_nonconverged" if joints is None else
                "joint_step_limit" if rejected else "accepted")
            ik_diagnostics["ik_joint_candidate_rad"] = None if joints is None else joints.tolist()
        if not rejected:
            self.tcp_target, self.joint_target = candidate, joints
        self._advance(float((gripper_action + 1) / 2))
        self.steps += 1
        obs, collision, normal = self._observe()
        finite = bool(np.isfinite(obs).all())
        reason = None
        if not finite or any(w.number for w in self.data.warning):
            reason = "simulation_fault"
            obs = np.where(np.isfinite(obs), obs, 0)
        elif collision:
            reason = "unexpected_contact"

        height = float(obs[27])
        bilateral, supported = bool(obs[31]), bool(obs[32])
        if reason is None and normal > 0 and not self.grasped and np.linalg.norm(obs[12:15]) > 0.035:
            reason = "unexpected_contact"
            collision = "pad / motor outside grasp region"
        if reason is None:
            self.max_lift = max(self.max_lift, height)
            self.peak_normal = max(self.peak_normal, normal)
            if bilateral and self.first_bilateral is None:
                self.first_bilateral = self.steps
            carrying = bilateral and not supported and height >= robot.config.grasp_confirmation_lift_m
            if carrying and not self.grasped:
                self.first_lift = self.steps
                self.grasped = True
            if carrying and height >= robot.config.min_lift_m and self.first_100mm is None:
                self.first_100mm = self.steps
            self.loss = min(LOSS_TICKS, self.loss + 1) if self.grasped and not carrying else 0
            if self.loss >= LOSS_TICKS:
                reason = "lost_grip"
            still = np.linalg.norm(obs[21:24]) <= 0.015 and np.linalg.norm(obs[24:27]) <= 0.15
            self.hold = self.hold + 1 if carrying and height >= robot.config.min_lift_m and still else 0
        obs[33:36] = [self.hold / HOLD_TICKS, self.grasped, self.loss / LOSS_TICKS]
        success = reason is None and self.hold >= HOLD_TICKS
        terminated = bool(success or reason)
        truncated = not terminated and self.steps >= self.max_steps
        self._done = terminated or truncated
        potential = self._potential(obs)
        reward = shaped_reward(self.potential, potential, rejected, success, reason is not None)
        self.potential = potential
        info = self._info(success, reason or ("success" if success else "timeout" if truncated else None), rejected)
        info["collision"] = collision
        return obs, reward, terminated, truncated, info
