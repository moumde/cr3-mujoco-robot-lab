"""Low-dimensional Oracle Reach on the accepted CR3 workcell."""
import gymnasium as gym
import mujoco
import numpy as np

from cr3_lab.core.model import CONTROL_HZ, TCP
from cr3_lab.tasks.ik import Unreachable
from cr3_lab.tasks.motor_expert import MotorExpert, tool_rotation
from cr3_lab.tasks.motor_scene import Orientation, SpawnRange, load_scene, oracle_pose, reset_scene

TCP_DELTA_M = 0.0015
SUCCESS_DISTANCE_M = 0.008
TRAIN_SEEDS = range(8000, 8100)
TEST_SEEDS = range(9000, 9020)


class MotorReachEnv(gym.Env):
    """Reach an above-object TCP waypoint. Oracle state is explicit in observations."""

    metadata = {"render_modes": []}

    def __init__(self, spawn_range=SpawnRange.LARGE, max_steps=150, *, task="reach"):
        if task != "reach":
            raise NotImplementedError(f"Task {task!r} is reserved for a later stage")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.spawn_range = SpawnRange(spawn_range)
        self.max_steps = max_steps
        self.model = load_scene()
        self.data = mujoco.MjData(self.model)
        self.robot = MotorExpert(self.model)
        self.slide_ids = [self.model.joint(name).qposadr[0]
                          for name in ("left_slide", "right_slide")]
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        lower, upper = self.robot.arm.limits_rad.T
        # Joint and slide constraints are soft. Include their tolerance in the space.
        low = np.r_[lower - 0.1, [-np.inf] * 6, [-np.inf] * 3, [-np.inf] * 3,
                    [-1.] * 3, [-np.inf] * 6, [-0.005] * 2, [-np.inf] * 3]
        high = np.r_[upper + 0.1, [np.inf] * 6, [np.inf] * 3, [np.inf] * 3,
                     [1.] * 3, [np.inf] * 6, [0.042] * 2, [np.inf] * 3]
        self.observation_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32))
        self._needs_reset = True
        self._support = {self.model.geom("table").id, self.model.geom("motor_housing").id}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        orientation = Orientation(options["orientation"]) if "orientation" in options else (
            list(Orientation)[int(self.np_random.integers(len(Orientation)))])
        scene_seed = int(options["scene_seed"]) if "scene_seed" in options else int(
            self.np_random.integers(0, 2**32))
        robot, data = self.robot, self.data
        reset_scene(self.model, data, orientation, scene_seed, spawn_range=self.spawn_range)
        self.tcp_target, self.joint_target = robot.grasp_ready(data)
        for _ in range(round(robot.config.settle_s * CONTROL_HZ)):
            robot.arm.step(data, self.joint_target + data.qfrc_bias[robot.arm.dofs] / robot.gains)
        center, _ = oracle_pose(data)
        self.reach_goal = center + [0, 0, 0.065]
        self.place_goal = np.array([*robot.scene.target_xy_m, robot.scene.rest_z_m])
        self.steps = 0
        self._needs_reset = False
        self.scene_seed = scene_seed
        self.orientation = orientation
        return self._observation(), self._info(False, None, False)

    def _observation(self):
        state = self.robot.arm.state(self.data)
        center, rotation = oracle_pose(self.data)
        return np.r_[state.position_rad, state.velocity_rad_s, self.data.site(TCP).xpos,
                     center, np.clip(rotation[:, 2], -1, 1), self.reach_goal, self.place_goal,
                     self.data.qpos[self.slide_ids], self.tcp_target].astype(np.float32)

    def _distance(self):
        distance = float(np.linalg.norm(self.data.site(TCP).xpos - self.reach_goal))
        return distance if np.isfinite(distance) else float("inf")

    def _info(self, success, reason, ik_rejected):
        return {"is_success": success, "reason": reason, "ik_rejected": ik_rejected,
                "distance_m": self._distance(), "scene_seed": self.scene_seed,
                "orientation": str(self.orientation), "observation_mode": "oracle",
                "steps": self.steps}

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before stepping")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Expected a finite 3D action in [-1, 1]")
        robot, data = self.robot, self.data
        scene = robot.scene
        x, y = scene.initial_xy_m
        previous_distance = self._distance()
        candidate = np.clip(self.tcp_target + TCP_DELTA_M * action,
                            [x - 0.06, y - 0.06, scene.rest_z_m + 0.04],
                            [x + 0.06, y + 0.06, robot.config.ready_tcp_z_m + 0.06])
        try:
            q = robot.ik.solve(candidate, tool_rotation(np.pi / 2), self.joint_target)
        except Unreachable:
            q = None
        ik_rejected = q is None or bool(np.max(abs(q - self.joint_target)) > 0.03)
        if not ik_rejected:
            self.tcp_target, self.joint_target = candidate, q
        robot.gripper.open(data)
        robot.arm.step(data, self.joint_target + data.qfrc_bias[robot.arm.dofs] / robot.gains)
        self.steps += 1
        observation = self._observation()
        reason = None
        if (not np.isfinite(observation).all() or not self.observation_space.contains(observation)
                or any(w.number for w in data.warning)):
            reason = "simulation_fault"
            # Preserve the API's observation contract when ending an invalid rollout.
            observation = np.clip(np.where(np.isfinite(observation), observation, 0), self.observation_space.low,
                                  self.observation_space.high)
        elif any(c.efc_address >= 0 and set(c.geom) != self._support for c in data.contact):
            reason = "unexpected_contact"
        distance = self._distance()
        success = bool(reason is None and distance <= SUCCESS_DISTANCE_M)
        terminated = success or reason is not None
        truncated = not terminated and self.steps >= self.max_steps
        self._needs_reset = terminated or truncated
        progress = np.clip(min(previous_distance, 1.0) - min(distance, 1.0), -0.01, 0.01)
        reward = float(5 * progress - 0.005 - 0.01 * ik_rejected
                       + success - 3 * (reason is not None))
        info = self._info(success, reason or ("success" if success else "timeout" if truncated else None),
                          ik_rejected)
        return observation, reward, terminated, truncated, info
