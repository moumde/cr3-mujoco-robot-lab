"""Oracle waypoint expert and contact-based evaluation for motor pick-and-place."""
from dataclasses import dataclass, field
from time import perf_counter

import mujoco
import numpy as np

from cr3_lab.core.control import JointPositionController
from cr3_lab.core.model import CONTROL_HZ, TCP
from cr3_lab.gripper.control import ParallelGripper
from .ik import TcpIK, Unreachable
from .motor_scene import DEFAULT_SCENE, MOTOR_GEOMS, PAD_NAMES, SpawnRange, oracle_pose, reset_scene


@dataclass(frozen=True)
class ExpertConfig:
    ready_tcp_z_m: float = 0.29
    approach_m: float = 0.100
    ready_ik_seed_rad: tuple = (0.48, -0.41, -0.82, -0.35, 1.57, 2.05)
    pregrasp_m: float = 0.045
    grasp_tcp_offset_m: float = -0.018
    lift_m: float = 0.14
    place_clearance_m: float = 0.002
    speed_m_s: float = 0.08
    yaw_speed_rad_s: float = 0.7
    joint_speed_rad_s: float = 1.5
    grasp_confirmation_lift_m: float = 0.020
    min_lift_m: float = 0.10
    hold_distance_m: float = 0.035
    loss_s: float = 0.10
    position_tolerance_m: float = 0.015
    angle_tolerance_deg: float = 10.0
    stable_window_s: float = 0.40
    stable_speed_m_s: float = 0.005
    stable_speed_rad_s: float = 0.10
    settle_s: float = 0.60
    joint_limit_slack_rad: float = 0.002


DEFAULT_EXPERT = ExpertConfig()


@dataclass
class EpisodeResult:
    orientation: str
    seed: int
    initial_xy_m: list[float] = field(default_factory=list)
    planned_grasp_tcp_m: list[float] | None = None
    failure_categories: list[str] = field(default_factory=list)
    wall_s: float = 0.0
    phases: list[str] = field(default_factory=list)
    grasped: bool = False
    max_lift_m: float = 0.0
    dropped: bool = False
    position_error_m: float = 0.0
    angle_error_deg: float = 0.0
    planned_yaw_deg: float | None = None
    measured_yaw_deg: float | None = None
    stable: bool = False
    unexpected_collisions: list[str] = field(default_factory=list)
    completed: bool = False
    success: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    simulation_s: float = 0.0


class EpisodeFailure(RuntimeError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def tool_rotation(yaw):
    """Tool +Z points down. At yaw 0 the fingers close along world Y."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[-s, c, 0], [c, s, 0], [0, 0, -1]])


def signed_yaw(start_axis, end_axis):
    """Signed world-Z rotation between the axes' horizontal projections."""
    return np.arctan2(start_axis[0] * end_axis[1] - start_axis[1] * end_axis[0],
                      start_axis[:2] @ end_axis[:2])


def cartesian_targets(ik, cfg, nominal, start, end, start_yaw, end_yaw):
    start, end = np.asarray(start), np.asarray(end)
    seconds = 1.5 * max(np.linalg.norm(end - start) / cfg.speed_m_s,
                        abs(end_yaw - start_yaw) / cfg.yaw_speed_rad_s)
    steps = max(1, int(np.ceil(seconds * CONTROL_HZ)))
    previous = nominal
    trajectory = []
    for step in range(1, steps + 1):
        u = step / steps
        blend = u * u * (3 - 2 * u)
        q = ik.solve(start + (end - start) * blend,
                          tool_rotation(start_yaw + (end_yaw - start_yaw) * blend),
                          previous)
        if np.max(np.abs(q - previous)) * CONTROL_HZ > cfg.joint_speed_rad_s:
            raise EpisodeFailure("joint_speed_limit", "planned joint speed exceeds limit")
        trajectory.append(q)
        previous = q
    return trajectory


class MotorExpert:
    def __init__(self, model, scene=DEFAULT_SCENE, config=DEFAULT_EXPERT):
        self.model = model
        self.scene = scene
        self.config = config
        self.arm = JointPositionController(model)
        self.gripper = ParallelGripper(model)
        self.ik = TcpIK(model, self.arm)
        self.names = tuple(model.geom(i).name for i in range(model.ngeom))
        self.motor = {model.geom(name).id for name in MOTOR_GEOMS}
        self.pads = {model.geom(name).id for name in PAD_NAMES}
        self.table = model.geom("table").id
        self.gains = model.actuator_gainprm[self.arm.actuators, 0]
        self.free_dof = model.joint("motor_free").dofadr[0]

    def grasp_ready(self, data):
        """Initialize the collision-checked, open-jaw stance during reset."""
        position = np.array([*self.scene.initial_xy_m, self.config.ready_tcp_z_m])
        q = self.ik.solve(position, tool_rotation(np.pi / 2), self.config.ready_ik_seed_rad)
        data.qpos[self.arm.qpos] = q
        self.gripper.open(data)
        mujoco.mj_forward(self.model, data)
        blocking = [contact for contact in data.contact if contact.efc_address >= 0
                    and set(contact.geom) != {self.table, self.model.geom("motor_housing").id}]
        if blocking:
            names = ["/".join(self.names[i] for i in contact.geom) for contact in blocking]
            raise EpisodeFailure("collision", f"grasp_ready contact: {', '.join(names)}")
        target = q + data.qfrc_bias[self.arm.dofs] / self.gains
        lower, upper = self.arm.limits_rad.T
        if np.any((target < lower) | (target > upper)):
            raise EpisodeFailure("joint_limit", "grasp_ready equilibrium target outside limits")
        data.ctrl[self.arm.actuators] = target
        mujoco.mj_forward(self.model, data)
        return position, q

    def grasp_point(self, data):
        """Top-down grasp TCP waypoint centered on the motor's mass."""
        return data.body("motor").xipos.copy() + [0, 0, self.config.grasp_tcp_offset_m]

    def run(self, data, orientation, seed, on_tick=None, *, spawn_range=SpawnRange.FIXED, closure_mode="gradual"):
        started = perf_counter()
        cfg, scene = self.config, self.scene
        result = EpisodeResult(str(orientation), seed)
        reset_scene(self.model, data, orientation, seed, scene, spawn_range=spawn_range)
        result.initial_xy_m = oracle_pose(data)[0][:2].tolist()
        nominal = np.zeros(6)
        closure = 0.0
        loss_ticks = 0
        lift_tracking = False
        stable_ticks = 0

        def phase(name):
            result.phases.append(name)

        def tick(target):
            nonlocal loss_ticks, stable_ticks, nominal
            nominal = target
            self.gripper.command_closure(data, closure, closure_mode)
            # Convert the arm bias force into a position-servo setpoint offset.
            self.arm.step(data, target + data.qfrc_bias[self.arm.dofs] / self.gains)
            position, _ = oracle_pose(data)
            if lift_tracking:
                result.max_lift_m = max(result.max_lift_m, float(position[2] - scene.rest_z_m))

            touched_pads = set()
            supported = False
            collisions = set()
            for contact in data.contact:
                a, b = map(int, contact.geom)
                if contact.efc_address < 0:
                    continue
                pair = {a, b}
                if pair & self.motor:
                    touched_pads.update(pair & self.pads)
                    if self.table in pair:
                        supported = True
                        if result.phases[-1] in {"Reorient", "Transfer"}:
                            collisions.add("motor / table during carry")
                    elif not pair & self.pads or result.phases[-1] in {"Initialize", "Approach", "Pre-Grasp"}:
                        collisions.add(" / ".join(sorted((self.names[a], self.names[b]))))
                else:
                    collisions.add(" / ".join(sorted((self.names[a], self.names[b]))))
            if collisions:
                result.unexpected_collisions.extend(sorted(collisions))
                raise EpisodeFailure("collision", "unexpected contact")

            q = self.arm.state(data).position_rad
            lower, upper = self.arm.limits_rad.T
            if np.any(q < lower - cfg.joint_limit_slack_rad) or np.any(q > upper + cfg.joint_limit_slack_rad):
                raise EpisodeFailure("joint_limit", "measured joint limit violation")
            if any(warning.number for warning in data.warning):
                raise EpisodeFailure("simulation_warning", "MuJoCo warning")

            bilateral = touched_pads == self.pads
            if (lift_tracking and bilateral and not supported
                    and position[2] > scene.rest_z_m + cfg.grasp_confirmation_lift_m):
                result.grasped = True
            if lift_tracking:
                distance = np.linalg.norm(position - data.site(TCP).xpos)
                loss_ticks = 0 if bilateral else loss_ticks + 1
                if distance > cfg.hold_distance_m or loss_ticks >= round(cfg.loss_s * CONTROL_HZ):
                    result.dropped = result.grasped
                    reason = "lost grip" if result.grasped else "grip not maintained during lift"
                    raise EpisodeFailure("lost_grip" if result.grasped else "grasp_failure", reason)

            velocity = data.qvel[self.free_dof:self.free_dof + 6]
            resting = (supported and not touched_pads
                       and np.linalg.norm(velocity[:3]) <= cfg.stable_speed_m_s
                       and np.linalg.norm(velocity[3:]) <= cfg.stable_speed_rad_s)
            stable_ticks = stable_ticks + 1 if resting else 0
            if on_tick is not None and on_tick(data, result.phases[-1]) is False:
                raise EpisodeFailure("interrupted", "viewer closed")
            return bilateral

        def hold(seconds):
            bilateral = False
            for _ in range(round(seconds * CONTROL_HZ)):
                bilateral = tick(nominal)
            return bilateral

        def move_cartesian(start, end, start_yaw, end_yaw):
            for target in cartesian_targets(self.ik, cfg, nominal, start, end, start_yaw, end_yaw):
                tick(target)

        try:
            phase("Initialize")
            ready, nominal = self.grasp_ready(data)
            hold(cfg.settle_s)
            _, rotation = oracle_pose(data)
            axis = rotation[:, 2]
            yaw = np.arctan2(axis[1], axis[0])
            # Choose the symmetric jaw orientation closest to the ready yaw (+pi/2).
            yaw = yaw % np.pi
            goal_axis = scene.target_normal_world
            grasp = self.grasp_point(data)
            result.planned_grasp_tcp_m = grasp.tolist()
            approach = grasp + [0, 0, cfg.approach_m]
            pregrasp = grasp + [0, 0, cfg.pregrasp_m]
            lift = grasp + [0, 0, cfg.lift_m]
            transfer = np.array([*scene.target_xy_m, lift[2]])

            phase("Approach")
            move_cartesian(ready, approach, np.pi / 2, yaw)
            hold(0.35)
            phase("Pre-Grasp")
            move_cartesian(approach, pregrasp, yaw, yaw)
            move_cartesian(pregrasp, grasp, yaw, yaw)
            hold(0.35)
            phase("Grasp")
            closure = 1.0
            if not hold(0.65):
                raise EpisodeFailure("grasp_failure", "bilateral pad contact missing")
            phase("Lift")
            lift_tracking = True
            move_cartesian(grasp, lift, yaw, yaw)
            hold(0.20)
            if not result.grasped or result.max_lift_m < cfg.min_lift_m:
                raise EpisodeFailure("grasp_failure", "minimum lift not reached")
            phase("Reorient")
            lift_axis = oracle_pose(data)[1][:, 2]
            rotation_angle = signed_yaw(lift_axis, goal_axis)
            result.planned_yaw_deg = float(np.degrees(rotation_angle))
            desired_yaw = yaw + rotation_angle
            move_cartesian(lift, lift, yaw, desired_yaw)
            result.measured_yaw_deg = float(np.degrees(
                signed_yaw(lift_axis, oracle_pose(data)[1][:, 2])))
            phase("Transfer")
            move_cartesian(lift, transfer, desired_yaw, desired_yaw)
            hold(0.30)
            phase("Place")
            center, _ = oracle_pose(data)
            place = transfer + (np.array([*scene.target_xy_m,
                                          scene.rest_z_m + cfg.place_clearance_m]) - center)
            move_cartesian(transfer, place, desired_yaw, desired_yaw)
            hold(0.2)
            phase("Release")
            lift_tracking = False
            closure = 0.0
            hold(0.40)
            phase("Retreat")
            move_cartesian(place, place + [0, 0, cfg.lift_m], desired_yaw, desired_yaw)
            hold(cfg.stable_window_s + 0.20)
            result.completed = True
        except (EpisodeFailure, Unreachable) as exc:
            result.failure_categories.append("unreachable_ik" if isinstance(exc, Unreachable) else exc.category)
            result.failure_reasons.append(f"{result.phases[-1]}: {exc}")

        center, rotation = oracle_pose(data)
        result.position_error_m = float(np.linalg.norm(
            center - [*scene.target_xy_m, scene.rest_z_m]))
        cosine = np.clip(rotation[:, 2] @ scene.target_normal_world, -1.0, 1.0)
        result.angle_error_deg = float(np.degrees(np.arccos(cosine)))
        result.stable = "Release" in result.phases and stable_ticks >= round(cfg.stable_window_s * CONTROL_HZ)
        result.simulation_s = data.time
        if result.completed:
            if result.position_error_m > cfg.position_tolerance_m:
                result.failure_categories.append("placement_position")
                result.failure_reasons.append("final position outside tolerance")
            if result.angle_error_deg > cfg.angle_tolerance_deg:
                result.failure_categories.append("placement_orientation")
                result.failure_reasons.append("final orientation outside tolerance")
            if not result.stable:
                result.failure_categories.append("placement_unstable")
                result.failure_reasons.append("final placement unstable")
        result.success = (result.completed and result.grasped and result.max_lift_m >= cfg.min_lift_m
                          and not result.dropped and not result.unexpected_collisions
                          and not result.failure_reasons)
        result.wall_s = perf_counter() - started
        return result
