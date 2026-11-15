"""Passive simulation observer. Its measurements never drive robot commands."""
import numpy as np

from cr3_lab.core.model import CONTROL_HZ, TCP
from .motor_scene import MOTOR_GEOMS, PAD_NAMES, oracle_pose


class VisionScore:
    def __init__(self, model, scene, config, result):
        self.scene, self.cfg, self.result = scene, config, result
        self.motor = {model.geom(name).id for name in MOTOR_GEOMS}
        self.pads = {model.geom(name).id for name in PAD_NAMES}
        self.table = model.geom("table").id
        self.names = [model.geom(i).name for i in range(model.ngeom)]
        self.dof = model.joint("motor_free").dofadr[0]
        self.stable_ticks = self.loss_ticks = 0
        self.bilateral_seen = False
        self.carry_hold_ticks = self.max_carry_hold_ticks = 0

    def sample(self, data, phase):
        result, cfg = self.result, self.cfg
        position, _ = oracle_pose(data)
        pads, collisions = set(), set()
        supported = False
        for contact in data.contact:
            if contact.efc_address < 0:
                continue
            a, b = map(int, contact.geom)
            pair = {a, b}
            if pair & self.motor:
                pads.update(pair & self.pads)
                if self.table in pair:
                    supported = True
                    if phase in {"Reorient", "Transfer"}:
                        collisions.add("motor / table during carry")
                    continue
                if pair & self.pads and phase not in {"Initialize", "Approach", "Pre-Grasp"}:
                    continue
            collisions.add(" / ".join(sorted((self.names[a], self.names[b]))))
        self.bilateral_seen |= pads == self.pads
        result.unexpected_collisions = sorted(set(result.unexpected_collisions) | collisions)
        if phase in {"Lift", "Reorient", "Transfer", "Place"}:
            height = float(position[2] - self.scene.rest_z_m)
            result.max_lift_m = max(result.max_lift_m, height)
            bilateral = pads == self.pads
            result.grasped |= bool(bilateral and not supported and height > cfg.grasp_confirmation_lift_m)
            self.loss_ticks = 0 if bilateral else self.loss_ticks + 1
            if result.grasped and (self.loss_ticks >= round(cfg.loss_s * CONTROL_HZ)
                                   or np.linalg.norm(position - data.site(TCP).xpos) > cfg.hold_distance_m):
                result.dropped = True
        velocity = data.qvel[self.dof:self.dof + 6]
        # Track the learned handoff's stability criterion on both methods for comparison.
        steady_carry = (phase in {"Lift", "Learned Grasp & Lift"} and not supported
                        and pads == self.pads and position[2] - self.scene.rest_z_m >= cfg.min_lift_m
                        and np.linalg.norm(velocity[:3]) <= .015 and np.linalg.norm(velocity[3:]) <= .15)
        self.carry_hold_ticks = self.carry_hold_ticks + 1 if steady_carry else 0
        self.max_carry_hold_ticks = max(self.max_carry_hold_ticks, self.carry_hold_ticks)
        resting = (supported and not pads and np.linalg.norm(velocity[:3]) <= cfg.stable_speed_m_s
                   and np.linalg.norm(velocity[3:]) <= cfg.stable_speed_rad_s)
        self.stable_ticks = self.stable_ticks + 1 if resting else 0

    def finish(self, data):
        result, cfg = self.result, self.cfg
        center, rotation = oracle_pose(data)
        result.position_error_m = float(np.linalg.norm(center - [*self.scene.target_xy_m, self.scene.rest_z_m]))
        cosine = np.clip(rotation[:, 2] @ self.scene.target_normal_world, -1, 1)
        result.angle_error_deg = float(np.degrees(np.arccos(cosine)))
        result.stable = "Release" in result.phases and self.stable_ticks >= round(cfg.stable_window_s * CONTROL_HZ)
        checks = [
            (result.unexpected_collisions, "collision", "unexpected contact"),
            (result.dropped, "lost_grip", "Oracle observer detected loss during carry"),
            (result.completed and (not result.grasped or result.max_lift_m < cfg.min_lift_m),
             "grasp_failure", "grasp or minimum lift not verified"),
            (result.completed and result.position_error_m > cfg.position_tolerance_m,
             "placement_position", "final position outside tolerance"),
            (result.completed and result.angle_error_deg > cfg.angle_tolerance_deg,
             "placement_orientation", "final orientation outside tolerance"),
            (result.completed and not result.stable, "placement_unstable", "final placement unstable"),
        ]
        for failed, category, reason in checks:
            if failed:
                result.failure_categories.append(category)
                result.failure_reasons.append(reason)
        result.success = result.completed and not result.failure_reasons
