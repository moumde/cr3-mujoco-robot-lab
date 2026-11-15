"""Vision task commands. This module has no simulator data access."""
import numpy as np

from cr3_lab.core.model import CONTROL_HZ
from .ik import Unreachable
from .motor_expert import EpisodeFailure, cartesian_targets, signed_yaw, tool_rotation


def grasp_geometry(estimate, scene, cfg):
    """Return the nominal housing center, directed axis and grasp TCP from RGB."""
    # The Inspection camera is on the +Y side of the entire spawn region.
    sign = 1 if estimate.orientation == "protrusion_facing_camera" else -1
    axis = np.array([0., sign, 0.])
    center = np.array([*estimate.position_xy, scene.rest_z_m])
    motor = scene.motor
    com_offset = (motor.length_m + motor.boss_length_m) / 2
    com_offset *= motor.boss_mass_kg / (motor.mass_kg + motor.boss_mass_kg)
    grasp = center + com_offset * axis + [0, 0, cfg.grasp_tcp_offset_m]
    return center, axis, grasp


def execute(estimate, scene, cfg, ik, ready, joints, send, *, carried=None):
    """Drive the shared CR3 Cartesian planner through a command-only callback."""
    if not estimate.valid:
        raise EpisodeFailure("perception", estimate.failure)

    center, axis, grasp = grasp_geometry(estimate, scene, cfg)
    yaw = np.pi / 2
    target_yaw = yaw + signed_yaw(axis, scene.target_normal_world)
    # Assume the planned housing-to-tool transform remains rigid while carried.
    housing_in_tool = tool_rotation(yaw).T @ (center - grasp)
    target_center = np.array([*scene.target_xy_m, scene.rest_z_m + cfg.place_clearance_m])
    place = target_center - tool_rotation(target_yaw) @ housing_in_tool
    approach = grasp + [0, 0, cfg.approach_m]
    pregrasp = grasp + [0, 0, cfg.pregrasp_m]
    lift = grasp + [0, 0, cfg.lift_m]
    transfer = np.array([*place[:2], lift[2]])

    def hold(phase, closure, seconds):
        for _ in range(round(seconds * CONTROL_HZ)):
            send(phase, joints, closure)

    def move(phase, closure, start, end, a, b):
        nonlocal joints
        try:
            targets = cartesian_targets(ik, cfg, joints, start, end, a, b)
        except Unreachable as exc:
            raise EpisodeFailure("unreachable_ik", f"{phase}: {exc}") from exc
        except EpisodeFailure as exc:
            raise EpisodeFailure(exc.category, f"{phase}: {exc}") from exc
        for joints in targets:
            send(phase, joints, closure)

    if carried is None:
        move("Approach", 0, ready, approach, yaw, yaw)
        hold("Approach", 0, 0.35)
        move("Pre-Grasp", 0, approach, pregrasp, yaw, yaw)
        move("Pre-Grasp", 0, pregrasp, grasp, yaw, yaw)
        hold("Pre-Grasp", 0, 0.35)
        hold("Grasp", 1, 0.65)
        move("Lift", 1, grasp, lift, yaw, yaw)
        hold("Lift", 1, 0.20)
    else:
        lift, yaw = carried
        target_yaw = yaw + signed_yaw(axis, scene.target_normal_world)
        place = target_center - tool_rotation(target_yaw) @ housing_in_tool
        transfer = np.array([*place[:2], lift[2]])
    move("Reorient", 1, lift, lift, yaw, target_yaw)
    move("Transfer", 1, lift, transfer, target_yaw, target_yaw)
    hold("Transfer", 1, 0.30)
    move("Place", 1, transfer, place, target_yaw, target_yaw)
    hold("Place", 1, 0.20)
    hold("Release", 0, 0.40)
    move("Retreat", 0, place, place + [0, 0, cfg.lift_m], target_yaw, target_yaw)
    hold("Retreat", 0, cfg.stable_window_s + 0.20)
