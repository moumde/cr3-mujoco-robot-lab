"""Paired force and motion measurements using the existing pick/place trajectory."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import mujoco
import numpy as np

from cr3_lab.core.model import CONTROL_HZ
from cr3_lab.perception.camera import VisualEstimate
from cr3_lab.tasks.motor_expert import EpisodeFailure, EpisodeResult, MotorExpert
from cr3_lab.tasks.motor_scene import MOTOR_GEOMS, PAD_NAMES, Orientation, SpawnRange, load_scene, oracle_pose, reset_scene
from cr3_lab.tasks.vision_motion import execute
from cr3_lab.tasks.vision_score import VisionScore

RAMP_S = 0.40


def run(robot, data, orientation, seed, ramp_s):
    started = perf_counter()
    model = robot.model
    reset_scene(model, data, orientation, seed, spawn_range=SpawnRange.LARGE)
    ready, joints = robot.grasp_ready(data)
    for _ in range(round(robot.config.settle_s * CONTROL_HZ)):
        robot.arm.step(data, joints + data.qfrc_bias[robot.arm.dofs] / robot.gains)
    estimate = VisualEstimate(tuple(oracle_pose(data)[0][:2]), orientation, 1)
    result = EpisodeResult(orientation, seed, initial_xy_m=list(estimate.position_xy))
    score = VisionScore(model, robot.scene, robot.config, result)
    pads = {model.geom(name).id for name in PAD_NAMES}
    motor = {model.geom(name).id for name in MOTOR_GEOMS}
    slides = [model.joint(name).dofadr[0] for name in ("left_slide", "right_slide")]
    force = np.zeros(6)
    twist = np.zeros(6)
    body = model.body("motor").id
    trace = []
    first_contact = bilateral = command_full = None
    closure = 0.0
    close_start = None
    initial_rotation = None

    def sample(data):
        nonlocal first_contact, bilateral
        now = float(data.time - close_start)
        normal = {}
        resultant = 0.0
        for i, contact in enumerate(data.contact):
            pair = set(map(int, contact.geom))
            if contact.efc_address >= 0 and pair & pads and pair & motor:
                mujoco.mj_contactForce(model, data, i, force)
                pad = next(iter(pair & pads))
                normal[pad] = normal.get(pad, 0.0) + float(force[0])
                resultant += float(np.linalg.norm(force[:3]))
        if normal and first_contact is None:
            first_contact = now
        if len(normal) == 2 and bilateral is None:
            bilateral = now
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, twist, 0)
        rotation = data.body(body).xmat.reshape(3, 3)
        angle = np.degrees(np.arccos(np.clip((np.trace(initial_rotation.T @ rotation) - 1) / 2, -1, 1)))
        trace.append({
            "t_s": now, "command": closure, "finger_speed_m_s": float(np.max(abs(data.qvel[slides]))),
            "pad_normal_sum_n": sum(normal.values()), "pad_force_magnitude_sum_n": resultant,
            "actuator_force_abs_n": float(abs(data.actuator_force[robot.gripper.actuator])),
            "object_speed_m_s": float(np.linalg.norm(twist[3:])),
            "object_angular_speed_rad_s": float(np.linalg.norm(twist[:3])),
            "object_rotation_deg": float(angle),
        })

    def send(phase, q, target):
        nonlocal close_start, initial_rotation, closure, command_full
        if not result.phases or result.phases[-1] != phase:
            result.phases.append(phase)
        if phase == "Grasp":
            if close_start is None:
                close_start = data.time
                initial_rotation = oracle_pose(data)[1]
            closure = min(1.0, closure + 1 / round(ramp_s * CONTROL_HZ)) if ramp_s else 1.0
            if closure >= 1 - 1e-12 and command_full is None:
                command_full = float(data.time - close_start)
        else:
            closure = target
        robot.gripper.command(data, closure)
        robot.arm.step(data, q + data.qfrc_bias[robot.arm.dofs] / robot.gains,
                       observe_substep=sample if phase == "Grasp" else None)
        score.sample(data, phase)

    try:
        execute(estimate, robot.scene, robot.config, robot.ik, ready, joints, send)
        result.completed = True
    except EpisodeFailure as exc:
        result.failure_categories.append(exc.category)
        result.failure_reasons.append(str(exc))
    score.finish(data)
    result.simulation_s = float(data.time)
    result.wall_s = perf_counter() - started
    peak = {key: max(row[key] for row in trace) for key in trace[0]
            if key not in {"t_s", "command"}} if trace else {}
    if first_contact is not None:
        peak["first_20ms_normal_sum_n"] = max(
            row["pad_normal_sum_n"] for row in trace if first_contact <= row["t_s"] <= first_contact + 0.020)
    return {**asdict(result), "ramp_s": ramp_s, "first_contact_s": first_contact,
            "bilateral_contact_s": bilateral, "full_command_s": command_full,
            "peak": peak, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_scene()
    robot = MotorExpert(model)
    data = mujoco.MjData(model)
    rows = [run(robot, data, orientation, seed, ramp)
            for orientation in Orientation for seed in range(7000, 7005) for ramp in (0, RAMP_S)]
    summary = {}
    for label, ramp in (("baseline", 0), ("ramp", RAMP_S)):
        group = [row for row in rows if row["ramp_s"] == ramp]
        timing = {}
        for key in ("first_contact_s", "bilateral_contact_s", "full_command_s"):
            values = [row[key] for row in group if row[key] is not None]
            timing[f"mean_{key}"] = mean(values) if values else None
        summary[label] = {
            "episodes": len(group), "grasps": sum(row["grasped"] for row in group),
            "placements": sum(row["success"] for row in group),
            "mean_simulation_s": mean(row["simulation_s"] for row in group),
            "mean_wall_s": mean(row["wall_s"] for row in group),
            **timing,
            "mean_peaks": {key: mean(row["peak"][key] for row in group if key in row["peak"])
                           for key in group[0]["peak"]},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
