"""Closed-loop chunk execution on the frozen test manifest."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from bc import Policy
from cr3_lab.core.control import site_pose
from cr3_lab.core.model import TCP, CONTROL_HZ
from cr3_lab.demonstrations.collect import Recorder, scene
from cr3_lab.demonstrations.chunks import PHASES
from cr3_lab.perception.dataset import HEIGHT, WIDTH
from cr3_lab.tasks.motor_expert import EpisodeResult, MotorExpert
from cr3_lab.tasks.motor_scene import PAD_NAMES, MOTOR_GEOMS, Orientation, SpawnRange, reset_scene
from cr3_lab.tasks.vision_score import VisionScore


def state_vector(robot, data):
    state = robot.arm.state(data)
    pose = site_pose(data, TCP)
    aperture = 2 * robot.gripper.travel_m - sum(abs(data.qpos[robot.model.joint(name).qposadr[0]])
        for name in ("left_slide", "right_slide"))
    return np.r_[state.position_rad, state.velocity_rad_s, aperture, pose.position_m,
                 pose.quaternion_wxyz].astype(np.float32)


@torch.no_grad()
def predict(policy, rgb, state):
    images = np.concatenate([np.asarray(Image.fromarray(image).resize(
        (48, 32), Image.Resampling.BILINEAR)).transpose(2, 0, 1) for image in rgb])
    pred = policy(torch.from_numpy(images)[None], torch.from_numpy(state)[None])
    return (pred * policy.as_ + policy.ax).numpy().reshape(5, 7)


def play(model, renderer, policy, case, media):
    robot = MotorExpert(model)
    data = mujoco.MjData(model)
    orientation = Orientation(case["orientation"])
    reset_scene(model, data, orientation, case["seed"], spawn_range=SpawnRange(case["profile"]))
    robot.grasp_ready(data)
    recorder = Recorder(model, renderer)
    result = EpisodeResult(str(orientation), case["seed"])
    vision_score = VisionScore(model, robot.scene, robot.config, result)
    frames, log = [], []
    # Evaluator-only phase clock uses the fixed training-derived reference timeline.
    # It never influences policy inputs or actuator commands.
    ticks = 0
    stopped = False
    pads = {model.geom(name).id for name in PAD_NAMES}
    object_geoms = {model.geom(name).id for name in MOTOR_GEOMS}
    bilateral = False
    open_seen = False
    clipped = 0
    for _ in range(1150 // 5):
        image = recorder.images(data)
        actions = predict(policy, image, state_vector(robot, data))
        for raw in actions:
            phase = next((name for boundary, name in zip(PHASE_START[::-1], PHASES[::-1])
                          if ticks >= boundary), "Initialize")
            if not result.phases or result.phases[-1] != phase:
                result.phases.append(phase)
            if not np.isfinite(raw).all():
                result.failure_reasons.append("nonfinite_policy")
                stopped = True
                break
            clipped += int(raw[6] < 0 or raw[6] > 1 or np.any(
                raw[:6] != np.clip(raw[:6], *robot.arm.limits_rad.T)))
            robot.gripper.command(data, float(np.clip(raw[6], 0, 1)))
            robot.arm.step(data, raw[:6])
            vision_score.sample(data, phase)
            bilateral |= vision_score.bilateral_seen
            open_seen |= phase == "Release" and raw[6] < .2
            ticks += 1
            touched = set()
            for contact in data.contact:
                pair = set(contact.geom)
                if contact.efc_address >= 0 and pair & object_geoms:
                    touched |= pair & pads
            log.append(dict(time_s=data.time, phase=phase, requested=raw.tolist(),
                            ctrl=data.ctrl.tolist(), state=state_vector(robot, data).tolist(),
                            bilateral=bilateral, bilateral_now=touched == pads,
                            lift_m=result.max_lift_m))
            if media and ticks % 5 == 0:
                renderer.update_scene(data, "overview")
                frame = Image.fromarray(renderer.render())
                ImageDraw.Draw(frame).text((8, 8), f"BC | {data.time:.2f}s", fill="white")
                frames.append(frame)
            if result.unexpected_collisions or result.dropped or any(w.number for w in data.warning):
                if any(w.number for w in data.warning):
                    result.failure_categories.append("simulation_warning")
                    result.failure_reasons.append("MuJoCo warning")
                stopped = True
                break
        if stopped:
            break
    result.completed = not stopped
    vision_score.finish(data)
    result.simulation_s = data.time
    if frames:
        frames[0].save(media, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return dict(**asdict(result), bilateral=bilateral, open_seen=open_seen, clipped_ticks=clipped), log


def main(raw, weights, out):
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    policy = Policy()
    policy.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    policy.eval()
    model = scene(False)
    cases = json.loads((raw / "manifest.json").read_text())
    from cr3_lab.demonstrations.chunks import control_rows
    timing = []
    for case in cases:
        if case["split"] != "train" or not case["success"]:
            continue
        rows = control_rows(raw / case["id"])
        timing.append([next(i for i,r in enumerate(rows) if r["phase"] == p) for p in PHASES])
    PHASE_START[:] = np.median(timing, axis=0).astype(int).tolist()
    results = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for case in cases:
            if case["split"] != "test":
                continue
            media = out / (case["id"]+".gif") if case["profile"] == "fixed" else None
            row, log = play(model, renderer, policy, case, media)
            results.append(dict(case=case["id"], profile=case["profile"], expert_success=case["success"], **row))
            (out / (case["id"]+".jsonl")).write_text("".join(json.dumps(x)+"\n" for x in log))
            print(case["id"], row["success"], row["phases"][-1], row["failure_reasons"], flush=True)
    (out / "results.json").write_text(json.dumps(dict(phase_clock_ticks=PHASE_START, rows=results),
                                                 indent=2)+"\n")


PHASE_START = []

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "weights", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    main(a.raw, a.weights, a.output)
