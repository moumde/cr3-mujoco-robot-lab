"""Run RGB-guided manipulation and paired Oracle/RGB evaluation."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cr3_lab.core.model import CONTROL_HZ, TCP
from cr3_lab.perception.camera import calibration
from cr3_lab.perception.dataset import CAMERAS, HEIGHT, WIDTH
from cr3_lab.perception.rgb_estimator import RgbMotorEstimator
from .ik import Unreachable
from .motor_expert import EpisodeFailure, EpisodeResult, MotorExpert
from .motor_scene import Orientation, SpawnRange, load_scene, oracle_pose, reset_scene
from .vision_motion import execute
from .vision_score import VisionScore


class VisionEpisode:
    def __init__(self, model, renderer):
        self.robot = MotorExpert(model)
        self.renderer = renderer

    def run(self, data, orientation, seed, profile, media=None, *, closure_mode="gradual", learned=None, motion_metrics=False,
            perception_observer=None, command_observer=None):
        started = perf_counter()
        robot = self.robot
        result = EpisodeResult(str(orientation), seed)
        score = VisionScore(robot.model, robot.scene, robot.config, result)
        reset_scene(robot.model, data, orientation, seed, spawn_range=profile)
        result.initial_xy_m = oracle_pose(data)[0][:2].tolist()
        result.phases.append("Initialize")
        estimate = None
        frames = []

        details = {}

        def observe(phase):
            if learned is not None and phase == "Learned Grasp & Lift":
                # Report the measured housing-to-tool offset separately from planning.
                center, _ = oracle_pose(data)
                tcp = data.site(TCP)
                details["passive_housing_in_tool_m"] = (
                    tcp.xmat.reshape(3, 3).T @ (center - tcp.xpos)).tolist()
            if phase != result.phases[-1]:
                result.phases.append(phase)
            score.sample(data, phase)
            if any(warning.number for warning in data.warning):
                raise EpisodeFailure("simulation_warning", "MuJoCo warning")
            if media and round(data.time * CONTROL_HZ) % 3 == 0:
                self.renderer.update_scene(data, "overview")
                frame = Image.fromarray(self.renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((8, 8, 310, 33), fill="#202028")
                draw.text((14, 14), f"RGB | {phase} | {data.time:.2f}s", fill="white")
                frames.append(frame)

        def send(phase, joints, closure):
            if command_observer is not None:
                command_observer(phase)
            robot.gripper.command_closure(data, closure, closure_mode)
            robot.arm.step(data, joints + data.qfrc_bias[robot.arm.dofs] / robot.gains)
            observe(phase)

        observation_xy_error = None
        orientation_correct = False
        try:
            ready, joints = robot.grasp_ready(data)
            for _ in range(round(robot.config.settle_s * CONTROL_HZ)):
                send("Initialize", joints, 0)
            cameras = {name: calibration(robot.model, data, name, WIDTH, HEIGHT) for name in CAMERAS}
            images = []
            for name in CAMERAS:
                self.renderer.update_scene(data, name)
                images.append(self.renderer.render().copy())
            estimate = RgbMotorEstimator(cameras).predict(*images)
            if perception_observer is not None:
                perception_observer(images)
            if media:
                sheet = Image.new("RGB", (WIDTH, HEIGHT * 2 + 40), "#202028")
                for index, image in enumerate(images):
                    sheet.paste(Image.fromarray(image), (0, HEIGHT * index))
                label = estimate.orientation or "unknown"
                ImageDraw.Draw(sheet).text((8, HEIGHT * 2 + 8),
                                           f"{label} | XY {estimate.position_xy} | score {estimate.confidence:.2f}\n"
                                           f"{estimate.failure or 'valid'}",
                                           fill="white")
                sheet.save(media / "rgb.png")
            # Evaluation only. These values are never passed to execute or send.
            truth_center, truth_rotation = oracle_pose(data)
            front = truth_rotation[:, 2] @ (data.cam("inspection").xpos - truth_center) > 0
            truth = Orientation.PROTRUSION if front else Orientation.FLAT
            orientation_correct = estimate.orientation == truth
            if estimate.valid:
                observation_xy_error = float(np.linalg.norm(estimate.position_xy - truth_center[:2]))
            if learned is None:
                execute(estimate, robot.scene, robot.config, robot.ik, ready, joints, send)
            else:
                if not estimate.valid:
                    raise EpisodeFailure("perception", estimate.failure)
                learned(data, orientation, seed, ready, joints, estimate, observe, send, details)
            result.completed = True
        except (EpisodeFailure, Unreachable) as exc:
            result.failure_categories.append("unreachable_ik" if isinstance(exc, Unreachable) else exc.category)
            result.failure_reasons.append(str(exc))
        score.finish(data)
        if learned is not None or motion_metrics:
            details["observed_bilateral"] = score.bilateral_seen
            details["observed_carry_hold_ticks"] = score.max_carry_hold_ticks
        result.simulation_s = data.time
        result.wall_s = perf_counter() - started
        if frames:
            frames[0].save(media / "motion.gif", save_all=True, append_images=frames[1:],
                           duration=60, loop=0)
        return {**asdict(result), "estimate": asdict(estimate) if estimate else None,
                **details, "orientation_correct": orientation_correct, "rgb_xy_error_m": observation_xy_error}


def summarize(rows):
    result = {}
    for method in sorted({row["method"] for row in rows}):
        result[method] = {}
        for profile in SpawnRange:
            for orientation in Orientation:
                group = [row for row in rows if (row["method"], row["profile"], row["orientation"])
                         == (method, profile, orientation)]
                if not group:
                    continue
                completed = [row for row in group if row["completed"]]
                stats = {"episodes": len(group), "successes": sum(row["success"] for row in group),
                         "grasps": sum(row["grasped"] for row in group),
                         "failures": dict(Counter(c for row in group for c in row["failure_categories"])),
                         "wall_s": sum(row["wall_s"] for row in group)}
                for key in ("position_error_m", "angle_error_deg"):
                    stats[f"completed_mean_{key}"] = float(np.mean([row[key] for row in completed])) if completed else None
                stats["mean_max_lift_m"] = float(np.mean([row["max_lift_m"] for row in group]))
                if method == "rgb":
                    errors = [row["rgb_xy_error_m"] for row in group if row["rgb_xy_error_m"] is not None]
                    stats["orientation_correct"] = sum(row["orientation_correct"] for row in group)
                    stats["rgb_xy_median_m"] = float(np.median(errors)) if errors else None
                result[method][f"{profile}/{orientation}"] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("paired", "rgb"), default="paired")
    parser.add_argument("--profile", choices=["all", *SpawnRange], default="all")
    parser.add_argument("--orientation", choices=["both", *Orientation], default="both")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media", action="store_true", help="save RGB and GIF for one episode")
    parser.add_argument("--dim", action="store_true", help="independent 20 percent light reduction")
    parser.add_argument("--closure", choices=("gradual", "fast"), default="gradual")
    args = parser.parse_args()
    profiles = list(SpawnRange) if args.profile == "all" else [SpawnRange(args.profile)]
    orientations = list(Orientation) if args.orientation == "both" else [Orientation(args.orientation)]
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    if args.media and (args.mode != "rgb" or len(profiles) * len(orientations) * args.episodes != 1):
        parser.error("media requires a single RGB episode")
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_scene()
    if args.dim:
        model.light_diffuse[:] *= 0.8
        model.light_ambient[:] *= 0.8
        model.vis.headlight.diffuse[:] *= 0.8
        model.vis.headlight.ambient[:] *= 0.8
    rows = []
    with ExitStack() as stack:
        renderer = stack.enter_context(mujoco.Renderer(model, height=HEIGHT, width=WIDTH))
        vision = VisionEpisode(model, renderer)
        data = mujoco.MjData(model)
        oracle = MotorExpert(model)
        for profile in profiles:
            for orientation in orientations:
                for seed in range(args.seed, args.seed + args.episodes):
                    if args.mode == "paired":
                        baseline = asdict(oracle.run(data, orientation, seed, spawn_range=profile, closure_mode=args.closure))
                        rows.append({"method": "oracle", "profile": profile, **baseline})
                    result = vision.run(data, orientation, seed, profile, args.output if args.media else None,
                                        closure_mode=args.closure)
                    rows.append({"method": "rgb", "profile": profile, **result})
    summary = summarize(rows)
    (args.output / "results.json").write_text(json.dumps(
        {"dim": args.dim, "closure": args.closure, "summary": summary, "episodes": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
