"""Run and evaluate Oracle motor pick-and-place."""
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco

from cr3_lab.core.model import CONTROL_HZ
from .motor_expert import MotorExpert
from .motor_scene import Orientation, SpawnRange, load_scene, reset_scene

CAPTURE_INTERVAL = 2


def evaluate(model, orientations, episodes, seed, on_tick=None, *, spawn_range=SpawnRange.FIXED, closure_mode="gradual"):
    data = mujoco.MjData(model)
    expert = MotorExpert(model)
    return [expert.run(data, orientation, seed + index, on_tick, spawn_range=spawn_range, closure_mode=closure_mode)
            for orientation in orientations for index in range(episodes)]


def summarize(results):
    summary = {}
    for orientation in Orientation:
        group = [result for result in results if result.orientation == orientation]
        if not group:
            continue
        count = sum(result.success for result in group)
        summary[str(orientation)] = {
            "episodes": len(group),
            "successes": count,
            "success_rate": count / len(group),
            "distinct_initial_xy": len({tuple(result.initial_xy_m) for result in group}),
            "wall_s": sum(result.wall_s for result in group),
            "failure_categories": dict(Counter(category for result in group
                                               for category in result.failure_categories)),
            "failure_reasons": dict(Counter(reason for result in group
                                            for reason in result.failure_reasons)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orientation", choices=["both", *Orientation], default="both")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per orientation")
    parser.add_argument("--seed", type=int, default=17, help="first reset seed")
    parser.add_argument("--spawn-range", type=SpawnRange, choices=list(SpawnRange),
                        default=SpawnRange.FIXED)
    parser.add_argument("--json", type=Path)
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--viewer", action="store_true")
    media.add_argument("--gif", type=Path)
    media.add_argument("--inspection-png", type=Path, help="render the initial end-face view")
    parser.add_argument("--closure", choices=("gradual", "fast"), default="gradual")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be positive and seed nonnegative")
    orientations = list(Orientation) if args.orientation == "both" else [Orientation(args.orientation)]
    if (args.viewer or args.gif or args.inspection_png) and (args.episodes != 1 or len(orientations) != 1):
        parser.error("select one orientation and --episodes 1 for a demonstration")

    model = load_scene()
    if args.inspection_png:
        from PIL import Image
        data = mujoco.MjData(model)
        reset_scene(model, data, orientations[0], args.seed, spawn_range=args.spawn_range)
        MotorExpert(model).grasp_ready(data)
        with mujoco.Renderer(model, height=400, width=600) as renderer:
            renderer.update_scene(data, "inspection")
            args.inspection_png.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(renderer.render()).save(args.inspection_png)
        return 0

    with ExitStack() as stack:
        if args.viewer or args.gif:
            data = mujoco.MjData(model)
            expert = MotorExpert(model)
            if args.viewer:
                from mujoco import viewer
                # Keep GUI interactions out of the expert's physics state.
                display_data = mujoco.MjData(model)
                reset_scene(model, display_data, orientations[0], args.seed, spawn_range=args.spawn_range)
                expert.grasp_ready(display_data)
                handle = viewer.launch_passive(model, display_data)

                def close_viewer():
                    handle.close()
                    # The passive render loop releases its GLFW resources asynchronously.
                    time.sleep(0.25)

                stack.callback(close_viewer)
                with handle.lock():
                    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    handle.cam.fixedcamid = model.camera("overview").id

                def on_tick(data, _phase):
                    if not handle.is_running():
                        return False
                    with handle.lock():
                        mujoco.mj_copyData(display_data, model, data)
                    handle.sync()
                    time.sleep(1 / CONTROL_HZ)
            else:
                from PIL import Image, ImageDraw
                renderer = stack.enter_context(mujoco.Renderer(model, height=400, width=600))
                frames = []

                def on_tick(data, phase):
                    tick = round(data.time * CONTROL_HZ)
                    if tick % CAPTURE_INTERVAL == 0:
                        renderer.update_scene(data, "overview")
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((8, 8, 246, 33), fill=(20, 22, 27))
                        draw.text((14, 14), f"{data.time:05.2f} s   {phase}", fill="white")
                        frames.append(frame)

            results = [expert.run(data, orientations[0], args.seed, on_tick, spawn_range=args.spawn_range, closure_mode=args.closure)]
            if args.gif and frames:
                args.gif.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                               duration=1000 * CAPTURE_INTERVAL // CONTROL_HZ, loop=0)
        else:
            results = evaluate(model, orientations, args.episodes, args.seed, spawn_range=args.spawn_range, closure_mode=args.closure)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"spawn_range": args.spawn_range, "half_width_m": args.spawn_range.half_width_m,
                                         "summary": summary, "episodes": [asdict(r) for r in results]},
                                        indent=2) + "\n")
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
