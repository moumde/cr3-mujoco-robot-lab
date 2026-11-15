"""Run the P0 motion sequence headlessly, in a viewer, or with GIF capture."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import mujoco
import numpy as np

from cr3_lab.core.control import JointPositionController, site_pose
from cr3_lab.core.model import CONTROL_HZ, TCP, load_model
from .control import ParallelGripper

TARGET_RAD = np.array([0.40, -0.55, 0.65, -0.45, 0.35, 0.25])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--gif", type=Path)
    args = parser.parse_args()

    model = load_model(gripper=True)
    data = mujoco.MjData(model)
    arm = JointPositionController(model)
    gripper = ParallelGripper(model)
    mujoco.mj_forward(model, data)
    frames = []
    with ExitStack() as stack:
        viewer = None
        renderer = None
        if args.viewer:
            from mujoco import viewer as mj_viewer
            viewer = stack.enter_context(mj_viewer.launch_passive(model, data))
        if args.gif:
            from PIL import Image
            renderer = stack.enter_context(mujoco.Renderer(model, height=400, width=600))
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [-0.13, -0.04, 0.35]
            camera.distance, camera.azimuth, camera.elevation = 1.35, 125, -18
        for tick in range(4 * CONTROL_HZ):
            if viewer and not viewer.is_running():
                break
            start = time.monotonic()
            elapsed = tick / CONTROL_HZ
            fraction = np.clip(elapsed - 0.5, 0, 1)
            blend = fraction**2 * (3 - 2 * fraction)
            gripper.command(data, float(2 <= elapsed < 3))
            arm.step(data, blend * TARGET_RAD)
            if tick in (99, 149, 199):
                slides = [float(data.joint(name).qpos[0])
                          for name in ("left_slide", "right_slide")]
                error = np.max(np.abs(arm.state(data).position_rad - TARGET_RAD))
                print(f"t={data.time:.2f} s  max_joint_error={error:.5f} rad  slides_m={slides}")
            if renderer and tick % 2 == 0:
                renderer.update_scene(data, camera)
                frames.append(Image.fromarray(renderer.render()))
            if viewer:
                viewer.sync()
                time.sleep(max(0, 1 / CONTROL_HZ - (time.monotonic() - start)))
    if frames:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"GIF: {args.gif}")
    print(f"TCP world position (m): {site_pose(data, TCP).position_m}")


if __name__ == "__main__":
    main()
