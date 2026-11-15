"""Record synchronous RGB and exact 50 Hz actuator commands from P4a episodes."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image

from cr3_lab.core.control import site_pose, STEPS_PER_COMMAND
from cr3_lab.core.model import CONTROL_HZ, DEFAULT_SOURCE, JOINT_NAMES, TCP, TIMESTEP_S
from cr3_lab.perception.dataset import CAMERAS, HEIGHT, WIDTH
from cr3_lab.tasks.motor_scene import Orientation, SpawnRange, load_scene
from cr3_lab.tasks.p2b2 import VisionEpisode
from cr3_lab.tasks.p4a import LearnedGrasp, policies

FPS = 10
TASK = "Pick up the motor part, orient its protruding end toward the robot, place it on the right target, and retreat."
STATE = ([f"{j}_rad" for j in JOINT_NAMES] + [f"{j}_rad_s" for j in JOINT_NAMES]
         + ["jaw_aperture_m", "tcp_world_x_m", "tcp_world_y_m", "tcp_world_z_m",
            "tcp_world_qw", "tcp_world_qx", "tcp_world_qy", "tcp_world_qz"])
ACTION = [f"{j}_target_rad" for j in JOINT_NAMES] + ["closure_01"]
CAMERA_KEYS = ("inspection_rgb", "localization_rgb")
INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


class Recorder:
    def __init__(self, model, renderer):
        self.model, self.renderer = model, renderer
        self.rows, self.frames = [], []
        self.phase = "Initialize"
        self.initial = np.empty(mujoco.mj_stateSize(model, INTEGRATION))

    def intercept(self, robot):
        step = robot.arm.step

        def recorded(data, target, **kwargs):
            tick = round(data.time / TIMESTEP_S)
            assert abs(data.time - tick * TIMESTEP_S) < 1e-8
            if not self.rows:
                mujoco.mj_getState(self.model, data, self.initial, INTEGRATION)
            assert tick == len(self.rows) * STEPS_PER_COMMAND
            joint = robot.arm.state(data)
            pose = site_pose(data, TCP)
            aperture = 2 * robot.gripper.travel_m - sum(abs(data.qpos[self.model.joint(name).qposadr[0]])
                                  for name in ("left_slide", "right_slide"))
            state = np.r_[joint.position_rad, joint.velocity_rad_s, aperture,
                          pose.position_m, pose.quaternion_wxyz]
            if tick % round(1 / (FPS * TIMESTEP_S)) == 0:
                self.frames.append(self.images(data))
            step(data, target, **kwargs)
            act = np.r_[data.ctrl[robot.arm.actuators],
                        data.ctrl[robot.gripper.actuator] / robot.gripper.travel_m]
            self.rows.append(dict(phase=self.phase, tick=tick, time_s=tick * TIMESTEP_S,
                                  observed_time_s=tick * TIMESTEP_S,
                                  action_end_s=(tick + STEPS_PER_COMMAND) * TIMESTEP_S,
                                  state=state.tolist(), action=act.tolist(),
                                  arm_request_with_bias_rad=np.asarray(target).tolist(),
                                  integration_after=self.integration(data).tolist()))
        return patch.object(robot.arm, "step", recorded)

    def integration(self, data):
        value = np.empty_like(self.initial)
        mujoco.mj_getState(self.model, data, value, INTEGRATION)
        return value

    def images(self, data):
        images = []
        for name in CAMERAS:
            self.renderer.update_scene(data, name)
            images.append(self.renderer.render().copy())
        return np.stack(images)

    def save(self, folder):
        folder.mkdir()
        np.save(folder / "initial.npy", self.initial)
        indices = np.arange(0, len(self.rows), CONTROL_HZ // FPS)
        assert len(indices) == len(self.frames)
        np.savez(folder / "frames.npz", rgb=np.asarray(self.frames), control_index=indices)
        (folder / "control.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in self.rows))


def scene(dim):
    model = load_scene()
    if dim:
        for lighting in (model.light_diffuse, model.light_ambient,
                         model.vis.headlight.diffuse, model.vis.headlight.ambient):
            lighting[:] *= .8
    return model


def collect(out, artifacts, *, configs=None, check_parity=True):
    if not DEFAULT_SOURCE.is_dir():
        raise FileNotFoundError(f"Restore CR3 source at {DEFAULT_SOURCE} before collection")
    out.mkdir(parents=True)
    (out / "schema.json").write_text(json.dumps(dict(state=STATE, action=ACTION, fps=FPS), indent=2) + "\n")
    # Group fixed and dim-fixed at one geometry in train. Hold large/5402 out entirely.
    if configs is None:
        configs = [(SpawnRange.FIXED, 5400, False, "rgb_expert", "train"),
                   (SpawnRange.LARGE, 5401, False, "rgb_expert", "train"),
                   (SpawnRange.LARGE, 5402, False, "rgb_learned_grasp", "validation"),
                   (SpawnRange.FIXED, 5400, True, "rgb_expert", "train")]
    frozen = policies(artifacts) if any(c[3] == "rgb_learned_grasp" for c in configs) else None
    manifest = []
    for profile, seed, dim, mode, split in configs:
        model = scene(dim)
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            vision = VisionEpisode(model, renderer)
            learned = LearnedGrasp(model, *frozen) if mode == "rgb_learned_grasp" else None
            for orientation in Orientation:
                data = mujoco.MjData(model)
                rec = Recorder(model, renderer)
                # Save the estimator input independently of the periodic stream.
                decision = []
                with ExitStack() as stack:
                    stack.enter_context(rec.intercept(vision.robot))
                    if learned:
                        stack.enter_context(rec.intercept(learned.plant.robot))

                    result = vision.run(data, orientation, seed, profile, learned=learned,
                                        perception_observer=lambda images: decision.append((data.time, images)),
                                        command_observer=lambda phase: setattr(rec, "phase", phase))
                # Compare all controls, integration states, and scores with periodic
                # image recording disabled.
                if check_parity:
                    terminal = rec.integration(data)
                    plain_data = mujoco.MjData(model)
                    plain_rec = Recorder(model, renderer)
                    plain_rec.images = lambda _data: None
                    with ExitStack() as stack:
                        stack.enter_context(plain_rec.intercept(vision.robot))
                        if learned:
                            stack.enter_context(plain_rec.intercept(learned.plant.robot))
                        plain = vision.run(plain_data, orientation, seed, profile, learned=learned,
                                           command_observer=lambda phase: setattr(plain_rec, "phase", phase))
                    assert rec.rows == plain_rec.rows
                    np.testing.assert_array_equal(terminal, plain_rec.integration(plain_data))
                    assert {k: v for k, v in result.items() if k != "wall_s"} == {
                        k: v for k, v in plain.items() if k != "wall_s"}
                identifier = f"episode-{len(manifest):03d}"
                folder = out / identifier
                rec.save(folder)
                decision_time, decision_images = decision[0]
                for key, image in zip(CAMERA_KEYS, decision_images, strict=True):
                    Image.fromarray(image).save(folder / f"decision-{key}.png")
                (folder / "oracle-debug.json").write_text(json.dumps({
                    k: result[k] for k in ("initial_xy_m", "position_error_m", "angle_error_deg",
                                            "rgb_xy_error_m", "orientation_correct")}, indent=2) + "\n")
                manifest.append(dict(id=identifier, seed=seed, orientation=str(orientation),
                    mode=mode, profile=str(profile), dim=dim, split=split, task=TASK,
                    control_hz=CONTROL_HZ, camera_fps=FPS, duration_s=result["simulation_s"],
                    frames=len(rec.frames), controls=len(rec.rows), success=result["success"],
                    perception_time_s=round(decision_time / TIMESTEP_S) * TIMESTEP_S,
                    terminal=True, failure_stage=result["phases"][-1] if not result["success"] else None,
                    estimate=result["estimate"], failure_reasons=result["failure_reasons"],
                    failure_categories=result["failure_categories"], render_control_parity=check_parity))
                print(identifier, mode, orientation, result["success"], len(rec.frames), flush=True)
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if profile == SpawnRange.FIXED and not dim:
                assert all(r["success"] for r in manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_SOURCE.parent)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    collect(args.output, args.artifacts)


if __name__ == "__main__":
    main()
