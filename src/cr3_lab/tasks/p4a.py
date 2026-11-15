"""RGB Perception + Privileged-State Learned Grasp + Motion Planning."""
import argparse
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from cr3_lab.core.model import CONTROL_HZ, TCP
from cr3_lab.perception.dataset import HEIGHT, WIDTH
from cr3_lab.rl.behavior_clone import model_for
from cr3_lab.rl.grasp_lift import HOLD_TICKS, MotorGraspLiftEnv
from cr3_lab.rl.residual_budget import BudgetedResidualEnv
from cr3_lab.rl.train_grasp_lift import SCALE
from .motor_expert import EpisodeFailure, signed_yaw, tool_rotation
from .motor_scene import Orientation, SpawnRange, load_scene
from .p2b2 import VisionEpisode
from .vision_motion import execute, grasp_geometry

FIXED_SEED = 19000
RANDOM_SEEDS = range(19001, 19011)


def checked_load(path, expected):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected checkpoint SHA-256: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def policies(root):
    base = checked_load(root / "p3c-6/run/model.pt",
                        "999e77405fb4bfdc1dc43c1ef6b8a4cef48e4cf6924ffdd9c5d918b2dc668617")
    bc = model_for(True)
    bc.load_state_dict(base["weights"])
    selection = json.loads((root / "p3c-10.6/selected.json").read_text())
    weights = checked_load(root / Path(selection["checkpoint"]).relative_to("outputs"),
                           selection["checkpoint_sha256"])
    residual = torch.nn.Sequential(torch.nn.Linear(43, 64), torch.nn.Tanh(),
                                   torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                   torch.nn.Linear(64, 3), torch.nn.Tanh())
    residual.load_state_dict({k.removeprefix("network."): v for k, v in weights.items()
                              if k.startswith("network.")})
    bc.eval()
    residual.eval()

    @torch.no_grad()
    def base_action(obs):
        x = np.clip((obs * SCALE - base["center"].numpy()) / base["spread"].numpy(), -8, 8)
        return bc(torch.from_numpy(x)).numpy()

    @torch.no_grad()
    def residual_action(obs):
        x = (torch.from_numpy(obs) - weights["center"]) / weights["spread"]
        return residual(x).numpy() * selection["normalized_residual_gain"]

    return base_action, residual_action, selection


class LearnedGrasp:
    def __init__(self, model, base_action, residual, cfg):
        self.plant = MotorGraspLiftEnv(model=model)
        self.env = BudgetedResidualEnv(self.plant, base_action, scale=cfg["xyz_scale"],
                                       budget_m=cfg["signed_budget_m"])
        self.residual = residual

    def __call__(self, data, orientation, seed, ready, joints, estimate, observe, send, details):
        plant = self.plant
        robot = plant.robot
        initial, _ = plant.attach_ready_state(data, orientation, seed, ready, joints)
        obs = self.env.begin_grasp(initial)
        start_time = data.time
        while True:
            obs, _, terminated, truncated, info = self.env.step(self.residual(obs))
            observe("Learned Grasp & Lift")
            if terminated or truncated:
                break
        details["learned_grasp"] = info
        if not (info["is_success"] and info["first_bilateral_step"] and info["first_lift_step"]
                and info["first_100mm_step"] and info["hold_steps"] >= HOLD_TICKS):
            raise EpisodeFailure("learned_grasp", info["reason"])

        arm = robot.arm
        site = data.site(TCP)
        state = arm.state(data)
        tcp = site.xpos.copy()
        rotation = site.xmat.reshape(3, 3)
        yaw = np.arctan2(-rotation[0, 0], rotation[1, 0])
        center, axis, nominal_grasp = grasp_geometry(estimate, robot.scene, robot.config)
        nominal_offset = tool_rotation(np.pi / 2).T @ (center - nominal_grasp)
        handoff = dict(time_s=data.time, learned_start_s=start_time,
                       actual_joints_rad=state.position_rad.tolist(),
                       joint_velocity_rad_s=state.velocity_rad_s.tolist(),
                       prior_targets_rad=plant.joint_target.tolist(),
                       prior_ctrl_rad=data.ctrl[arm.actuators].tolist(),
                       tcp_world_m=tcp.tolist(), tcp_yaw_rad=yaw, closure=float(obs[30]),
                       yaw_change_deg=float(np.degrees(signed_yaw(axis, robot.scene.target_normal_world))),
                       rgb_grasp_axis=axis.tolist(), rgb_nominal_grasp_tcp_m=nominal_grasp.tolist(),
                       nominal_housing_in_tool_m=nominal_offset.tolist(),
                       passive_grip_offset_error_m=float(np.linalg.norm(
                           details["passive_housing_in_tool_m"] - nominal_offset)))
        details["handoff"] = handoff
        phase_ticks = details["phase_ticks"] = {}
        transitions = details["motion_transitions"] = []
        previous_phase = None

        def carry(phase, q, closure):
            nonlocal previous_phase
            before = site.xpos.copy()
            previous_ctrl = data.ctrl[arm.actuators].copy()
            previous_closure = float(data.ctrl[robot.gripper.actuator] / robot.gripper.travel_m)
            send(phase, q, closure)
            speed = float(np.linalg.norm(site.xpos - before) * CONTROL_HZ)
            phase_ticks[phase] = phase_ticks.get(phase, 0) + 1
            if phase != previous_phase:
                transitions.append(dict(phase=phase, time_s=data.time, tcp_speed_m_s=speed,
                    ctrl_jump_rad=float(np.max(np.abs(previous_ctrl - data.ctrl[arm.actuators]))),
                    closure_before=previous_closure,
                    closure_after=float(data.ctrl[robot.gripper.actuator] / robot.gripper.travel_m)))
                previous_phase = phase

        execute(estimate, robot.scene, robot.config, robot.ik, tcp, state.position_rad,
                carry, carried=(tcp, yaw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs"))
    parser.add_argument("--mode", choices=("paired", "rgb_expert", "rgb_learned_grasp"), default="paired")
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--media", action="store_true", help="save full fixed-scene GIFs")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_scene()
    data = mujoco.MjData(model)
    learned = LearnedGrasp(model, *policies(args.artifacts)) if args.mode != "rgb_expert" else None
    modes = ("rgb_expert", "rgb_learned_grasp") if args.mode == "paired" else (args.mode,)
    args.output.mkdir(parents=True)
    cases = [(SpawnRange.FIXED, FIXED_SEED)]
    if not args.fixed_only:
        cases += [(SpawnRange.LARGE, seed) for seed in RANDOM_SEEDS]
    (args.output / "manifest.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        vision = VisionEpisode(model, renderer)
        for (profile, seed), orientation in product(cases, Orientation):
            initial = None
            for mode in modes:
                use_learned = learned if mode == "rgb_learned_grasp" else None
                media = None
                if profile == SpawnRange.FIXED and (use_learned is not None or args.media):
                    media = args.output / str(orientation)
                    if len(modes) > 1 and use_learned is None:
                        media = args.output / ("expert-" + str(orientation))
                    media.mkdir()
                row = vision.run(data, orientation, seed, profile, media, learned=use_learned,
                                 motion_metrics=True)
                if initial is not None:
                    assert row["initial_xy_m"] == initial
                initial = row["initial_xy_m"]
                rows.append(dict(method=mode, profile=profile, **row))
                print(mode, profile, seed, orientation, row["success"], row["failure_categories"], flush=True)
                (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
            if (profile == SpawnRange.FIXED and orientation == list(Orientation)[-1]
                    and not all(r["success"] for r in rows)):
                return 1
    return int(not all(row["success"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
