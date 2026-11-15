"""Read-only diagnosis of the P3c-7 ±35 mm development failures."""
import gzip
from itertools import product
import json
from pathlib import Path
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3c-7"))
from benchmark import ChallengeEnv, CHECKPOINT, load_policy, sample_xy

from cr3_lab.core.model import TCP
from cr3_lab.rl.check_grasp_lift import expert_actions
from cr3_lab.rl.grasp_benchmark import HOLD
from cr3_lab.rl.train_grasp_lift import SCALE
from cr3_lab.tasks.ik import Unreachable
from cr3_lab.tasks.motor_scene import Orientation

ROOT = Path(__file__).resolve().parent
SEEDS = (17208, 17211, 17218)


def contacts(env):
    force = np.zeros(6)
    body = env.data.body("motor")
    rotation = body.xmat.reshape(3, 3)
    rows = []
    for index, contact in enumerate(env.data.contact):
        if contact.efc_address < 0:
            continue
        mujoco.mj_contactForce(env.model, env.data, index, force)
        rows.append({"geoms": [env.model.geom(int(i)).name for i in contact.geom],
                     "world_m": contact.pos.tolist(),
                     "motor_local_m": ((contact.pos - body.xpos) @ rotation).tolist(),
                     "normal_n": float(force[0]), "distance_m": float(contact.dist)})
    return rows


def record(env, seed, orientation, method, policy, center, spread, lower, upper):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
    initial = obs.copy()
    actions = iter(expert_actions(env)) if method == "oracle" else None
    solver = env.robot.ik
    solve = solver.solve
    ik = {}

    def observe_solve(position, rotation, seed):
        ik["candidate_m"] = position.tolist()
        try:
            q = solve(position, rotation, seed)
        except Unreachable as exc:
            ik.update(status="solver_nonconverged", detail=str(exc))
            raise
        delta = q - seed
        ik.update(status="joint_step_limit" if np.max(abs(delta)) > 0.03 else "accepted",
                  candidate_delta_rad=delta.tolist(), candidate_joints_rad=q.tolist())
        return q

    path = ROOT / f"{seed}-{orientation}-{method}.jsonl.gz"
    with gzip.open(path, "xt") as stream, patch.object(solver, "solve", observe_solve):
        while True:
            original = next(actions, HOLD) if actions is not None else policy(obs)
            action = np.asarray(original, dtype=np.float32)
            z = (obs * SCALE - center) / spread
            joint_before = env.robot.arm.state(env.data).position_rad
            command_before = env.tcp_target.copy()
            nominal_before = env.joint_target.copy()
            ik.clear()
            after, reward, terminated, truncated, info = env.step(action)
            assert info["ik_rejected"] == (ik["status"] != "accepted")
            body = env.data.body("motor")
            phase = ("hold" if info["first_100mm_step"] else "lift" if info["first_lift_step"]
                     else "contact" if info["first_bilateral_step"] else "before_bilateral")
            row = {
                "step": info["steps"], "sim_time_s": float(env.data.time), "phase": phase,
                "observation": obs.tolist(), "normalized_before_clip": z.tolist(),
                "normalized": np.clip(z, -8, 8).tolist(),
                "outside_train_features": np.flatnonzero((obs < lower - 1e-6) | (obs > upper + 1e-6)).tolist(),
                "raw_action": list(map(float, original)), "env_action": action.tolist(),
                "applied_tcp_delta_m": (env.tcp_target - command_before).tolist(),
                "tcp_world_m": env.data.site(TCP).xpos.tolist(), "tcp_target_m": env.tcp_target.tolist(),
                "ik": ik, "joint_rad": after[:6].tolist(),
                "actual_joint_delta_rad": (after[:6] - joint_before).tolist(),
                "target_joint_delta_rad": (env.joint_target - nominal_before).tolist(),
                "slides_m": after[28:30].tolist(), "closure": float(after[30]),
                "object_world_m": body.xpos.tolist(), "object_quat_wxyz": body.xquat.tolist(),
                "grasp_distance_m": float(np.linalg.norm(after[12:15])),
                "contacts": contacts(env), "reward": reward, "info": info,
                "terminated": terminated, "truncated": truncated,
            }
            stream.write(json.dumps(row) + "\n")
            obs = after
            if terminated or truncated:
                return initial, info


def main():
    torch.set_num_threads(1)
    policy = load_policy()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    with np.load(CHECKPOINT.parent / "demonstrations.npz") as data:
        train = data["observation"][:data["offsets"][128]]
    center, spread = (checkpoint[k].numpy() for k in ("center", "spread"))
    lower, upper = train.min(0), train.max(0)
    history = json.loads((ROOT.parent / "p3c-7/results.json").read_text())["ranges"]["35"]["pairs"]
    history = {(p["seed"], p["orientation"]): p for p in history}
    env = ChallengeEnv()
    for seed, orientation in product(SEEDS, Orientation):
        env.xy = sample_xy(0.035, seed)
        a, oracle = record(env, seed, orientation, "oracle", policy, center, spread, lower, upper)
        b, bc = record(env, seed, orientation, "bc", policy, center, spread, lower, upper)
        np.testing.assert_array_equal(a, b)
        for name, result in (("oracle", oracle), ("bc", bc)):
            expected = history[seed, orientation][name]
            for key in ("reason", "steps", "first_bilateral_step", "first_lift_step"):
                assert result[key] == expected[key], (seed, orientation, name, key)
        print(seed, orientation, oracle["reason"], oracle["steps"], bc["reason"], bc["steps"])
    env.close()


if __name__ == "__main__":
    main()
