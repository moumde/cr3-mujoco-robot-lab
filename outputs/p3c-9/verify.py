"""Zero-residual parity and bounded diagnostic controls. No optimizers are run."""
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np
import torch

from residual import ChallengeEnv, ResidualEnv, compose, load_policy, sample_xy
from benchmark import DEV
from cr3_lab.rl.behavior_clone import model_for
from cr3_lab.rl.train_grasp_lift import SCALE
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.position_bc import TEST
from cr3_lab.tasks.motor_scene import Orientation, SpawnRange

ROOT = Path(__file__).resolve().parent
FAILED = (17208, 17211, 17218)
ZERO = np.zeros(4, dtype=np.float32)


def rollout(env, seed, orientation, policy, controller=None):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
    rows = [obs.copy()]
    while True:
        if controller is None:
            action = policy(obs) if not isinstance(env, ResidualEnv) else ZERO
        else:
            action = controller(obs, policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append((obs.copy(), reward, terminated, truncated, info))
        if terminated or truncated:
            return rows


def parity(raw, wrapped, seed, orientation, bc):
    base = rollout(raw, seed, orientation, bc)
    residual = rollout(wrapped, seed, orientation, bc)
    if len(base) != len(residual):
        raise AssertionError("Episode lengths differ")
    np.testing.assert_array_equal(base[0], residual[0])
    for a, b in zip(base[1:], residual[1:]):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        control = b[4]["residual_control"]
        assert a[4] == {k: v for k, v in b[4].items() if k != "residual_control"}
        assert control["bc_action"] == control["final_action"]
    return residual[-1][4]


def probe(kind):
    """Use only the existing observation. Hand-tuned controls for mechanism diagnosis."""
    def action(obs, base):
        relative = obs[12:15]
        desired_xy = np.clip(relative[:2] / 0.004, -0.85, 0.85)
        if kind == "ik":
            # At high TCP Z, replace upward drift with a descent after lateral alignment.
            desired_z = -0.65 if np.linalg.norm(relative[:2]) < 0.007 else min(base[2], 0.)
            target = np.r_[desired_xy, desired_z]
            return np.r_[np.clip(target - base[:3], -1, 1), 0]
        # Align only laterally near the object, keeping the BC vertical trajectory.
        delta = np.clip(desired_xy - base[:2], -1, 1) if obs[12:15].dot(obs[12:15]) < 0.065**2 else [0, 0]
        return np.r_[delta, 0, 0]
    return action



def closure_probe(obs, base):
    # Hold the BC gripper open until the fingertips are down at grasp height.
    action = probe("alignment")(obs, base)
    if (obs[12:15].dot(obs[12:15]) > 0.010**2 and obs[30] < 0.2):
        action[3] = -1.
    return action

def main():
    torch.set_num_threads(1)
    bc = load_policy()
    with np.load("outputs/p3c-6/run/demonstrations.npz") as data:
        observations = data["observation"][::37]
        state = torch.load("outputs/p3c-6/run/model.pt", weights_only=True)
        reference = model_for(True)
        reference.load_state_dict(state["weights"])
        normalized = np.clip((observations * SCALE - state["center"].numpy()) / state["spread"].numpy(), -8, 8)
        with torch.no_grad():
            expected = np.array([reference(torch.from_numpy(obs)).numpy() for obs in normalized])
        np.testing.assert_array_equal(expected, np.array([bc(obs) for obs in observations]))
        composed = compose(expected, ZERO, np.array([0.5, 0.5, 0.5, 0.], dtype=np.float32))
        np.testing.assert_array_equal(expected, composed)
    report = {"frozen_observations": len(observations), "max_action_error": 0., "parity": [], "probes": []}
    for profile, seeds in ((SpawnRange.FIXED, [15000]), (SpawnRange.LARGE, TEST)):
        raw = MotorGraspLiftEnv(profile)
        wrapped = ResidualEnv(MotorGraspLiftEnv(profile), bc)
        for seed, orientation in product(seeds, Orientation):
            info = parity(raw, wrapped, seed, orientation, bc)
            report["parity"].append({"profile": profile, **info})
        raw.close()
        wrapped.close()

    raw, variant = ChallengeEnv(), ChallengeEnv()
    wrapped = ResidualEnv(variant, bc)
    for seed, orientation in product(DEV[35], Orientation):
        raw.xy = variant.xy = sample_xy(0.035, seed)
        info = parity(raw, wrapped, seed, orientation, bc)
        report["parity"].append({"profile": "35mm", **info})
        if seed not in FAILED:
            continue
        mode = "ik" if info["reason"] == "timeout" else "alignment"
        variants = [("zero", 0., None), (mode + "-half", 0.5, probe(mode)), (mode, 1., probe(mode))]
        if mode == "alignment":
            variants.append(("alignment-grip", 1., closure_probe))
        for name, scale, controller in variants:
            grip = name.endswith("-grip")
            env = ResidualEnv(variant, bc, (scale, scale, scale, 1. if grip else 0.), gripper=grip)
            trajectory = rollout(env, seed, orientation, bc, controller)[1:]
            outcomes = [row[4] for row in trajectory]
            counts = Counter(row["residual_control"]["ik_status"] for row in outcomes)
            report["probes"].append({"name": name, "summary": outcomes[-1], "ik": dict(counts)})
            path = ROOT / f"{seed}-{orientation}-{name}.jsonl"
            with path.open("w") as stream:
                for obs, reward, term, trunc, step_info in trajectory:
                    stream.write(json.dumps({"obs": obs.tolist(), "reward": reward, "terminated": term,
                                             "truncated": trunc, **step_info}) + "\n")
    raw.close()
    wrapped.close()
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Parity episodes:", len(report["parity"]))
    for row in report["probes"]:
        r = row["summary"]
        print(r["scene_seed"], r["orientation"], row["name"], r["reason"], row["ik"])


if __name__ == "__main__":
    main()
