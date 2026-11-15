"""Reproduce the 82-case parity panel, then probe six known failures without training."""
import argparse
from contextlib import nullcontext
import gzip
from itertools import product
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-7"))
from benchmark import ChallengeEnv, load_policy
from cr3_lab.core.model import TCP
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.position_bc import TEST as HISTORICAL_25MM_PANEL
from cr3_lab.rl.reach import TCP_DELTA_M
from cr3_lab.rl.residual_budget import BudgetedResidualEnv, DEFAULT_BUDGET_M
from cr3_lab.tasks.motor_scene import Orientation

FAILED = {17208, 17211, 17218}
ZERO = np.zeros(3, dtype=np.float32)


def write_step(stream, obs, reward, terminated, truncated, info, **extra):
    stream.write(json.dumps(dict(obs=obs.tolist(), reward=reward, terminated=terminated,
                                 truncated=truncated, **info, **extra), allow_nan=False) + "\n")


def panel():
    for profile, seeds in (("fixed", [15000]), ("large", HISTORICAL_25MM_PANEL)):
        for seed, orientation in product(seeds, Orientation):
            yield profile, seed, orientation, None, None
    cases = json.loads((ROOT.parent / "p3c-7/results.json").read_text())["ranges"]["35"]["pairs"]
    assert len(cases) == 40
    assert {r["seed"] for r in cases} == set(range(17200, 17220))
    assert len({(r["seed"], r["orientation"]) for r in cases}) == 40
    for row in cases:
        yield "35mm", row["seed"], row["orientation"], row["spawn_xy_m"], row["bc"]


def plant_for(profile, xy):
    if xy is None:
        return MotorGraspLiftEnv(profile)
    env = ChallengeEnv()
    env.xy = xy
    return env


def parity(case, bc, stream):
    profile, seed, orientation, xy, history = case
    raw = plant_for(profile, xy)
    wrappers = [BudgetedResidualEnv(plant_for(profile, xy), bc, budget_m=budget)
                for budget in (DEFAULT_BUDGET_M, (0, 0, 0))]
    try:
        options = {"scene_seed": seed, "orientation": orientation}
        obs, reset = raw.reset(seed=seed, options=options)
        for env in wrappers:
            augmented, info = env.reset(seed=seed, options=options)
            np.testing.assert_array_equal(augmented, np.r_[obs, ZERO])
            assert info == reset
        while True:
            before = raw.data.site(TCP).xpos.copy()
            action = bc(obs)
            obs, reward, term, trunc, info = raw.step(action)
            measured = raw.data.site(TCP).xpos - before
            for mode, env in zip(("bounded_zero_residual", "zero_budget"), wrappers, strict=True):
                wrapped_obs, other_reward, other_term, other_trunc, other = env.step(ZERO)
                trace = other.pop("residual_budget")
                np.testing.assert_array_equal(wrapped_obs, np.r_[obs, ZERO])
                for actual, expected in ((trace["bc_action"], action), (trace["final_action"], action),
                                         (trace["tcp_target_m"], raw.tcp_target),
                                         (trace["measured_tcp_delta_m"], measured),
                                         (trace["absolute_budget_m"], ZERO)):
                    np.testing.assert_array_equal(actual, expected)
                assert (other_reward, other_term, other_trunc, other) == (reward, term, trunc, info)
                write_step(stream, wrapped_obs, reward, term, trunc, other, profile=profile,
                           mode=mode, residual_budget=trace)
            if term or trunc:
                if history:
                    for key in ("is_success", "steps", "reason", "collision"):
                        assert info[key] == history[key], (case[:3], key)
                return {"profile": profile, **info}
    finally:
        raw.close()
        for env in wrappers:
            env.close()


def feedback(obs, base):
    """Use only the live relative grasp point and contact state in the accepted observation."""
    relative = obs[12:15]
    target = base[:3].copy()
    if obs[34]:
        if obs[27] >= .102:
            target[2] = 0
    else:
        if np.linalg.norm(relative) < .065:
            target[:2] = np.clip(relative[:2] / .004, -.85, .85)
        if np.linalg.norm(relative[:2]) < .007:
            target[2] = np.clip(relative[2] / .004, -.85, .85)
        elif relative[2] < -.035:
            target[2] = min(base[2], 0)
    return np.clip((target - base[:3]) / .5, -1, 1)


def oracle_probe(case):
    _, seed, orientation, _, _ = case
    with gzip.open(ROOT.parent / f"p3c-8/{seed}-{orientation}-oracle.jsonl.gz", "rt") as stream:
        reference = [json.loads(line)["tcp_target_m"] for line in stream]

    def track(obs, base):
        index = min(round(float(obs[36]) * 400), len(reference) - 1)
        desired = (np.asarray(reference[index]) - obs[37:40]) / TCP_DELTA_M
        return np.clip((desired - base[:3]) / .5, -1, 1)

    return track


def diagnose(case, bc, mode, output, render):
    import PIL.Image
    import PIL.ImageDraw

    profile, seed, orientation, xy, _ = case
    plant = plant_for(profile, xy)
    env = BudgetedResidualEnv(plant, bc)
    controller = feedback if mode == "feedback" else oracle_probe(case)
    name = f"{seed}-{orientation}-{mode}"
    frames = []
    renderer_context = mujoco.Renderer(plant.model, 320, 480) if render else nullcontext()
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
    peak = np.zeros(3)
    ik = {}
    signed = np.zeros(3)
    absolute = np.zeros(3)
    try:
        with renderer_context as renderer, gzip.open(output / f"{name}.jsonl.gz", "xt") as stream:
            while True:
                residual = controller(obs, env.base_action())
                obs, reward, term, trunc, info = env.step(residual)
                trace = info["residual_budget"]
                assert np.isfinite(obs).all() and info["reason"] != "simulation_fault"
                np.testing.assert_array_less(np.abs(trace["signed_budget_m"]),
                                             np.asarray(DEFAULT_BUDGET_M) + 1e-12)
                accepted = np.asarray(trace["accepted_residual_target_delta_m"])
                signed += accepted
                absolute += abs(accepted)
                np.testing.assert_allclose(trace["signed_budget_m"], signed, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(trace["absolute_budget_m"], absolute)
                np.testing.assert_allclose(obs[40:], signed / DEFAULT_BUDGET_M, atol=1e-7, rtol=0)
                peak = np.maximum(peak, np.abs(trace["signed_budget_m"]))
                status = trace["ik_status"]
                ik[status] = ik.get(status, 0) + 1
                write_step(stream, obs, reward, term, trunc, info)
                if renderer is not None and (info["steps"] % 8 == 1 or term or trunc):
                    renderer.update_scene(plant.data, "overview")
                    frame = PIL.Image.fromarray(renderer.render())
                    PIL.ImageDraw.Draw(frame).text((8, 8), f"{mode} | {seed} | {info['steps']}",
                                                   fill="white", stroke_width=2, stroke_fill="black")
                    frames.append(frame)
                if term or trunc:
                    break
        if frames:
            frames[0].save(output / f"{name}.gif", save_all=True, append_images=frames[1:],
                           duration=160, loop=0)
        return {"mode": mode, "peak_signed_m": peak.tolist(), "ik": ik, **info}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    bc = load_policy()
    cases = list(panel())
    with gzip.open(args.output / "parity.jsonl.gz", "xt") as stream:
        paired = [parity(case, bc, stream) for case in cases]
    assert len(paired) == 82
    assert sum(r["is_success"] for r in paired if r["profile"] == "35mm") == 34
    assert sum(r["is_success"] for r in paired if r["profile"] == "fixed") == 2
    print("82 parity cases passed (two wrapper variants each)", flush=True)
    probes = []
    for case in cases:
        if case[1] in FAILED:
            for mode in ("feedback", "oracle_reference"):
                probes.append(diagnose(case, bc, mode, args.output, args.render))
    (args.output / "results.json").write_text(json.dumps({"parity": paired, "probes": probes},
                                                         indent=2, allow_nan=False) + "\n")
    print("completed diagnostic probes:", len(probes), "rescued:", sum(r["is_success"] for r in probes))
    # Apply the same feedback on the original success set to distinguish the
    # wrapper's transparent behavior from the diagnostic controller's behavior.
    protected = [diagnose(case, bc, "feedback", args.output, False)
                 for case in cases if case[0] == "fixed" or (case[4] and case[4]["is_success"])]
    (args.output / "feedback-regression.json").write_text(json.dumps(protected, indent=2) + "\n")
    print("nonzero feedback protection:", sum(r["is_success"] for r in protected), "/", len(protected))


if __name__ == "__main__":
    main()
