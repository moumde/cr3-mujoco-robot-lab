"""Compare scene-specific Oracle planning and a fixed recorded action sequence."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np

from cr3_lab.tasks.motor_scene import Orientation, SpawnRange
from .check_grasp_lift import expert_actions, rollout
from .grasp_lift import MotorGraspLiftEnv
from .train_grasp_lift import summarize

REFERENCE_SEED = 15000
HOLD = (0., 0., 0., 1.)


def record_reference():
    env = MotorGraspLiftEnv()
    actions = []
    result = rollout(env, REFERENCE_SEED, True, on_transition=lambda _, a, *rest: actions.append(a))
    env.close()
    if not result["is_success"]:
        raise RuntimeError(f"Nominal expert failed: {result}")
    return np.asarray(actions), result


def execute(env, seed, orientation, action_source=None, *, policy=None, on_action=None):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
    initial = obs.copy()
    actions = iter(action_source()) if policy is None else None
    reward_sum, rejects = 0., 0
    while True:
        action = next(actions, HOLD) if policy is None else policy(obs)
        if on_action is not None:
            on_action(obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        if not env.observation_space.contains(obs) or not np.isfinite(obs).all() or not np.isfinite(reward):
            raise RuntimeError("Invalid Gymnasium transition")
        reward_sum += reward
        rejects += info["ik_rejected"]
        if terminated or truncated:
            phase = ("holding" if info["first_100mm_step"] else
                     "lifting" if info["first_lift_step"] else
                     "after_bilateral_contact" if info["first_bilateral_step"] else "before_bilateral_contact")
            return initial, {**info, "initial_xy_m": env.initial_xy, "reward": reward_sum,
                             "ik_rejections": rejects, "outcome_phase": phase,
                             "terminated": terminated, "truncated": truncated}


def evaluate(seeds, reference):
    pairs = []
    for profile in SpawnRange:
        env = MotorGraspLiftEnv(profile)
        for orientation, seed in product(Orientation, seeds):
            initial, oracle = execute(env, seed, orientation, lambda: expert_actions(env))
            replay_initial, replay = execute(env, seed, orientation, lambda: reference)
            np.testing.assert_array_equal(initial, replay_initial)
            pairs.append({"profile": profile, "orientation": orientation, "seed": seed,
                          "oracle": oracle, "replay": replay})
        env.close()
    return pairs


def report_groups(pairs):
    groups = {}
    for profile, orientation in product(SpawnRange, Orientation):
        group = [p for p in pairs if p["profile"] == profile and p["orientation"] == orientation]
        key = f"{profile}/{orientation}"
        groups[key] = {}
        for method in ("oracle", "replay"):
            rows = [p[method] for p in group]
            stats = summarize(rows)
            stats["collision_count"] = sum(r["reason"] == "unexpected_contact" for r in rows)
            stats["lost_grip_count"] = sum(r["reason"] == "lost_grip" for r in rows)
            stats["failure_phases"] = dict(Counter(r["outcome_phase"] for r in rows if not r["is_success"]))
            groups[key][method] = stats
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    seeds = range(15100, 15103) if args.split == "dev" else range(15200, 15220)
    reference, result = record_reference()
    pairs = evaluate(seeds, reference)
    groups = report_groups(pairs)
    cases = [p for p in pairs if p["oracle"]["is_success"] and not p["replay"]["is_success"]]
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "reference-actions.npy", reference)
    report = {"split": args.split, "seeds": list(seeds), "reference": result,
              "max_steps": 400, "groups": groups, "oracle_only_successes": cases, "pairs": pairs}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    for key, group in groups.items():
        print(key, {name: (stats["held"], stats["episodes"], stats["reasons"])
                    for name, stats in group.items()})


if __name__ == "__main__":
    main()
