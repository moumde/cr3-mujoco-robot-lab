"""Audit expert redundancy and compare frozen BC against nominal action replay."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import torch

from .behavior_clone import features, model_for
from .expert_replay import verify
from .grasp_lift import MotorGraspLiftEnv
from .train_grasp_lift import SCALE, ScaledGrasp, summarize

LEVELS_MM = (0, 2, 5, 10)
SEEDS = range(14000, 14010)


def redundancy(data):
    """Compare matching timesteps across whole trajectories."""
    slices = [slice(a, b) for a, b in zip(data["offsets"][:-1], data["offsets"][1:])]
    obs, act = [], []
    for a, b in combinations(slices, 2):
        n = min(a.stop - a.start, b.stop - b.start)
        i, j = slice(a.start, a.start + n), slice(b.start, b.start + n)
        obs.extend(np.linalg.norm((data["observation"][i] - data["observation"][j]) * SCALE, axis=1))
        act.extend(np.max(abs(data["action"][i] - data["action"][j]), axis=1))
    obs, act = np.asarray(obs), np.asarray(act)
    return {"aligned_pairs": len(obs), "observation_exact": float(np.mean(obs == 0)),
            "observation_within_0.001": float(np.mean(obs <= 0.001)),
            "observation_max_scaled_l2": float(obs.max()),
            "action_exact": float(np.mean(act == 0)),
            "action_within_0.001": float(np.mean(act <= 0.001)),
            "action_max_abs_delta": float(act.max())}


def scenarios():
    for level in LEVELS_MM:
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0),
                      *product((-1, 1), repeat=2)] if level else [(0, 0)]
        for x, y in directions:
            for seed in SEEDS:
                yield level, (x * level / 1000, y * level / 1000), seed


def run(env, method, predict, seed, offset):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "fixed_pregrasp_offset_m": offset})
    initial = obs.copy()
    reward_sum = 0.
    rejected = 0
    for tick in range(env.unwrapped.max_steps):
        action = predict(obs, tick)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        rejected += info["ik_rejected"]
        if terminated or truncated:
            return {**info, "method": method, "seed": seed, "offset_m": offset,
                    "initial_xy_m": env.unwrapped.initial_xy, "initial_tcp_target_m": (initial[37:40] / 10).tolist(),
                    "reward": reward_sum, "ik_rejections": rejected}
    raise RuntimeError("Episode did not terminate within its horizon")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demos", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with np.load(args.demos, allow_pickle=False) as archive:
        data = dict(archive)
    verify(data, SCALE)
    reference = data["action"][:data["offsets"][1]]

    def replay(_, tick):
        if tick < len(reference):
            return reference[tick]
        # After the recording ends, hold TCP and retain the recorded final closure.
        return np.array([0, 0, 0, reference[-1, 3]], dtype=np.float32)

    policies = {"replay": replay}
    for name, clock in (("full", True), ("without_clock", False)):
        model = model_for(clock)
        model.load_state_dict(torch.load(args.checkpoints / name / "model.pt", weights_only=True))
        model.eval()

        @torch.no_grad()
        def predict(obs, _, model=model, clock=clock):
            return model(torch.from_numpy(features(obs, clock))).numpy()

        policies[name] = predict
    env = ScaledGrasp(MotorGraspLiftEnv(start="pregrasp"))
    rows = []
    for level, offset, seed in scenarios():
        for name, policy in policies.items():
            row = run(env, name, policy, seed, offset)
            rows.append({**row, "level_mm": level})
    env.close()
    grouped = {name: {str(level): summarize([row for row in rows
                                              if row["method"] == name and row["level_mm"] == level])
                      for level in LEVELS_MM} for name in policies}
    report = {"redundancy": redundancy(data), "reference_seed": int(data["seeds"][0]),
              "test_seeds": list(SEEDS), "summary": grouped, "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
    for name, groups in grouped.items():
        print(name, [(level, group["held"], group["episodes"], group["reasons"])
                     for level, group in groups.items()])


if __name__ == "__main__":
    main()
