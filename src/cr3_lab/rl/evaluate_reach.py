"""Deterministic Reach evaluation, shared by baselines and learned policies."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from cr3_lab.tasks.motor_scene import Orientation, SpawnRange
from .reach import MotorReachEnv, TCP_DELTA_M, TEST_SEEDS


R1_TEST_SEEDS = range(9500, 9520)


class RelativeReach(gym.ObservationWrapper):
    """Fixed scaling of target error and servo tracking error, in units of 0.1 m."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)

    def observation(self, obs):
        return np.concatenate((obs[21:24] - obs[29:32], obs[29:32] - obs[12:15])) * np.float32(10)


def assess(policy, seeds, profiles=tuple(SpawnRange)):
    rows = []
    for profile in profiles:
        env = RelativeReach(MotorReachEnv(profile))
        for orientation in Orientation:
            for seed in seeds:
                obs, info = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
                reward_sum, rejected = 0.0, 0
                initial = env.unwrapped.data.body("motor").xpos[:2].tolist()
                while True:
                    action = policy(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise FloatingPointError("Nonfinite Reach rollout")
                    reward_sum += reward
                    rejected += info["ik_rejected"]
                    if terminated or truncated:
                        break
                rows.append({**info, "profile": profile, "initial_xy_m": initial,
                             "reward": reward_sum, "ik_rejections": rejected})
        env.close()
    return rows


def summarize(rows):
    def group_stats(group):
        return {"episodes": len(group), "success_rate": float(np.mean([r["is_success"] for r in group])),
                "distance_mm": 1000 * float(np.mean([r["distance_m"] for r in group])),
                "mean_steps": float(np.mean([r["steps"] for r in group])),
                "mean_success_steps": float(np.mean([r["steps"] for r in group if r["is_success"]]))
                if any(r["is_success"] for r in group) else None,
                "mean_reward": float(np.mean([r["reward"] for r in group])),
                "ik_rejection_rate": sum(r["ik_rejections"] for r in group) / sum(r["steps"] for r in group)}

    result = {"all": group_stats(rows)}
    for profile in SpawnRange:
        for orientation in Orientation:
            group = [r for r in rows if r["profile"] == profile and r["orientation"] == orientation]
            if group:
                result[f"{profile}/{orientation}"] = group_stats(group)
    return result


def evaluate(name, output, checkpoint=None, *, seeds=TEST_SEEDS):
    if name == "random":
        rng = np.random.default_rng(1234)
        policy = lambda obs: rng.uniform(-1, 1, 3).astype(np.float32)
    elif name == "ik":
        policy = lambda obs: np.clip(obs[:3] / (10 * TCP_DELTA_M), -1, 1)
    else:
        from stable_baselines3 import PPO, SAC
        model = {"ppo": PPO, "sac": SAC}[name].load(checkpoint, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    rows = assess(policy, seeds)
    result = {"method": name, "checkpoint": str(checkpoint) if checkpoint else None,
              "scene_seeds": list(seeds), "summary": summarize(rows), "episodes": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["all"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("random", "ik", "ppo", "sac"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test-set", choices=("p3b", "r1"), default="p3b")
    args = parser.parse_args()
    if args.method in {"ppo", "sac"} and args.checkpoint is None:
        parser.error("--checkpoint is required for learned policies")
    evaluate(args.method, args.output, args.checkpoint,
             seeds=R1_TEST_SEEDS if args.test_set == "r1" else TEST_SEEDS)


if __name__ == "__main__":
    main()
