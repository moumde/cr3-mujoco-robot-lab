"""Inspect frozen P3b checkpoints and matched Reach actions without training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from cr3_lab.tasks.motor_scene import Orientation, SpawnRange
from .evaluate_reach import RelativeReach, assess, summarize
from .reach import MotorReachEnv, TCP_DELTA_M
from .train_reach import VALIDATION_SEEDS


def training_metrics(path):
    fields = {"total_timesteps", "n_updates", "ent_coef", "ent_coef_loss", "actor_loss", "critic_loss"}
    rows, row = [], {}
    for line in path.read_text().splitlines():
        columns = line.strip().strip("|").split("|")
        if len(columns) == 2:
            key, value = map(str.strip, columns)
            if key in fields:
                row[key] = float(value)
        if line.startswith("---") and "total_timesteps" in row:
            rows.append(row)
            row = {}
    return rows


def trace(policy, profile, model=None):
    env = RelativeReach(MotorReachEnv(profile))
    obs, info = env.reset(seed=8500, options={"scene_seed": 8500, "orientation": Orientation.FLAT})
    rows = []
    for _ in range(env.unwrapped.max_steps):
        action = policy(obs)
        relative = (obs[:3] + obs[3:]) / 10
        previous = info["distance_m"]
        row = {"action": action.tolist(), "relative_xyz_m": relative.tolist()}
        if model is not None and not rows:
            with torch.no_grad():
                features = torch.as_tensor(obs[None, :])
                controls = np.array([action, [0, 0, -1], [0, 0, 1]], dtype=np.float32)
                row["critic_values_policy_down_up"] = torch.cat(
                    model.critic(features.repeat(3, 1), torch.as_tensor(controls)), dim=1).tolist()
        obs, reward, terminated, truncated, info = env.step(action)
        row.update(step=info["steps"], distance_m=info["distance_m"],
                   distance_change_m=info["distance_m"] - previous, reward=reward,
                   ik_rejected=info["ik_rejected"])
        rows.append(row)
        if terminated or truncated:
            break
    env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    models = {f"sac-{seed}": SAC.load(args.baseline / f"formal-sac-{seed}/model.zip", device="cpu")
              for seed in (11, 22, 33)}
    ppo = PPO.load(args.baseline / "formal-ppo-11/model.zip", device="cpu")
    policies = {"ppo": lambda obs: ppo.predict(obs, deterministic=True)[0],
                "ik": lambda obs: np.clip(obs[:3] / (10 * TCP_DELTA_M), -1, 1),
                "down": lambda obs: np.array([0, 0, -1], dtype=np.float32),
                "up": lambda obs: np.array([0, 0, 1], dtype=np.float32)}
    report = {}
    for name, model in models.items():
        torch.manual_seed(1600)
        deterministic = lambda obs, model=model: model.predict(obs, deterministic=True)[0]
        stochastic = lambda obs, model=model: model.predict(obs, deterministic=False)[0]
        policies[name] = deterministic
        log = training_metrics(args.baseline / f"formal-{name}.log")
        report[name] = {
            "updates": model._n_updates, "train_freq": model.train_freq.frequency,
            "gradient_steps": model.gradient_steps, "target_entropy": model.target_entropy,
            "ent_coef_spec": model.ent_coef, "final_alpha": float(model.log_ent_coef.detach().exp()),
            "training_log": log,
            "deterministic": summarize(assess(deterministic, VALIDATION_SEEDS)),
            "stochastic": summarize(assess(stochastic, VALIDATION_SEEDS)),
        }
    traces = {profile: {name: trace(policy, profile, models.get(name))
                        for name, policy in policies.items()} for profile in SpawnRange}
    report["traces"] = traces
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name in models:
        row = report[name]
        print(name, row["updates"], row["final_alpha"],
              {mode: row[mode]["all"] for mode in ("deterministic", "stochastic")})


if __name__ == "__main__":
    main()
