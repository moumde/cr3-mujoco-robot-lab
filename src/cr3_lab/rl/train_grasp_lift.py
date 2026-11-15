"""Bounded, fixed-workcell PPO/SAC experiments with independent evaluation."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch

from .check_grasp_lift import expert_actions
from .grasp_lift import GAMMA, MotorGraspLiftEnv
from .train_reach import PARAMETERS

TRAIN_SEEDS = range(11000, 11100)
VALIDATION_SEEDS = range(12000, 12004)
TEST_SEEDS = range(13000, 13010)
STARTS = ("pregrasp", "grasp_ready")
# Fixed units bring small geometric distances into the MLP's input scale.
SCALE = np.ones(40, dtype=np.float32)
SCALE[12:18] = 10
SCALE[27:30] = 20
SCALE[37:40] = 10
ALGORITHMS = {"ppo": PPO, "sac": SAC}


class ScaledGrasp(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low * SCALE, env.observation_space.high * SCALE)

    def observation(self, observation):
        return observation * SCALE


class TrainingScenes(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options={"scene_seed": int(self.rng.choice(TRAIN_SEEDS))})


def collect(policy, start, seeds):
    env = ScaledGrasp(MotorGraspLiftEnv(start=start))
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed, options={"scene_seed": seed})
        expert = iter(expert_actions(env.unwrapped)) if policy is None else None
        total = 0.
        rejected = 0
        while True:
            action = (next(expert, [0, 0, 0, 1]) if expert is not None else policy(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise FloatingPointError("Nonfinite Grasp & Lift transition")
            total += reward
            rejected += info["ik_rejected"]
            if terminated or truncated:
                rows.append({**info, "reward": total, "ik_rejections": rejected})
                break
    env.close()
    return rows


def summarize(rows):
    count = len(rows)
    return {
        "episodes": count,
        "bilateral": sum(r["first_bilateral_step"] is not None for r in rows),
        "grasped": sum(r["first_lift_step"] is not None for r in rows),
        "lifted": sum(r["first_100mm_step"] is not None for r in rows),
        "held": sum(r["is_success"] for r in rows),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "mean_steps": sum(r["steps"] for r in rows) / count,
        "mean_reward": sum(r["reward"] for r in rows) / count,
        "mean_max_lift_mm": 1000 * sum(r["max_lift_m"] for r in rows) / count,
        "ik_rejection_rate": sum(r["ik_rejections"] for r in rows) / sum(r["steps"] for r in rows),
    }


class Progress(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.episodes = []

    def record(self):
        policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
        stats = summarize(collect(policy, "pregrasp", VALIDATION_SEEDS))
        with (self.directory / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps({"steps": self.num_timesteps, **stats}) + "\n")

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.episodes.append({"interaction_step": self.num_timesteps,
                                  **{key: info[key] for key in ("first_bilateral_step",
                                     "first_lift_step", "first_100mm_step", "reason", "is_success",
                                     "max_lift_m", "scene_seed")}})
        if self.num_timesteps % 2000 == 0:
            self.record()
        return True


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def train(method, output, seed, budget, expert_data=None):
    output.mkdir(parents=True, exist_ok=False)
    vec = VecCheckNan(DummyVecEnv([lambda: ScaledGrasp(TrainingScenes(
        MotorGraspLiftEnv(start="pregrasp")))]), raise_exception=True)
    settings = dict(PARAMETERS[method])
    if method == "sac":
        settings.update(ent_coef=0.005, learning_starts=500)
    else:
        settings.update(n_steps=400, batch_size=100)
    model = ALGORITHMS[method]("MlpPolicy", vec, device="cpu", seed=seed, verbose=1,
                              gamma=GAMMA, policy_kwargs={"net_arch": [64, 64]}, **settings)
    expert_steps = 0
    if expert_data is not None:
        from .expert_replay import prefill, verify

        with np.load(expert_data, allow_pickle=False) as data:
            verify(data, SCALE)
            expert_steps = prefill(model, data, SCALE)
    write(output / "config.json", {"method": method, "seed": seed, "budget": budget,
          "train_scenes": list(TRAIN_SEEDS), "validation_scenes": list(VALIDATION_SEEDS),
          "gamma": GAMMA, "policy_layers": [64, 64], "start": "pregrasp",
          "scale": SCALE.tolist(), "parameters": settings,
          "expert_data": str(expert_data) if expert_data else None})
    callback = Progress(output)
    started = perf_counter()
    model.learn(total_timesteps=budget, callback=callback)
    elapsed = perf_counter() - started
    callback.record()
    if not all(torch.isfinite(p).all().item() for p in model.policy.parameters()):
        raise FloatingPointError("Nonfinite model parameters")
    model.save(output / "model")
    write(output / "exploration.json", callback.episodes)
    write(output / "training.json", {"steps": model.num_timesteps, "expert_steps": expert_steps,
          "updates": model._n_updates,
          "wall_s_including_periodic_validation": elapsed, "finite_parameters": True})
    vec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=(*ALGORITHMS, "expert", "random"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expert-data", type=Path)
    parser.add_argument("--start", choices=STARTS, action="append", help="evaluation start; defaults to both")
    args = parser.parse_args()
    if args.expert_data and (args.method != "sac" or args.steps is None):
        parser.error("--expert-data requires SAC training")
    torch.set_num_threads(1)
    if args.steps is not None:
        if args.method not in ALGORITHMS or args.steps not in (2000, 20000) or args.checkpoint:
            parser.error("training requires PPO/SAC and a budget of 2000 or 20000 steps")
        train(args.method, args.output, args.seed, args.steps, args.expert_data)
        return
    if args.method in ALGORITHMS:
        if args.checkpoint is None:
            parser.error("evaluation of PPO/SAC requires --checkpoint")
        model = ALGORITHMS[args.method].load(args.checkpoint, device="cpu")
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    elif args.method == "random":
        rng = np.random.default_rng(args.seed)
        policy = lambda obs: rng.uniform(-1, 1, 4)
    else:
        policy = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for start in args.start or STARTS:
        rows = collect(policy, start, TEST_SEEDS)
        result[start] = {"summary": summarize(rows), "episodes": rows}
    write(args.output, result)


if __name__ == "__main__":
    main()
