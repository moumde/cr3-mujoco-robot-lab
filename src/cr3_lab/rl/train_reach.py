"""Train one CPU MLP on a bounded budget with independent validation snapshots."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch

from cr3_lab.tasks.motor_scene import SpawnRange
from .evaluate_reach import RelativeReach, assess, summarize
from .reach import MotorReachEnv, TRAIN_SEEDS

SAC_CONTROLS = {
    "baseline": {},
    "updates": {"gradient_steps": 4},
    "temperature": {"ent_coef": 0.005},
}
VALIDATION_SEEDS = range(8500, 8504)
PARAMETERS = {
    "ppo": {"n_steps": 500, "batch_size": 100, "n_epochs": 5, "learning_rate": 0.0003},
    "sac": {"buffer_size": 100_000, "learning_starts": 1000, "batch_size": 128,
            "train_freq": 4, "gradient_steps": 1, "learning_rate": 0.0003},
}


class TrainingScenes(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        self.unwrapped.spawn_range = list(SpawnRange)[int(self.rng.integers(3))]
        return super().reset(seed=seed, options={"scene_seed": int(self.rng.choice(TRAIN_SEEDS))})


class RecordProgress(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.started = perf_counter()

    def record(self):
        if isinstance(self.model, SAC):
            alpha = (self.model.log_ent_coef.detach().exp() if self.model.log_ent_coef is not None
                     else self.model.ent_coef_tensor)
            diagnostics = {"optimization_updates": self.model._n_updates, "alpha": float(alpha)}
        else:
            diagnostics = {}
        rows = assess(lambda obs: self.model.predict(obs, deterministic=True)[0], VALIDATION_SEEDS)
        stats = summarize(rows)["all"]
        stats.update(steps=self.num_timesteps, elapsed_s=perf_counter() - self.started, **diagnostics)
        with (self.directory / "progress.jsonl").open("a") as log:
            log.write(json.dumps(stats) + "\n")

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        if self.num_timesteps % 5000 == 0:
            self.record()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=tuple(PARAMETERS))
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sac-control", choices=tuple(SAC_CONTROLS), default="baseline")
    args = parser.parse_args()
    if args.method != "sac" and args.sac_control != "baseline":
        parser.error("--sac-control is only available for SAC")
    if args.steps < 500 or args.steps % 500:
        parser.error("--steps must be a positive multiple of 500")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = VecCheckNan(DummyVecEnv([lambda: RelativeReach(TrainingScenes(MotorReachEnv()))]),
                      raise_exception=True)
    algorithm = {"ppo": PPO, "sac": SAC}[args.method]
    settings = {"policy_kwargs": {"net_arch": [64, 64]}, "gamma": 0.99,
                **PARAMETERS[args.method], **SAC_CONTROLS[args.sac_control]}
    model = algorithm("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1, **settings)
    config = {"method": args.method, "training_seed": args.seed, "budget": args.steps,
              "train_scene_seeds": list(TRAIN_SEEDS), "validation_scene_seeds": list(VALIDATION_SEEDS),
              "episode_max_steps": 150, "profiles": list(SpawnRange),
              "observation": "oracle_relative_v1", "sac_control": args.sac_control, "settings": settings}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    callback = RecordProgress(args.output)
    started = perf_counter()
    model.learn(total_timesteps=args.steps, callback=callback)
    elapsed = perf_counter() - started
    callback.num_timesteps = model.num_timesteps
    callback.record()
    model.save(args.output / "model")
    restored = algorithm.load(args.output / "model.zip", device="cpu")
    obs, _ = MotorReachEnv().reset(seed=0)
    feature = RelativeReach.observation(None, obs)
    np.testing.assert_array_equal(restored.predict(feature, deterministic=True)[0],
                                  model.predict(feature, deterministic=True)[0])
    (args.output / "training.json").write_text(json.dumps(
        {"steps": model.num_timesteps, "wall_s_including_validation": elapsed,
         "checkpoint_reload_equal": True, "optimization_updates": model._n_updates}, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
