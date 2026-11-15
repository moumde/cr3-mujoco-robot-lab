"""Bounded online Residual SAC smoke, with paired development-only evaluation."""
import argparse
from collections import Counter
import gzip
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-9"))
from residual import ChallengeEnv, ResidualEnv, TrainingChallenge, initialize_sac_actor, load_policy, sample_xy

from cr3_lab.core.model import TCP
from cr3_lab.rl.reach import TCP_DELTA_M
from cr3_lab.rl.train_grasp_lift import summarize
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation

FAILED = (17208, 17211, 17218)
CASES = [(group, seed, orientation) for group, seeds in
         (("failed", FAILED), ("nominal", (15000,)), ("random", (17200,)))
         for seed, orientation in product(seeds, Orientation)]
ZERO = np.zeros(3, dtype=np.float32)


class XYZResidual(gym.ActionWrapper):
    def __init__(self, env, bc, scale):
        super().__init__(ResidualEnv(env, bc, (*scale, 0.)))
        self.xyz_scale = np.asarray(scale, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Expected a normalized XYZ residual")
        return np.r_[action, np.float32(0)]

    def step(self, action):
        plant = self.unwrapped
        before = plant.data.site(TCP).xpos.copy()
        obs, reward, term, trunc, info = super().step(action)
        control = info["residual_control"]
        control["residual_action"] = control["residual_action"][:3]
        control["composition_clipped"] = (abs(np.asarray(control["bc_action"][:3], dtype=np.float32)
                                             + self.xyz_scale * action) > 1).tolist()
        control["measured_tcp_delta_m"] = (plant.data.site(TCP).xpos - before).tolist()
        control["grasp_distance_m"] = float(np.linalg.norm(obs[12:15]))
        return obs, reward, term, trunc, info


class BoundedWarmupSAC(SAC):
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Use the initialized squashed Gaussian during warm-up too. SB3's
        # learning_starts still gates optimizer updates in learn().
        return super()._sample_action(0, action_noise, n_envs)


def create(env, seed, log_std):
    model = BoundedWarmupSAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                            learning_starts=500, buffer_size=10000, batch_size=128,
                            train_freq=4, gradient_steps=1, learning_rate=0.0003,
                            ent_coef="auto_0.005", target_entropy=-3,
                            policy_kwargs={"net_arch": [64, 64], "activation_fn": nn.Tanh})
    initialize_sac_actor(model.actor, log_std)
    return model


def distributions(rows):
    residual = np.abs([r["residual_action"] for r in rows])
    applied = np.abs([r["applied_tcp_delta_m"] for r in rows]) * 1000
    measured = np.abs([r["measured_tcp_delta_m"] for r in rows]) * 1000
    final = np.asarray([r["final_action"] for r in rows])
    raw = np.asarray([r["bc_action"] for r in rows])
    return {"abs_residual_p50_p95_max": np.percentile(residual, [50, 95, 100], axis=0).tolist(),
            "abs_applied_target_mm_p50_p95_max": np.percentile(applied, [50, 95, 100], axis=0).tolist(),
            "abs_measured_tcp_mm_p50_p95_max": np.percentile(measured, [50, 95, 100], axis=0).tolist(),
            "residual_saturation": np.mean(residual >= .95, axis=0).tolist(),
            "final_saturation": np.mean(abs(final) >= .95, axis=0).tolist(),
            "final_clipped_fraction": np.mean([r["composition_clipped"] for r in rows], axis=0).tolist(),
            "mean_bc_action": raw.mean(0).tolist(),
            "ik": dict(Counter(r["ik_status"] for r in rows))}


def evaluate(model, bc, scale, steps, root):
    env = XYZResidual(ChallengeEnv(), bc, scale)
    pairs = []
    # Evaluation sampling must not advance the training Torch RNG.
    with torch.random.fork_rng(), gzip.open(root / f"eval-{steps}.jsonl.gz", "wt") as trace:
        torch.manual_seed(6100)
        for group, seed, orientation in CASES:
            env.unwrapped.xy = (np.asarray(DEFAULT_SCENE.initial_xy_m) if group == "nominal"
                                else sample_xy(.035, seed))
            pair = {"group": group, "seed": seed, "orientation": orientation}
            initial = None
            for method in ("bc", "deterministic", "stochastic"):
                obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
                if initial is None:
                    initial = obs.copy()
                else:
                    np.testing.assert_array_equal(obs, initial)
                rows, total, closure_step = [], 0., None
                while True:
                    action = ZERO if method == "bc" else model.predict(obs, deterministic=method == "deterministic")[0]
                    obs, reward, term, trunc, info = env.step(action)
                    control = info["residual_control"]
                    if closure_step is None and control["applied_closure"] >= .5:
                        closure_step = info["steps"]
                    rows.append(control)
                    total += reward
                    trace.write(json.dumps({"seed": seed, "orientation": orientation, "method": method,
                                            "step": info["steps"], **control}) + "\n")
                    if term or trunc:
                        pair[method] = {**info, "reward": total,
                                        "ik_rejections": sum(r["ik_status"] != "accepted" for r in rows),
                                        "closure_half_step": closure_step,
                                        "min_grasp_distance_m": min(r["grasp_distance_m"] for r in rows),
                                        "distribution": distributions(rows)}
                        break
            pairs.append(pair)
    env.close()
    summary = {group: {method: summarize([p[method] for p in pairs if group == "all" or p["group"] == group])
                       for method in ("bc", "deterministic", "stochastic")}
               for group in ("all", "failed", "nominal", "random")}
    (root / f"eval-{steps}.json").write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=2))
    print(steps, {group: {m: v["held"] for m, v in data.items()} for group, data in summary.items()}, flush=True)


class Record(BaseCallback):
    def __init__(self, root, bc, scale, trace):
        super().__init__()
        self.root, self.bc, self.scale, self.trace = root, bc, scale, trace
        self.window, self.ends = [], []
        self.last = 0

    def checkpoint(self):
        steps = self.num_timesteps
        self.model.save(self.root / f"model-{steps}")
        metrics = {key: float(self.model.logger.name_to_value[key])
                   for key in ("train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss")
                   if key in self.model.logger.name_to_value}
        metrics["alpha"] = float(self.model.log_ent_coef.detach().exp())
        with torch.no_grad(), torch.random.fork_rng():
            obs = torch.as_tensor(self.model._last_obs)
            _, log_prob = self.model.actor.action_log_prob(obs.repeat(128, 1))
            metrics["sampled_entropy_at_current_obs"] = float(-log_prob.mean())
        with (self.root / "training.jsonl").open("a") as stream:
            stream.write(json.dumps({"steps": steps, "updates": self.model._n_updates, "metrics": metrics,
                                     "distribution": distributions(self.window) if self.window else None,
                                     "ended_episodes": self.ends}) + "\n")
        evaluate(self.model, self.bc, self.scale, steps, self.root)
        self.window, self.ends = [], []
        self.last = steps

    def _on_rollout_start(self):
        if self.num_timesteps % 1000 == 0 and self.num_timesteps != self.last:
            self.checkpoint()

    def _on_step(self):
        info = self.locals["infos"][0]
        control = info["residual_control"]
        np.testing.assert_allclose(self.locals["buffer_actions"][0], control["residual_action"], atol=1e-7)
        if not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("Nonfinite reward")
        self.window.append(control)
        self.trace.write(json.dumps({"online_step": self.num_timesteps, **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()}}) + "\n")
        if self.locals["dones"][0]:
            self.ends.append({k: v for k, v in info.items() if k not in {"terminal_observation", "residual_control"}})
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(11, 22), required=True)
    parser.add_argument("--scale", type=float, nargs=3, default=(.5, .5, .5))
    args = parser.parse_args()
    torch.set_num_threads(1)
    subprocess.run(["sha256sum", "-c", "outputs/p3c-7/frozen.sha256"], check=True)
    args.output.mkdir(parents=True, exist_ok=False)
    bc = load_policy()
    env = VecCheckNan(DummyVecEnv([lambda: XYZResidual(TrainingChallenge(), bc, args.scale)]),
                      raise_exception=True)
    model = create(env, args.seed, -3)
    assert model.target_entropy == -3 and model.replay_buffer.size() == 0
    assert all(torch.equal(a, b) for a, b in zip(model.critic.parameters(), model.critic_target.parameters()))
    (args.output / "config.json").write_text(json.dumps(
        {"seed": args.seed, "scale": args.scale, "log_std": -3, "initial_alpha": .005,
         "target_entropy": -3, "budget": 5000, "warmup": 500, "train_freq": 4, "batch": 128}, indent=2))
    with gzip.open(args.output / "online.jsonl.gz", "wt") as trace:
        callback = Record(args.output, bc, args.scale, trace)
        evaluate(model, bc, args.scale, 0, args.output)
        model.save(args.output / "model-0")
        model.learn(5000, callback=callback)
        callback.checkpoint()
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise FloatingPointError("Nonfinite network")
    model.save_replay_buffer(args.output / "replay.pkl")
    env.close()
    subprocess.run(["sha256sum", "-c", "outputs/p3c-7/frozen.sha256"], check=True)


if __name__ == "__main__":
    main()
