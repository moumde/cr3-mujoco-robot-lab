"""Run API checks, MotorExpert waypoint baselines, and bounded SB3 smoke tests."""
import argparse
import json
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env as check_sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
import torch

from cr3_lab.tasks.motor_scene import Orientation, SpawnRange
from .grasp_lift import MotorGraspLiftEnv
from .reach import TCP_DELTA_M


def expert_actions(env):
    """Use MotorExpert's COM grasp and vertical lift waypoints in the RL action space."""
    cfg = env.robot.config
    grasp = env.grasp_point()
    # XY first at ready height; descend along the same top-down grasp axis.
    approach = env.tcp_target.copy()
    approach[:2] = grasp[:2]
    targets = [(approach, -1, 10), (grasp, -1, 18), (grasp, 1, 33),
               (grasp + [0, 0, cfg.lift_m], 1, 35)]
    command = env.tcp_target.copy()
    for waypoint, closure, dwell in targets:
        while np.max(abs(waypoint - command)) > 1e-6:
            delta = np.clip((waypoint - command) / TCP_DELTA_M, -0.8, 0.8)
            command += TCP_DELTA_M * delta
            yield np.r_[delta, closure]
        for _ in range(dwell):
            yield np.array([0, 0, 0, closure])


def rollout(env, seed, expert, *, on_transition=None):
    observation, _ = env.reset(seed=seed, options={"scene_seed": seed})
    rng = np.random.default_rng(seed)
    actions = iter(expert_actions(env)) if expert else None
    total, rejected = 0.0, 0
    while True:
        action = next(actions, np.array([0, 0, 0, 1])) if expert else rng.uniform(-1, 1, 4)
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(obs).all() or not np.isfinite(reward) or not env.observation_space.contains(obs):
            raise RuntimeError("Invalid rollout output")
        if on_transition is not None:
            on_transition(observation.copy(), np.asarray(action, dtype=np.float32).copy(),
                          reward, obs.copy(), terminated, truncated)
        observation = obs
        total += reward
        rejected += info["ik_rejected"]
        if terminated or truncated:
            return {**info, "reward": total, "ik_rejections": rejected, "initial_xy_m": env.initial_xy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = MotorGraspLiftEnv()
    check_env(env, skip_render_check=True)
    check_sb3(env)
    result = {"gymnasium": True, "sb3": True, "rows": [], "smoke": {}}
    for profile in SpawnRange:
        for orientation in Orientation:
            env = MotorGraspLiftEnv(profile, orientation=orientation)
            for seed in range(10000, 10005):
                for expert in (True, False):
                    row = rollout(env, seed, expert)
                    result["rows"].append({"profile": profile, "method": "expert" if expert else "random", **row})

    for name, algorithm, settings in (
        ("ppo", PPO, {"n_steps": 128, "batch_size": 64, "n_epochs": 1}),
        ("sac", SAC, {"learning_starts": 64, "batch_size": 32, "buffer_size": 1024,
                      "ent_coef": 0.005, "train_freq": 4}),
    ):
        vec = VecCheckNan(DummyVecEnv([MotorGraspLiftEnv]), raise_exception=True)
        model = algorithm("MlpPolicy", vec, device="cpu", seed=11, verbose=1,
                          policy_kwargs={"net_arch": [32, 32]}, **settings)
        model.learn(total_timesteps=512)
        result["smoke"][name] = {"steps": model.num_timesteps, "updates": model._n_updates,
                                 "finite_parameters": all(torch.isfinite(p).all().item()
                                                          for p in model.policy.parameters())}
        vec.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
