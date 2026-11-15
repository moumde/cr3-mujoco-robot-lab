"""Run Gymnasium and SB3 API checks and a held-out scripted Reach rollout."""
import argparse
import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env as check_gymnasium
from stable_baselines3.common.env_checker import check_env as check_sb3

from cr3_lab.tasks.motor_scene import SpawnRange
from .reach import MotorReachEnv, TCP_DELTA_M, TEST_SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for profile in SpawnRange:
        env = MotorReachEnv(profile)
        check_gymnasium(env, skip_render_check=True)
        check_sb3(env)
        reached = 0
        for seed in TEST_SEEDS:
            env.reset(seed=seed, options={"scene_seed": seed})
            while True:
                delta = (env.reach_goal - env.tcp_target) / TCP_DELTA_M
                _, _, terminated, truncated, info = env.step(np.clip(delta, -1, 1))
                if terminated or truncated:
                    reached += info["is_success"]
                    break
        results[profile] = {"gymnasium_check": True, "sb3_check": True,
                            "scripted_successes": reached, "episodes": len(TEST_SEEDS)}
        env.close()
    report = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
