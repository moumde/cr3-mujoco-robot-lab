"""Replay recorded warm-up actions to audit saturation and reconstruct evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

from conservative_sac import (ConservativeSAC, Gain, BudgetedResidualEnv, TrainPlant,
                              load_policy, SAT_THRESHOLD, SAT_WINDOW, GAIN)


def replay(root):
    model = ConservativeSAC.load(root / "model-211-reconstructed.zip", device="cpu")
    rows = [json.loads(line) for line in (root / "online.jsonl").read_text().splitlines()]
    env = Gain(BudgetedResidualEnv(TrainPlant(), load_policy()))
    buffer = ReplayBuffer(2000, env.observation_space, env.action_space, device="cpu")
    means = []
    try:
        obs, _ = env.reset(seed=11, options=dict(scene_seed=rows[0]["scene_seed"],
                                                  orientation=rows[0]["orientation"]))
        for row in rows:
            action = np.asarray(row["actor_residual"], dtype=np.float32)
            means.append(model.predict(obs, deterministic=True)[0])
            after, reward, term, trunc, info = env.step(action)
            assert info["residual_budget"] == row["residual_budget"]
            buffer.add(obs, after, action, np.array([reward]), np.array([term or trunc]),
                       [{"TimeLimit.truncated": trunc and not term}])
            obs = after
    finally:
        env.close()
    model.replay_buffer = buffer
    model.save_replay_buffer(root / "replay-reconstructed.pkl")
    np.save(root / "replayed_deterministic_actor.npy", means)
    recorded = np.abs([r["actor_residual"] for r in rows])
    audit = dict(replayed=len(rows),
                 deterministic_saturation_last_window=(np.abs(means[-SAT_WINDOW:]) >= SAT_THRESHOLD).mean(0).tolist(),
                 recorded_saturation_last_window=(recorded[-SAT_WINDOW:] >= SAT_THRESHOLD).mean(0).tolist(),
                 executed_saturation=(recorded * GAIN >= SAT_THRESHOLD).mean(0).tolist())
    (root / "stop-audit.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    replay(parser.parse_args().root)
