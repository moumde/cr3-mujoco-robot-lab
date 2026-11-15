"""Check a SAC actor weight mapping without optimization or rollouts."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.sac.policies import SACPolicy

from cr3_lab.rl.behavior_clone import model_for
from cr3_lab.rl.train_grasp_lift import SCALE


def check():
    with np.load("outputs/p3c-6/run/demonstrations.npz") as data:
        obs = data["observation"][::37].copy()
    state = torch.load("outputs/p3c-6/run/model.pt", weights_only=True)
    bc = model_for(True)
    bc.load_state_dict(state["weights"])
    actor = SACPolicy(gym.spaces.Box(-8, 8, (40,), dtype=np.float32),
                      gym.spaces.Box(-1, 1, (4,), dtype=np.float32),
                      lambda _: 0., net_arch=[64, 64], activation_fn=nn.Tanh).actor
    actor.latent_pi.load_state_dict(bc[:4].state_dict())
    actor.mu.load_state_dict(bc[4].state_dict())
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-3.)
        x = torch.clamp((torch.from_numpy(obs * SCALE) - state["center"]) / state["spread"], -8, 8)
        expected = bc(x)
        deterministic = actor(x, deterministic=True)
        torch.testing.assert_close(deterministic, expected, atol=1e-6, rtol=0)
        return {"observations": len(x), "max_abs_action_error": float((deterministic - expected).abs().max()),
                "log_std": -3, "optimizer_steps": 0, "environment_steps": 0}


if __name__ == "__main__":
    torch.manual_seed(11)
    torch.set_num_threads(1)
    result = check()
    Path(__file__).with_name("actor-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)
