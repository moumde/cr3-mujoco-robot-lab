"""Gymnasium residual actions on the frozen P3c-6 policy."""
from pathlib import Path
import sys
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3c-7"))
from benchmark import ChallengeEnv, load_policy, sample_xy

from cr3_lab.rl.reach import TCP_DELTA_M
from cr3_lab.tasks.ik import Unreachable
from cr3_lab.tasks.motor_scene import Orientation

TRAIN_SCENES = range(18000, 18100)


def compose(base, residual, scale):
    return np.clip(base + scale * residual, -1, 1)


class ResidualEnv(gym.Wrapper):
    """Expose raw observations and normalized residuals; the plant sees composed actions."""

    def __init__(self, env, bc, scale=(0.5, 0.5, 0.5, 0.), *, xyz=True, gripper=False):
        super().__init__(env)
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape != (4,) or not np.isfinite(scale).all() or np.any((scale < 0) | (scale > 1)):
            raise ValueError("Residual scale must be four values in [0,1]")
        self.scale = scale * np.array([xyz, xyz, xyz, gripper], dtype=np.float32)
        self.bc = bc
        self.current = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current = obs.copy()
        return obs, info

    def base_action(self):
        if self.current is None:
            raise gym.error.ResetNeeded("Reset before requesting a BC action")
        return self.bc(self.current)

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(residual):
            raise ValueError("Expected a normalized four-dimensional residual")
        base = self.base_action()
        final = compose(base, residual, self.scale)
        env = self.unwrapped
        before = env.tcp_target.copy()
        joints = env.joint_target.copy()
        solver = env.robot.ik
        solve = solver.solve
        status = "accepted"

        def observe(position, rotation, seed):
            nonlocal status
            try:
                candidate = solve(position, rotation, seed)
            except Unreachable:
                status = "solver_nonconverged"
                raise
            if np.max(abs(candidate - joints)) > 0.03:
                status = "joint_step_limit"
            return candidate

        with patch.object(solver, "solve", observe):
            obs, reward, terminated, truncated, info = self.env.step(final)
        applied = (env.tcp_target - before) / TCP_DELTA_M
        self.current = None if terminated or truncated else obs.copy()
        return obs, reward, terminated, truncated, {**info, "residual_control": {
            "bc_action": base.tolist(), "residual_action": residual.tolist(),
            "final_action": final.tolist(), "applied_tcp_delta_m": (env.tcp_target - before).tolist(),
            "applied_xyz_action": applied.tolist(), "applied_closure": float(obs[30]),
            "ik_status": status}}


class TrainingChallenge(ChallengeEnv):
    """Opt-in ±35 mm training pool with both orientations, disjoint from held-out Test."""

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if "scene_seed" not in options:
            options["scene_seed"] = int(self.np_random.choice(TRAIN_SCENES))
        if options["scene_seed"] not in TRAIN_SCENES:
            raise ValueError("Scene seed is outside the training pool")
        if "orientation" not in options:
            options["orientation"] = list(Orientation)[int(self.np_random.integers(2))]
        self.xy = sample_xy(0.035, options["scene_seed"])
        return super().reset(seed=seed, options=options)


def initialize_sac_actor(actor, log_std=-3.):
    """Zero the independent mean head; keep critic and BC weights independent."""
    with torch.no_grad():
        actor.mu.weight.zero_()
        actor.mu.bias.zero_()
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(log_std)
