"""Bounded XYZ residual control around a frozen Grasp & Lift policy."""
import gymnasium as gym
import numpy as np

from cr3_lab.core.model import TCP
from .reach import TCP_DELTA_M

DEFAULT_BUDGET_M = (0.006, 0.003, 0.012)


def _axis_config(values, name, upper):
    array = np.array(values, dtype=np.float64)
    if (array.shape != (3,) or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > upper)):
        raise ValueError(f"Invalid XYZ {name}")
    array.setflags(write=False)
    return array


class BudgetedResidualEnv(gym.Wrapper):
    """BC reads the original observation. The residual sees the budget ledger too."""

    def __init__(self, env, bc, *, budget_m=DEFAULT_BUDGET_M, scale=(0.5, 0.5, 0.5)):
        super().__init__(env)
        self.bc = bc
        self.budget_m = _axis_config(budget_m, "budget", np.inf)
        self.scale = _axis_config(scale, "scale", 1).astype(np.float32)
        self.scale.setflags(write=False)
        self.action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            np.r_[env.observation_space.low, [-1, -1, -1]].astype(np.float32),
            np.r_[env.observation_space.high, [1, 1, 1]].astype(np.float32))
        self._current = None
        self._signed = np.zeros(3)
        self._absolute = np.zeros(3)

    def _observation(self, obs):
        normalized = np.divide(self._signed, self.budget_m, out=np.zeros(3),
                               where=self.budget_m > 0)
        return np.concatenate((obs, normalized.astype(np.float32)))

    def reset(self, **kwargs):
        self._current = None
        obs, info = self.env.reset(**kwargs)
        return self.begin_grasp(obs), info

    def begin_grasp(self, obs):
        """Clear residual bookkeeping for an initialized plant."""
        self._signed.fill(0)
        self._absolute.fill(0)
        self._current = obs.copy()
        return self._observation(obs)

    def base_action(self):
        if self._current is None:
            raise gym.error.ResetNeeded("Reset before requesting a BC action")
        return np.asarray(self.bc(self._current.copy()), dtype=np.float32)

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(residual):
            raise ValueError("Expected finite normalized XYZ residual")
        base = self.base_action()
        if not self.env.action_space.contains(base):
            raise ValueError("Invalid frozen BC action")
        scaled = self.scale * residual
        mixed = base.copy()
        mixed[:3] = np.clip(base[:3] + scaled, -1, 1)
        plant = self.unwrapped
        before = plant.tcp_target.copy()
        tcp_before = plant.data.site(TCP).xpos.copy()
        base_requested, base_target = plant.tcp_candidates(base[:3])
        mixed_requested, mixed_target = plant.tcp_candidates(mixed[:3])
        difference = mixed_target - base_target
        next_signed = np.clip(self._signed + difference, -self.budget_m, self.budget_m)
        projected = base_target + (next_signed - self._signed)
        # Preserve the original float representation wherever projection is inactive.
        projected = np.where((self._signed + difference == next_signed), mixed_target, projected)
        trace = {}
        obs, reward, terminated, truncated, info = plant._step_target(
            projected, base[3], ik_diagnostics=trace)
        accepted = np.zeros(3)
        if not info["ik_rejected"]:
            accepted = plant.tcp_target - base_target
            self._signed[:] = next_signed
            self._absolute += abs(accepted)
        final = mixed.astype(float)
        final[:3] = np.where(projected == base_target, base[:3],
                             np.where(projected == mixed_target, mixed[:3],
                                      (projected - before) / TCP_DELTA_M))
        trace.update(
            bc_action=base.tolist(), raw_residual=residual.tolist(), scaled_residual=scaled.tolist(),
            mixed_action=mixed.tolist(), final_action=final.tolist(),
            bc_requested_target_m=base_requested.tolist(), mixed_requested_target_m=mixed_requested.tolist(),
            bc_workspace_target_m=base_target.tolist(), mixed_workspace_target_m=mixed_target.tolist(),
            budget_target_m=projected.tolist(), accepted_residual_target_delta_m=accepted.tolist(),
            applied_tcp_target_delta_m=(plant.tcp_target - before).tolist(),
            tcp_target_m=plant.tcp_target.tolist(), tcp_world_m=plant.data.site(TCP).xpos.tolist(),
            measured_tcp_delta_m=(plant.data.site(TCP).xpos - tcp_before).tolist(),
            signed_budget_m=self._signed.tolist(), absolute_budget_m=self._absolute.tolist(),
        )
        self._current = None if terminated or truncated else obs.copy()
        return self._observation(obs), reward, terminated, truncated, {**info, "residual_budget": trace}
