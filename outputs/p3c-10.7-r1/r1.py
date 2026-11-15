"""One preregistered R1 run. Reuse P3c-10.7 actor and update equations."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from stable_baselines3 import __version__ as sb3_version

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / "p3c-10.7"
sys.path.insert(0, str(PARENT))
from conservative_sac import (ABS_LIMIT_MM, GAIN, TRAIN, BudgetedResidualEnv, ConservativeSAC,
                              DistilledPolicy, Gain, TrainPlant, action, assess, digest,
                              load_policy, model_at, panels, rollout, violations)


class Stop(RuntimeError):
    pass


def project(env, base, ledger, residual):
    plant = env.unwrapped
    _, base_target = plant.tcp_candidates(base[:3])
    mixed = np.clip(base[:3] + env.scale * (residual * GAIN), -1, 1)
    _, target = plant.tcp_candidates(mixed)
    total = ledger + (target - base_target)
    bounded = np.clip(total, -env.budget_m, env.budget_m)
    return np.where(total == bounded, target, base_target + (bounded - ledger))


def verify_trace(env, obs, expected, info, after, reward):
    trace = info["residual_budget"]
    signed = np.asarray(trace["signed_budget_m"])
    absolute = np.asarray(trace["absolute_budget_m"])
    if (not np.isfinite(after).all() or not np.isfinite(reward)
            or info["reason"] == "simulation_fault"):
        raise Stop("nonfinite or abnormal physics")
    try:
        np.testing.assert_allclose(trace["budget_target_m"], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(signed, obs[0] + trace["accepted_residual_target_delta_m"],
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(absolute, obs[1] + np.abs(trace["accepted_residual_target_delta_m"]),
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(after[40:], signed / env.budget_m, atol=1e-7, rtol=0)
        if info["ik_rejected"]:
            np.testing.assert_array_equal(trace["accepted_residual_target_delta_m"], np.zeros(3))
        assert np.all(np.abs(signed) <= env.budget_m + 1e-12)
    except AssertionError as error:
        raise Stop("budget or environment interface inconsistency") from error


class MonitoredGain(Gain):
    def step(self, residual):
        env = self.env
        obs = self.model._last_obs[0]
        residual = np.asarray(residual)
        if not self.action_space.contains(residual):
            raise Stop("illegal action")
        base = env.base_action()
        ledger = env._signed.copy(), env._absolute.copy()
        current = self.model.predict(obs, deterministic=True)[0]
        reference = action(self.reference, obs, 1.)
        targets = [project(env, base, ledger[0], r) for r in (residual, current, reference)]
        self.latest_target_delta_mm = 1000 * np.abs(targets[1] - targets[2])
        try:
            after, reward, term, trunc, info = super().step(residual)
            record = dict(step=self.model.num_timesteps + 1, observation=obs.tolist(),
                          next_observation=after.tolist(), deterministic=current.tolist(),
                          initialization=reference.tolist(), stochastic=residual.tolist(),
                          policy_delta=(current-reference).tolist(),
                          target_delta_m=(targets[1]-targets[2]).tolist(),
                          reward=reward, terminated=term, truncated=trunc, **info)
            self.stream.write(json.dumps(record) + "\n")
            self.stream.flush()
            verify_trace(env, ledger, targets[0], info, after, reward)
        except Stop as error:
            # SB3 has not incremented its counter yet for this physical step.
            self.model.failed_physical_step = self.model.num_timesteps + 1
            raise error
        return after, reward, term, trunc, info


def check_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            check_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            check_equal(x, y)
    elif isinstance(a, (torch.Tensor, np.ndarray)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    else:
        assert a == b


class R1SAC(ConservativeSAC):
    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "gate", "window"]

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        np.testing.assert_array_equal(buffer_action[0], np.asarray(infos[0]["actor_residual"],
                                                                    dtype=np.float32))
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        if self.num_timesteps in (250, 500, 750):
            self.gate(self.num_timesteps)


def snapshot(out, model, label, initial):
    path = out / f"model-{label}.zip"
    replay_path = out / f"replay-{label}.pkl"
    model.save(path)
    model.save_replay_buffer(replay_path)
    loaded = R1SAC.load(path, device="cpu")
    loaded.load_replay_buffer(replay_path)
    check_equal(model.policy.state_dict(), loaded.policy.state_dict())
    check_equal(model.actor.optimizer.state_dict(), loaded.actor.optimizer.state_dict())
    check_equal(model.critic.optimizer.state_dict(), loaded.critic.optimizer.state_dict())
    a, b = model.replay_buffer, loaded.replay_buffer
    assert (a.pos, a.full) == (b.pos, b.full)
    for field in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        check_equal(getattr(a, field)[:a.size()], getattr(b, field)[:b.size()])
    check_equal(initial["reference"], model.actor.reference.state_dict())
    return {name: max(float((initial[name][k] - v).abs().max())
                      for k, v in module.state_dict().items())
            for name, module in (("actor", model.actor), ("critic", model.critic))}


def calibration(reference, bc):
    history = PARENT / "run"
    rows = [json.loads(line) for line in (history / "online.jsonl").read_text().splitlines()]
    env = Gain(BudgetedResidualEnv(TrainPlant(), bc))
    sample, projected = [], []
    try:
        obs, _ = env.reset(seed=11, options=dict(scene_seed=rows[0]["scene_seed"],
                                                  orientation=rows[0]["orientation"]))
        for row in rows:
            plant = env.env
            raw = np.asarray(row["actor_residual"], dtype=np.float32)
            deterministic = action(reference, obs, 1.)
            base = plant.base_action()
            targets = [project(plant, base, plant._signed, r) for r in (raw, deterministic)]
            sample.append(np.abs(raw-deterministic))
            projected.append(1000 * np.abs(targets[0]-targets[1]))
            obs, _, _, _, info = env.step(raw)
            assert info["residual_budget"] == row["residual_budget"]
    finally:
        env.close()
    return dict(steps=len(rows), quantiles=[.5, .95, 1.],
                exploration_abs=np.quantile(sample, [.5, .95, 1.], axis=0).tolist(),
                exploration_projected_mm=np.quantile(projected, [.5, .95, 1.], axis=0).tolist(),
                zero_update_deterministic_delta=[0, 0, 0])


def config():
    return dict(seed=11, steps=1000, train_seeds=[18000, 18079], warmup=500,
                effective_train_freq=4, batch=128, buffer=2000, actor_lr=1e-4, critic_lr=3e-4,
                actor_critic_hidden=[64, 64], gain=GAIN, alpha=.005, anchor=.05, gamma=.99,
                tau=.005, initial_log_std=-3, log_std=[-4, -3], sb3="2.7.1",
                absolute_review_mm=ABS_LIMIT_MM.tolist(), saturation_reporting_threshold=.95,
                drift_window=50, drift_raw_p95=.25, drift_projected_p95_mm=.046875,
                rule="Post-update only: stop at paired gate if both same-axis drift limits exceeded "
                     "in any rolling 50-step window since previous gate. Stop for any lost baseline "
                     "success, new contact, simulation fault, or paired absolute review exceedance. "
                     "Online contact and absolute review exceedance are logged; invariant violations stop.")


def main(out):
    started = time.perf_counter()
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    assert sb3_version == "2.7.1"
    chosen = json.loads((ROOT.parent / "p3c-10.6/selected.json").read_text())
    assert digest(Path(chosen["checkpoint"])) == chosen["checkpoint_sha256"]
    distilled = model_at(Path(chosen["checkpoint"]))
    bc = load_policy()
    cfg = config()
    calibrated = calibration(distilled, bc)
    assert max(calibrated["exploration_abs"][1]) < cfg["drift_raw_p95"]
    (out / "calibration.json").write_text(json.dumps(calibrated, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    env = MonitoredGain(BudgetedResidualEnv(TrainPlant(), bc))
    model = R1SAC(DistilledPolicy, env, seed=11, device="cpu", ent_coef=.005,
                  learning_starts=500, buffer_size=2000, batch_size=128, train_freq=1,
                  gradient_steps=1, gamma=.99, tau=.005, learning_rate=.0003,
                  policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh,
                                     optimizer_kwargs=dict(eps=1e-8)))
    model.actor.initialize(distilled)
    env.model, env.reference = model, distilled
    initial = {name: {k: v.clone() for k, v in module.state_dict().items()}
               for name, module in (("actor", model.actor), ("critic", model.critic),
                                    ("reference", model.actor.reference))}
    cases = panels()
    assert len(cases) == 62 and all(c[1] not in TRAIN and c[1] not in range(17700, 17740) for c in cases)

    def reference(obs):
        return action(distilled, obs, GAIN)

    def policy(obs):
        return model.predict(obs, deterministic=True)[0] * GAIN

    baseline = assess(out, "residual-bc", cases, bc, reference)
    assert all(r["is_success"] for r in baseline)
    old = json.loads((PARENT / "run/residual-bc.json").read_text())
    assert baseline == old
    checkpoints = []
    model.window = deque(maxlen=cfg["drift_window"])
    drift = np.zeros(3, dtype=bool)
    evaluation_steps = sum(r["steps"] for r in baseline)

    def evaluate(step):
        nonlocal evaluation_steps
        change = snapshot(out, model, step, initial)
        expected = max(0, (step - 500) // 4)
        assert model._n_updates == expected and model.replay_buffer.size() == step
        assert all(v > 0 for v in change.values()) if expected else all(v == 0 for v in change.values())
        if not expected:
            for row in baseline:
                obs = np.asarray(row["initial"], dtype=np.float32)
                np.testing.assert_allclose(policy(obs), reference(obs), atol=5e-8, rtol=0)
        rows = assess(out, f"step-{step}", cases, bc, policy, baseline,
                      exact=reference if not expected else None)
        evaluation_steps += sum(r["steps"] for r in rows)
        unsafe = violations(rows, baseline)
        checkpoints.append(dict(step=step, updates=model._n_updates, parameter_change=change,
                                violations=unsafe, relative_drift=drift.tolist()))
        if unsafe or drift.any():
            raise Stop("paired capability or drift gate")
        drift[:] = False

    evaluate(0)

    def store_gate(step):
        evaluate(step)

    model.gate = store_gate
    reason = "budget_complete"
    online_start = time.perf_counter()
    with (out / "updates.jsonl").open("w") as model.update_stream, (out / "online.jsonl").open("w") as env.stream:
        original_step = env.step

        def monitored_step(residual):
            pair = original_step(residual)
            obs = model._last_obs[0]
            current = model.predict(obs, deterministic=True)[0]
            expected = action(distilled, obs, 1.)
            raw = np.abs(current - expected)
            # Full post-projection deltas are logged by MonitoredGain.
            if model._n_updates:
                model.window.append((raw, env.latest_target_delta_mm))
                if len(model.window) == cfg["drift_window"]:
                    p95 = np.quantile(np.stack(model.window), .95, axis=0)
                    drift[:] |= (p95[0] > cfg["drift_raw_p95"]) & (p95[1] > cfg["drift_projected_p95_mm"])
            return pair

        env.step = monitored_step
        try:
            model.learn(total_timesteps=1000)
            evaluate(1000)
        except (Stop, FloatingPointError, AssertionError, OSError, ValueError) as error:
            reason = str(error) or type(error).__name__
            (out / "failure.txt").write_text(reason + "\n")
            if not checkpoints or checkpoints[-1]["step"] != model.num_timesteps:
                try:
                    snapshot(out, model, "failure", initial)
                except Exception as save_error:
                    (out / "failure.txt").open("a").write(repr(save_error) + "\n")
    (out / "status.json").write_text(json.dumps(dict(checkpoints=checkpoints,
        steps=max(model.num_timesteps, getattr(model, "failed_physical_step", 0)),
        updates=model._n_updates, stop_reason=reason, calibration_steps=calibrated["steps"],
        evaluation_steps=evaluation_steps, online_including_gates_s=time.perf_counter()-online_start,
        wall_s=time.perf_counter()-started), indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
