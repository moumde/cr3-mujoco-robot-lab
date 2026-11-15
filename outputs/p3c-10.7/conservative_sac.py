"""1,000-step development-only residual SAC gate using SB3 replay and rollout."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, SACPolicy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-10.6"))
from experiment import (BudgetedResidualEnv, TrainingChallenge, action, load_policy,
                        model_at, panel, rollout, summarize)
from dataset import digest, plan, sample_xy

TRAIN = range(18000, 18080)
GAIN = .25
# Predeclared paired drift guard based on the selected policy's panel bounds.
ABS_LIMIT_MM = np.array([20., 5., 22.])
SAT_THRESHOLD = .95
SAT_WINDOW = 50
SAT_LIMIT = .50


class DistilledActor(Actor):
    """Apply checkpoint normalization before the copied pre-tanh mean."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("center", torch.zeros(43))
        self.register_buffer("spread", torch.ones(43))
        self.register_buffer("reference_center", torch.zeros(43))
        self.register_buffer("reference_spread", torch.ones(43))
        self.reference = nn.Sequential(nn.Linear(43, 64), nn.Tanh(), nn.Linear(64, 64),
                                       nn.Tanh(), nn.Linear(64, 3))
        self.reference.requires_grad_(False)

    def forward(self, obs, deterministic=False):
        if deterministic and obs.ndim == 2 and len(obs) == 1:
            # Match the distilled policy's unbatched linear kernels exactly.
            features = self.extract_features(obs, self.features_extractor)[0]
            return self.mu(self.latent_pi((features - self.center) / self.spread)).tanh()[None]
        return super().forward(obs, deterministic)

    def get_action_dist_params(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        latent = self.latent_pi((features - self.center) / self.spread)
        return self.mu(latent), self.log_std(latent).clamp(-4., -3.), {}

    def initialize(self, model):
        with torch.no_grad():
            self.center.copy_(model.center)
            self.spread.copy_(model.spread)
            self.reference_center.copy_(model.center)
            self.reference_spread.copy_(model.spread)
            self.latent_pi.load_state_dict(model.network[:4].state_dict())
            self.mu.load_state_dict(model.network[4].state_dict())
            self.reference.load_state_dict(model.network[:5].state_dict())
            self.log_std.weight.zero_()
            self.log_std.bias.fill_(-3.)


class DistilledPolicy(SACPolicy):
    # This policy's action box is exactly [-1,1]. Avoid float32 affine round trips.
    def scale_action(self, action):
        return action

    def unscale_action(self, action):
        return action

    def make_actor(self, features_extractor=None):
        cfg = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return DistilledActor(**cfg).to(self.device)


class ConservativeSAC(SAC):
    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "update_stream"]


    def _setup_model(self):
        super()._setup_model()
        self.actor.optimizer = torch.optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad], lr=1e-4, eps=1e-8)


    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)

    def train(self, gradient_steps, batch_size=128):
        # One-step rollouts give exact checkpoint ticks. Updates remain 1 per 4 steps.
        if self.num_timesteps % 4:
            return
        self.policy.set_training_mode(True)
        self.actor.optimizer.param_groups[0]["lr"] = 1e-4
        self.critic.optimizer.param_groups[0]["lr"] = 3e-4
        batch = self.replay_buffer.sample(batch_size)
        alpha = self.ent_coef_tensor
        with torch.no_grad():
            next_action, next_logp = self.actor.action_log_prob(batch.next_observations)
            next_q = torch.cat(self.critic_target(batch.next_observations, next_action), 1).min(1, keepdim=True).values
            target = batch.rewards + (1 - batch.dones) * self.gamma * (next_q - alpha * next_logp[:, None])
        values = self.critic(batch.observations, batch.actions)
        critic_loss = .5 * sum(nn.functional.mse_loss(q, target) for q in values)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("nonfinite critic loss")
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        actions, logp = self.actor.action_log_prob(batch.observations)
        q = torch.cat(self.critic(batch.observations, actions), 1).min(1).values
        mu, logstd, _ = self.actor.get_action_dist_params(batch.observations)
        with torch.no_grad():
            reference = self.actor.reference((batch.observations - self.actor.reference_center) /
                                             self.actor.reference_spread)
        anchor = (mu.tanh() - reference.tanh()).square().mean()
        actor_loss = (alpha * logp - q).mean() + .05 * anchor
        if not torch.isfinite(actor_loss):
            raise FloatingPointError("nonfinite actor loss")
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
        self._n_updates += 1
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
            raise FloatingPointError("nonfinite policy parameter")
        self.update_stream.write(json.dumps(dict(step=self.num_timesteps, updates=self._n_updates,
                                            actor=actor_loss.item(), critic=critic_loss.item(),
                                            alpha=alpha.item(), anchor=anchor.item(),
                                            entropy=-logp.mean().item(), log_std=logstd.mean(0).detach().tolist())) + "\n")
        self.update_stream.flush()


class TrainPlant(TrainingChallenge):
    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        if "scene_seed" not in options:
            options["scene_seed"] = int(self.np_random.choice(TRAIN))
        if options["scene_seed"] not in TRAIN:
            raise ValueError("outside training split")
        return super().reset(options=options)


class Gain(gym.Wrapper):
    def step(self, residual):
        obs, reward, term, trunc, info = self.env.step(residual * GAIN)
        return obs, reward, term, trunc, dict(info, actor_residual=residual.tolist())


def panels():
    history = [case for case in panel() if case[0] in ("fixed", "35mm")]
    validation = [("validation", seed, orientation, sample_xy(.035, seed).tolist(), None)
                  for seed, orientation, split in plan() if split == "validation"]
    return history + validation


def assess(out, label, cases, bc, policy, baseline=None, exact=None):
    results = []
    with gzip.open(out / f"{label}.jsonl.gz", "xt") as stream:
        for index, case in enumerate(cases):
            def predict(obs):
                result = policy(obs)
                if exact is not None:
                    np.testing.assert_array_equal(result, exact(obs))
                return result
            initial, info = rollout(case, bc, "residual", predict, stream)
            record = {**info, "profile": case[0], "initial": initial.tolist()}
            if baseline is not None:
                np.testing.assert_array_equal(record["initial"], baseline[index]["initial"])
                if exact is not None:
                    assert record == baseline[index]
            results.append(record)
    (out / f"{label}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(label, sum(r["is_success"] for r in results[:42]),
          sum(r["is_success"] for r in results[42:]), flush=True)
    return results


def violations(rows, baseline):
    reasons = []
    for row, previous in zip(rows, baseline, strict=True):
        if (previous["is_success"] and not row["is_success"] or
                row["collision"] and row["collision"] != previous["collision"] or
                row["reason"] == "simulation_fault" or
                np.any(1000 * np.array(row["absolute_m"]) > ABS_LIMIT_MM)):
            reasons.append([row["scene_seed"], row["orientation"], row["reason"], row["absolute_m"]])
    return reasons


class GateCallback(BaseCallback):
    def __init__(self, out, baseline, evaluate, stream):
        super().__init__()
        self.out, self.baseline, self.evaluate, self.stream = out, baseline, evaluate, stream
        self.stopped = False
        self.stop_reason = None
        self.saturation = []

    def _on_step(self):
        info = self.locals["infos"][0]
        self.stream.write(json.dumps(dict(step=self.num_timesteps, **info), default=lambda x: x.tolist()) + "\n")
        monitor = info["residual_budget"]
        self.saturation.append(np.abs(info["actor_residual"]) >= SAT_THRESHOLD)
        saturated = len(self.saturation) >= SAT_WINDOW and np.any(
            np.mean(self.saturation[-SAT_WINDOW:], axis=0) > SAT_LIMIT)
        if (saturated or info["reason"] == "simulation_fault" or
                np.any(1000 * np.array(monitor["absolute_budget_m"]) > ABS_LIMIT_MM)):
            self.stopped = True
            self.stop_reason = "persistent_saturation" if saturated else "budget_or_simulation_fault"
            self.model.save(self.out / f"model-{self.num_timesteps}-unsafe")
            self.evaluate(self.num_timesteps)
            return False
        if self.num_timesteps in (250, 500, 750):
            self.stopped = bool(self.evaluate(self.num_timesteps))
        return not self.stopped


def main(out):
    started = time.perf_counter()
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    chosen = json.loads((ROOT.parent / "p3c-10.6/selected.json").read_text())
    assert chosen["normalized_residual_gain"] == GAIN
    assert digest(Path(chosen["checkpoint"])) == chosen["checkpoint_sha256"]
    distilled = model_at(Path(chosen["checkpoint"]))
    bc = load_policy()
    env = Gain(BudgetedResidualEnv(TrainPlant(), bc))
    model = ConservativeSAC(DistilledPolicy, env, seed=11, device="cpu", ent_coef=.005,
                            learning_starts=500, buffer_size=2000, batch_size=128,
                            train_freq=1, gradient_steps=1, gamma=.99, learning_rate=.0003,
                            policy_kwargs=dict(net_arch=[64, 64], activation_fn=nn.Tanh,
                                               optimizer_kwargs=dict(eps=1e-8)))
    model.actor.initialize(distilled)
    cases = panels()
    assert len(cases) == 62 and all(c[1] not in TRAIN for c in cases)
    (out / "config.json").write_text(json.dumps(dict(
        seed=11, steps=1000, train_seeds=[18000, 18079], warmup=500, effective_train_freq=4,
        batch=128, buffer=2000, actor_lr=1e-4, critic_lr=3e-4, gain=GAIN, alpha=.005,
        anchor=.05, gamma=.99, tau=.005, log_std=[-4, -3],
        absolute_stop_mm=ABS_LIMIT_MM.tolist(), saturation_threshold=SAT_THRESHOLD,
        saturation_window=SAT_WINDOW, saturation_fraction_limit=SAT_LIMIT), indent=2) + "\n")

    def policy(obs):
        return model.predict(obs, deterministic=True)[0] * GAIN

    def reference(obs):
        return action(distilled, obs, GAIN)

    baseline = assess(out, "residual-bc", cases, bc, reference)
    assert sum(row["is_success"] for row in baseline[:42]) == 42
    prior = json.loads((ROOT.parent / "p3c-10.6/run/gain-0.25.json").read_text())["rows"]
    for row, saved in zip(baseline[:42], prior, strict=True):
        assert all(row[k] == saved["residual"][k] for k in ("steps", "reason", "absolute_m"))
    model.save(out / "model-0")
    assess(out, "step-0", cases, bc, policy, baseline, exact=reference)
    with gzip.open(out / "step-0.jsonl.gz", "rb") as a, gzip.open(out / "residual-bc.jsonl.gz", "rb") as b:
        from hashlib import file_digest
        assert file_digest(a, "sha256").digest() == file_digest(b, "sha256").digest()
    checkpoints = [{"step": 0, "updates": 0, "violations": []}]

    def evaluate(step):
        model.save(out / f"model-{step}")
        rows = assess(out, f"step-{step}", cases, bc, policy, baseline)
        unsafe = violations(rows, baseline)
        checkpoints.append(dict(step=step, updates=model._n_updates, violations=unsafe))
        return unsafe

    started_online = time.perf_counter()
    with (out / "updates.jsonl").open("w") as model.update_stream, (out / "online.jsonl").open("w") as stream:
        callback = GateCallback(out, baseline, evaluate, stream)
        try:
            model.learn(total_timesteps=1000, callback=callback)
        except FloatingPointError as error:
            model.save(out / f"model-{model.num_timesteps}-nonfinite")
            checkpoints.append(dict(step=model.num_timesteps, updates=model._n_updates, error=str(error)))
        if model.num_timesteps == 1000 and not callback.stopped:
            evaluate(1000)
    model.save_replay_buffer(out / "replay.pkl")
    (out / "status.json").write_text(json.dumps(dict(checkpoints=checkpoints, steps=model.num_timesteps,
        updates=model._n_updates, stop_reason=callback.stop_reason, wall_s=time.perf_counter()-started,
        online_including_gates_s=time.perf_counter()-started_online), indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
