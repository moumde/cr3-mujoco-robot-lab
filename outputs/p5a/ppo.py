"""One conservative PPO experiment with a bounded residual distribution."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-10.7-r1"))
from r1 import (MonitoredGain, BudgetedResidualEnv, TrainPlant, Stop, GAIN, action, assess,
                check_equal, config, digest, load_policy, model_at, panels, violations)


class Squashed(SquashedDiagGaussianDistribution):
    def entropy(self):
        return None

    def log_prob(self, actions, gaussian_actions=None):
        # Both rollout and update evaluate the same float32 stored action.
        return super().log_prob(actions)


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(43))
        self.register_buffer("spread", torch.ones(43))

    def forward(self, x):
        return (x - self.center) / self.spread


class Networks(nn.Module):
    latent_dim_pi = latent_dim_vf = 64

    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(Normalize(), nn.Linear(43, 64), nn.Tanh(),
                                        nn.Linear(64, 64), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(43, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class DistilledPPOPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = Networks().to(self.device)
        self.action_dist = Squashed(3)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std.clamp(-4, -3))

    def _predict(self, observation, deterministic=False):
        if deterministic and len(observation) == 1:
            latent = self.mlp_extractor.forward_actor(observation[0])
            return self.action_net(latent).tanh()[None]
        return super()._predict(observation, deterministic)

    @torch.no_grad()
    def initialize(self, model):
        net = self.mlp_extractor.policy_net
        net[0].center.copy_(model.center)
        net[0].spread.copy_(model.spread)
        for dest, source in zip(net[1:], model.network[:4], strict=True):
            dest.load_state_dict(source.state_dict())
        self.action_net.load_state_dict(model.network[4].state_dict())
        self.log_std.fill_(-3.)


class ResidualPPO(PPO):
    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "gate", "update_stream"]

    def train(self):
        before = self._n_updates
        count = 0

        def counted(*_args):
            nonlocal count
            count += 1

        hook = self.policy.optimizer.register_step_post_hook(counted)
        try:
            super().train()
        finally:
            hook.remove()
        self.gradient_steps = getattr(self, "gradient_steps", 0) + count
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
            raise FloatingPointError("nonfinite PPO parameter")
        measures = {k.removeprefix("train/"): float(v) for k, v in self.logger.name_to_value.items()
                    if k.startswith("train/")}
        self.update_stream.write(json.dumps(dict(step=self.num_timesteps, epochs=self._n_updates-before, gradient_steps=count,
                                                 log_std=self.policy.log_std.detach().clamp(-4,-3).tolist(),
                                                 **measures), allow_nan=False)+"\n")
        self.update_stream.flush()
        self.gate(self.num_timesteps)


def main(out, *, learning_rate=.0001):
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    selection = json.loads((ROOT.parent / "p3c-10.6/selected.json").read_text())
    assert digest(Path(selection["checkpoint"])) == selection["checkpoint_sha256"]
    reference = model_at(Path(selection["checkpoint"]))
    bc = load_policy()
    cfg = dict(seed=11, steps=750, n_steps=250, batch_size=125, n_epochs=2, learning_rate=learning_rate,
               clip_range=.1, target_kl=.01, gamma=.99, gae_lambda=.95, ent_coef=0.,
               vf_coef=.5, max_grad_norm=.5)
    (out/"config.json").write_text(json.dumps(dict(cfg, gain=GAIN, log_std=[-4,-3],
        absolute_review_mm=config()["absolute_review_mm"], drift_raw_p95=.25,
        drift_projected_p95_mm=.046875, drift_window=50), indent=2)+"\n")
    env = MonitoredGain(BudgetedResidualEnv(TrainPlant(), bc))
    model = ResidualPPO(
        DistilledPPOPolicy, env, device="cpu", policy_kwargs=dict(ortho_init=False),
        **{k:v for k,v in cfg.items() if k != "steps"})
    model.policy.initialize(reference)
    env.model, env.reference = model, reference
    cases = panels()
    baseline = json.loads((ROOT.parent/"p3c-10.7-r1/run/residual-bc.json").read_text())
    assert len(cases) == len(baseline) == 62
    initial = {k:v.clone() for k,v in model.policy.state_dict().items()}
    checks = []

    def gate(step):
        path = out/f"model-{step}.zip"
        model.save(path)
        restored = ResidualPPO.load(path, device="cpu")
        check_equal(restored.policy.state_dict(), model.policy.state_dict())
        check_equal(restored.policy.optimizer.state_dict(), model.policy.optimizer.state_dict())
        rows = assess(out, f"step-{step}", cases, bc,
                      lambda obs: model.predict(obs, deterministic=True)[0] * GAIN,
                      baseline, exact=(lambda obs: action(reference, obs, GAIN)) if not step else None)
        unsafe = violations(rows, baseline)
        if not step:
            assert all(r["is_success"] and not r["collision"] for r in rows)
        drift = [False]*3
        if step:
            last = [json.loads(l) for l in (out/"online.jsonl").read_text().splitlines()][-250:]
            pairs = np.abs(np.array([[r["policy_delta"], r["target_delta_m"]] for r in last]))
            for start in range(len(pairs)-49):
                a,b = np.quantile(pairs[start:start+50], .95, axis=0)
                drift = (np.array(drift) | ((a>.25)&(b>.000046875))).tolist()
        changes = {group: max(float((v-initial[k]).abs().max()) for k,v in model.policy.state_dict().items()
                              if k.startswith(prefixes))
                   for group,prefixes in (("actor", ("mlp_extractor.policy", "action_net", "log_std")),
                                          ("critic", ("mlp_extractor.value", "value_net")))}
        checks.append(dict(step=step, epochs=model._n_updates, gradient_steps=getattr(model,"gradient_steps",0), violations=unsafe, drift=drift,
                           parameter_change=changes))
        if unsafe or any(drift):
            raise Stop("paired review")

    model.gate = gate
    gate(0)
    reason = "budget_complete"
    started = perf_counter()
    with (out/"updates.jsonl").open("w") as model.update_stream,(out/"online.jsonl").open("w") as env.stream:
        try:
            model.learn(total_timesteps=750)
        except (Stop, FloatingPointError, ValueError, AssertionError, OSError) as exc:
            reason = str(exc) or type(exc).__name__
            if not checks or checks[-1]["step"] != model.num_timesteps:
                model.save(out/"failure")
    (out/"status.json").write_text(json.dumps(dict(stop=reason, steps=max(model.num_timesteps,
        getattr(model,"failed_physical_step",0)), checkpoints=checks,
        optimization_epochs=model._n_updates, wall_including_eval_s=perf_counter()-started),indent=2)+"\n")
    env.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--learning-rate", type=float, choices=(.0001, .00001), default=.0001)
    args = p.parse_args()
    main(args.out, learning_rate=args.learning_rate)
