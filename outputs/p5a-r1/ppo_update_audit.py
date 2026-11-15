"""Read-only PPO rollout reconstruction and paired failure traces."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p5a"))
from ppo import ResidualPPO, GAIN, BudgetedResidualEnv, action, load_policy, model_at, panels
from experiment import plant_for
from dataset import phase
from conservative_sac import TRAIN

HISTORY = ROOT.parent / "p5a/run"


def stats(x):
    x = np.asarray(x)
    return dict(mean=np.mean(x, axis=0).tolist(),
                p=np.quantile(x, [0, .05, .5, .95, 1], axis=0).tolist())


def advantage(rewards, values, next_values, terminal, boundary):
    result = np.empty_like(rewards)
    tail = 0.
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + .99 * next_values[i] * (not terminal[i]) - values[i]
        tail = delta + .99 * .95 * tail * (not boundary[i])
        result[i] = tail
    return result


def rollout_audit(old, new):
    records = [json.loads(line) for line in (HISTORY / "online.jsonl").read_text().splitlines()]
    assert len(records) == 250 and all(r["scene_seed"] in TRAIN for r in records)
    x, after, actions = [torch.tensor([r[key] for r in records], dtype=torch.float32)
                         for key in ("observation", "next_observation", "stochastic")]
    rewards = np.asarray([r["reward"] for r in records], np.float32)
    terminated = [r["terminated"] for r in records]
    boundaries = [r["terminated"] or r["truncated"] for r in records]

    with torch.no_grad():
        old_values, old_lp, _ = old.policy.evaluate_actions(x, actions)
        new_values, new_lp, _ = new.policy.evaluate_actions(x, actions)
        old_v = old_values.flatten().numpy()
        next_v = old.policy.predict_values(after).flatten().numpy()
        advantages = advantage(rewards, old_v, next_v, terminated, boundaries)
        p = old.policy.get_distribution(x).distribution
        q = new.policy.get_distribution(x).distribution
        kl = torch.distributions.kl_divergence(p, q).sum(-1)
        log_ratio = new_lp-old_lp
        ratio = log_ratio.exp()
        recomputed = torch.stack([old.policy.evaluate_actions(o[None], a[None])[1][0]
                                  for o, a in zip(x, actions, strict=True)])
        mean_old = p.mean.tanh()
        mean_new = q.mean.tanh()
        raw = [r["actor_residual"] for r in records]
        np.testing.assert_array_equal(actions.numpy(), np.asarray(raw, np.float32))
        np.testing.assert_allclose([r["residual_budget"]["raw_residual"] for r in records],
                                   actions.numpy()*GAIN, atol=0, rtol=0)
        assert actions.abs().max() < 1 and torch.isfinite(old_lp).all()
        max_lp_error = float((old_lp-recomputed).abs().max())
        assert max_lp_error < .001
    # Read-only gradient of the full-rollout surrogate at the old parameters.
    # It differs from the historical shuffled 125-item minibatch gradient.
    values, logp, _ = old.policy.evaluate_actions(x, actions)
    adv = torch.from_numpy(advantages)
    normalized = (adv-adv.mean())/(adv.std()+1e-8)
    objective = -(normalized*(logp-old_lp).exp()).mean() + .5*(values.flatten()-adv-old_values.flatten()).square().mean()
    grads = torch.autograd.grad(objective, tuple(old.policy.parameters()))
    gradient_norm = float(torch.stack([g.norm().square() for g in grads]).sum().sqrt())
    return dict(
        steps=len(records), success=sum(r["is_success"] for r in records),
        terminal_rewards=sum(boundaries), positive_reward_steps=int((rewards > 0).sum()),
        phase_counts=dict(Counter(phase(np.array(r["observation"])) for r in records)),
        old_mean=stats(mean_old), new_mean=stats(mean_new),
        old_std=p.scale[0].tolist(), new_std=q.scale[0].tolist(),
        deterministic_abs_delta=stats((mean_new-mean_old).abs()),
        probability_ratio=stats(ratio), approximate_kl=float((ratio-1-log_ratio).mean()),
        gaussian_kl=stats(kl), clip_fraction=float(((ratio-1).abs()>.1).float().mean()),
        advantages=stats(advantages), value_delta=stats((new_values-old_values).flatten()),
        full_rollout_gradient_norm=gradient_norm, old_single_batch_max_logp_error=max_lp_error,
        original_optimizer_steps=sorted({float(s["step"]) for s in new.policy.optimizer.state.values()}),
        raw_std_units_of_mean_delta=stats(((q.mean-p.mean)/p.scale).abs()),
    )


def trace(out, case, bc, reference, old, new, extra=None):
    profile, seed, orientation, xy, _ = case
    initial = None
    result = {}
    methods = [("bc",lambda o:action(reference,o,1.)),
               ("ppo0",lambda o:old.predict(o,deterministic=True)[0]),
               ("ppo250",lambda o:new.predict(o,deterministic=True)[0])]
    if extra is not None:
        methods.append(("r1", lambda o:extra.predict(o,deterministic=True)[0]))
    for name, policy in methods:
        plant = plant_for(profile, xy)
        env = BudgetedResidualEnv(plant,bc)
        obs,_ = env.reset(seed=seed,options=dict(scene_seed=seed,orientation=orientation))
        if initial is None:
            initial = obs.copy()
        np.testing.assert_array_equal(initial,obs)
        rows=[]
        try:
            while True:
                raw = policy(obs)
                same_old = old.predict(obs,deterministic=True)[0]
                same_new = new.predict(obs,deterministic=True)[0]
                after,reward,term,trunc,info = env.step(raw*GAIN)
                rows.append(dict(observation=obs.tolist(),next_observation=after.tolist(),
                                 action=raw.tolist(),gain_action=(raw*GAIN).tolist(),
                                 old_on_state=same_old.tolist(),new_on_state=same_new.tolist(),
                                 reward=reward,terminated=term,truncated=trunc,**info))
                obs=after
                if term or trunc:
                    break
        finally:
            env.close()
        with gzip.open(out/f"{orientation}-{name}.jsonl.gz","xt") as stream:
            for row in rows:
                stream.write(json.dumps(row,allow_nan=False)+"\n")
        result[name]=rows
    assert result["bc"] == result["ppo0"]
    return result


def events(traces):
    a,b=traces["ppo0"],traces["ppo250"]
    first={}
    for i,(r,s) in enumerate(zip(a,b),1):
        x,y=r["residual_budget"],s["residual_budget"]
        tcp = np.linalg.norm(np.subtract(x["tcp_world_m"],y["tcp_world_m"]))
        tests = {"same_state_action": s["old_on_state"]!=s["new_on_state"],
                 "effective_target": x["tcp_target_m"]!=y["tcp_target_m"],
                 "tcp_1mm": tcp>=.001, "first_ik": s["ik_rejected"]}
        for event,yes in tests.items():
            if yes and event not in first:
                first[event]=dict(step=i,tcp_delta_mm=1000*tcp, actual_x=s["residual_budget"]["tcp_world_m"],
                    target=s["residual_budget"]["budget_target_m"],
                    ik=s["residual_budget"]["ik_status"],
                    old_on_state=s["old_on_state"],new_on_state=s["new_on_state"])
    return dict(first=first, ends={k:{**{f:v[-1][f] for f in ("reason","steps","max_lift_m",
        "first_bilateral_step","first_lift_step","first_100mm_step","hold_steps")},
        "rejections":dict(Counter(r["residual_budget"]["ik_status"] for r in v if r["ik_rejected"]))}
        for k,v in traces.items()})


def main(out):
    torch.set_num_threads(1)
    out.mkdir(parents=True)
    old,new = (ResidualPPO.load(HISTORY/f"model-{n}.zip",device="cpu") for n in (0,250))
    bc,ref = load_policy(),model_at(ROOT.parent/"p3c-10.5/uniform/model.pt")
    audit=rollout_audit(old,new)
    audit["cases"]={str(c[2]):events(trace(out,c,bc,ref,old,new)) for c in panels() if c[1]==17218}
    (out/"audit.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    main(p.parse_args().output)
