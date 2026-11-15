"""Read-only checkpoint, replay and matched-scenario audit."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-10"))
from smoke import BoundedWarmupSAC, ChallengeEnv, XYZResidual, ZERO, distributions, load_policy
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation
from cr3_lab.rl.train_grasp_lift import summarize

FAILED = {17208, 17211, 17218}


def manifest():
    rows = json.loads((ROOT.parent / "p3c-7/results.json").read_text())["ranges"]["35"]["pairs"]
    assert len(rows) == 40 and len({p["seed"] for p in rows}) == 20
    assert all(17200 <= p["seed"] < 17220 for p in rows)
    for seed in {p["seed"] for p in rows}:
        pair = [p for p in rows if p["seed"] == seed]
        assert {p["orientation"] for p in pair} == set(Orientation) and len(pair) == 2
        assert pair[0]["spawn_xy_m"] == pair[1]["spawn_xy_m"]
    return rows


def load(seed, steps):
    return BoundedWarmupSAC.load(ROOT.parent / f"p3c-10/seed-{seed}/model-{steps}.zip", device="cpu")


def evaluate(seed, steps):
    model = load(seed, steps) if seed else None
    scale = json.loads((ROOT.parent / "p3c-10/seed-11/config.json").read_text())["scale"]
    env = XYZResidual(ChallengeEnv(), load_policy(), scale)
    results = []
    cases = manifest()
    cases += [{"seed": 15000, "orientation": o, "spawn_xy_m": DEFAULT_SCENE.initial_xy_m}
              for o in Orientation]
    for case in cases:
        scene, orientation, xy = (case[k] for k in ("seed", "orientation", "spawn_xy_m"))
        env.unwrapped.xy = np.asarray(xy)
        obs, _ = env.reset(seed=scene, options={"scene_seed": scene, "orientation": orientation})
        rows, total = [], 0.
        initial = obs.tolist()
        initial_tcp = np.asarray(env.unwrapped.tcp_target).copy()
        while True:
            action = model.predict(obs, deterministic=True)[0] if model else ZERO
            obs, reward, terminated, truncated, info = env.step(action)
            control = info["residual_control"]
            rows.append(control)
            total += reward
            if terminated or truncated:
                if not seed and "bc" in case:
                    for key in ("steps", "reason", "first_lift_step", "first_bilateral_step"):
                        assert info[key] == case["bc"][key], (scene, orientation, key)
                break
        results.append({**info, "seed": scene, "xy": xy, "initial_obs": initial,
                        "reward": total, "first_action": rows[0],
                        "ik_rejections": sum(r["ik_status"] != "accepted" for r in rows),
                        "min_distance_mm": 1000 * min(r["grasp_distance_m"] for r in rows),
                        "last_distance_mm": 1000 * rows[-1]["grasp_distance_m"],
                        "tcp_path_mm": 1000 * float(np.linalg.norm(
                            [r["measured_tcp_delta_m"] for r in rows], axis=1).sum()),
                        "target_z_change_mm": 1000 * (env.unwrapped.tcp_target[2] - initial_tcp[2]),
                        "mean_action_delta": np.mean(abs(np.array([r["final_action"] for r in rows])
                                                        - np.array([r["bc_action"] for r in rows])), axis=0).tolist(),
                        "distribution": distributions(rows)})
    env.close()
    groups = {"all35": [r for r in results if r["seed"] != 15000],
              "nominal": [r for r in results if r["seed"] == 15000],
              "failed": [r for r in results if r["seed"] in FAILED],
              "other": [r for r in results if r["seed"] not in FAILED | {15000}]}
    groups.update({str(o): [r for r in groups["all35"] if r["orientation"] == o] for o in Orientation})
    summary = {}
    for name, rows in groups.items():
        ik = Counter()
        for row in rows:
            ik.update(row["distribution"]["ik"])
        summary[name] = {**summarize(rows), "ik": dict(ik),
                         "collision": dict(Counter(r["collision"] for r in rows if r["collision"])),
                         **{key: float(np.mean([r[key] for r in rows]))
                            for key in ("min_distance_mm", "last_distance_mm", "tcp_path_mm",
                                        "target_z_change_mm")}}
    (ROOT / f"eval-{seed}-{steps}.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(seed, steps, summary["all35"]["held"], summary["nominal"]["held"], flush=True)


def stage(obs):
    if obs[33] > 0:
        return "holding"
    if obs[34] and obs[27] >= 0.10:
        return "100mm"
    if obs[34]:
        return "confirmed"
    return "bilateral" if obs[31] else "before_bilateral"


def describe(tensor):
    return np.percentile(tensor.detach().cpu().numpy(), [0, 5, 50, 95, 100], axis=0).tolist()


def gradient_probe(model, obs, action, target):
    """Snapshot gradient norms on a fixed prefix batch, without optimizer state changes."""
    q = torch.cat(model.critic(obs, action), 1)
    critic_loss = (q - target[:, None]).square().mean()
    sampled, log_prob = model.actor.action_log_prob(obs)
    q_policy = torch.cat(model.critic(obs, sampled), 1).min(1).values
    actor_loss = (model.log_ent_coef.detach().exp() * log_prob - q_policy).mean()
    result = {}
    for name, loss, network in (("actor", actor_loss, model.actor),
                                ("critic", critic_loss, model.critic)):
        gradients = torch.autograd.grad(loss, tuple(network.parameters()))
        result[name] = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
    return result


def replay_audit(seed):
    source = ROOT.parent / f"p3c-10/seed-{seed}"
    final = load(seed, 5000)
    final.load_replay_buffer(source / "replay.pkl")
    buf = final.replay_buffer
    with gzip.open(source / "online.jsonl.gz", "rt") as stream:
        online = list(map(json.loads, stream))
    n = buf.size()
    assert n == len(online) == 5000
    np.testing.assert_allclose(buf.actions[:n, 0], [r["residual_control"]["residual_action"] for r in online], atol=1e-7)
    obs = torch.as_tensor(buf.observations[:n, 0])
    nxt = torch.as_tensor(buf.next_observations[:n, 0])
    action = torch.as_tensor(buf.actions[:n, 0])
    reward = torch.as_tensor(buf.rewards[:n, 0])
    done = torch.as_tensor(buf.dones[:n, 0] * (1 - buf.timeouts[:n, 0]))
    ends = [(i, r["reason"]) for i, r in enumerate(online) if r["reason"] is not None]
    np.testing.assert_array_equal(np.flatnonzero(buf.dones[:n, 0]), [i for i, _ in ends])
    np.testing.assert_array_equal(np.flatnonzero(buf.timeouts[:n, 0]), [i for i, why in ends if why == "timeout"])
    labels = np.asarray([stage(o) for o in obs])
    successor = np.asarray([stage(o) for o in nxt])
    episodes = []
    start = 0
    for i, reason in ends:
        episodes.append({"start": start, "end": i + 1, "reason": reason, "steps": i + 1 - start,
                         "return": float(reward[start:i + 1].sum())})
        start = i + 1
    report = {"prefixes": {}, "episodes": episodes, "training_log": [
        json.loads(line) for line in (source / "training.jsonl").read_text().splitlines()]}
    for steps in range(0, 5001, 1000):
        model = load(seed, steps)
        limit = steps or 500
        torch.manual_seed(1801)
        with torch.no_grad():
            # Fixed first 500 inputs for between-checkpoint Actor comparisons.
            mu, log_std, _ = model.actor.get_action_dist_params(obs[:500])
            next_action, log_prob = model.actor.action_log_prob(nxt[:limit])
            alpha = model.log_ent_coef.exp()
            q = torch.cat(model.critic(obs[:limit], action[:limit]), 1)
            target = reward[:limit] + model.gamma * (1 - done[:limit]) * (
                torch.cat(model.critic_target(nxt[:limit], next_action), 1).min(1).values - alpha * log_prob)
            error = q - target[:, None]
        grad = gradient_probe(model, obs[:128], action[:128], target[:128])
        by_stage = {}
        for name in sorted(set(labels[:limit])):
            mask = labels[:limit] == name
            by_stage[name] = {"count": int(mask.sum()), "q": describe(q[mask]),
                              "target": describe(target[mask]), "td_error": describe(error[mask])}
        report["prefixes"][steps] = {
            "observed_steps": limit, "prefix_note": "0k uses first 500 warmup transitions" if not steps else "",
            "actor_mu_pre_tanh": describe(mu), "actor_mean_action": describe(mu.tanh()),
            "actor_log_std": describe(log_std),
            "mean_action_saturation": (mu.tanh().abs() >= .95).float().mean(0).tolist(),
            "alpha": float(model.log_ent_coef.detach().exp()),
            "q": describe(q), "target_q": describe(target), "td_error": describe(error),
            "offline_critic_mse": float(error.square().mean()), "offline_gradient_l2": grad, "stages": by_stage,
            "successor_stages": dict(Counter(successor[:limit])),
            "completed": dict(Counter(reason for i, reason in ends if i < limit)),
            "episode_transition_counts": dict(Counter({
                reason: sum(e["steps"] for e in episodes if e["reason"] == reason and e["end"] <= limit)
                for reason in {e["reason"] for e in episodes}})),
            "terminated": int(done[:limit].sum()), "truncated": int(buf.timeouts[:limit, 0].sum()),
            "100mm_or_more_with_contact": int(((nxt[:limit, 27] >= .10) & (nxt[:limit, 31] > 0)
                                               & (nxt[:limit, 32] == 0)).sum()),
            "observed_online_distribution": distributions([r["residual_control"] for r in online[:limit]]),
        }
    (ROOT / f"replay-{seed}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "replay"))
    parser.add_argument("seed", type=int, choices=(0, 11, 22))
    parser.add_argument("--steps", type=int, choices=range(0, 5001, 1000), default=0)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.seed, args.steps)
    elif args.seed:
        replay_audit(args.seed)
    else:
        parser.error("replay requires seed 11 or 22")
