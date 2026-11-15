"""Offline gradients and command displacement from frozen artifacts."""
import gzip
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-10.1"))
from audit import load


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def stats(values):
    return np.round(np.quantile(np.abs(values), [0.5, 0.95, 1], axis=0), 5).tolist()


def actor_audit():
    observations = []
    for seed in (11, 22):
        with (ROOT.parent / f"p3c-10/seed-{seed}/replay.pkl").open("rb") as stream:
            observations.append(pickle.load(stream).observations[:500, 0])
    obs = torch.as_tensor(np.concatenate(observations))
    for seed in (11, 22):
        for step in (0, 1000, 5000):
            model = load(seed, step)
            actor = model.actor
            torch.manual_seed(902)
            mean, log_std, _ = actor.get_action_dist_params(obs)
            action, log_prob = actor.action_log_prob(obs)
            alpha = model.log_ent_coef.detach().exp().item()
            q = torch.cat(model.critic(obs, action), dim=1).min(dim=1).values
            entropy_loss, q_loss = alpha * log_prob.mean(), -q.mean()
            params = (*actor.mu.parameters(), *actor.log_std.parameters())
            entropy_grad = torch.autograd.grad(entropy_loss, params, retain_graph=True)
            q_grad = torch.autograd.grad(q_loss, params)
            yield {
                "seed": seed, "step": step, "alpha": alpha,
                "entropy": -log_prob.mean().item(),
                "loss_parts": [entropy_loss.item(), q_loss.item()],
                "mean_abs_quantiles": stats(mean.detach().numpy()),
                "median_log_std": log_std.detach().median(dim=0).values.tolist(),
                "residual_abs_quantiles": stats(action.detach().numpy()),
                "fraction_abs_residual_ge_095": (action.abs() >= .95).float().mean(dim=0).tolist(),
                "bias_grad_entropy_mu_logstd": [entropy_grad[i].tolist() for i in (1, 3)],
                "bias_grad_q_mu_logstd": [q_grad[i].tolist() for i in (1, 3)],
            }


def motion_audit():
    online = {}
    for seed in (11, 22):
        records = rows(
            ROOT.parent / f"p3c-10/seed-{seed}/online.jsonl.gz")
        controls = [r["residual_control"] for r in records]
        episodes = [r["steps"] for r in records]
        final = np.array([c["final_action"][:3] for c in controls])
        bc = np.array([c["bc_action"][:3] for c in controls])
        accepted = np.array([c["ik_status"] == "accepted" for c in controls])
        # This is action composition before workspace clipping, not a paired TCP difference.
        proxy = 1.5 * (final - bc) * accepted[:, None]
        cumulative = np.zeros(3)
        history = []
        for index, delta in enumerate(proxy):
            cumulative += delta
            history.append(cumulative.copy())
            if index + 1 < len(controls) and episodes[index + 1] == 1:
                cumulative[:] = 0
        online[seed] = {
            "composition_delta_mm": stats(1.5 * (final - bc)),
            "accepted_proxy_cumulative_mm": stats(history),
            "measured_tcp_delta_mm": stats([np.array(c["measured_tcp_delta_m"]) * 1000 for c in controls]),
            "clipped_fraction_any_axis": np.mean([np.any(c["composition_clipped"]) for c in controls]),
        }
    pairs = {}
    for path in sorted((ROOT.parent / "p3c-8").glob("*-bc.jsonl.gz")):
        bc = rows(path)
        oracle = rows(path.with_name(path.name.replace("-bc.", "-oracle.")))
        event = next((r for r in bc if r["ik"]["status"] != "accepted"), bc[-1])
        reference = oracle[min(event["step"] - 1, len(oracle) - 1)]
        pairs[path.name.split("-bc.")[0]] = {
            "spawn_offset_xy_mm": (1000 * (np.array(bc[0]["object_world_m"][:2]) - [.45, .09])).tolist(),
            "bc_end": bc[-1]["info"]["reason"], "event_step": event["step"],
            "bc_event_grasp_distance_mm": 1000 * event["grasp_distance_m"],
            "bc_event_grasp_vector_mm": (1000 * np.array(event["observation"][12:15])).tolist(),
            "oracle_event_grasp_vector_mm": (1000 * np.array(reference["observation"][12:15])).tolist(),
            "oracle_same_step_grasp_distance_mm": 1000 * reference["grasp_distance_m"],
            "bc_minus_oracle_same_step_tcp_mm": (1000 * (np.array(event["tcp_world_m"]) - reference["tcp_world_m"])).tolist(),
            "oracle_min_grasp_distance_mm": 1000 * min(r["grasp_distance_m"] for r in oracle),
        }
    return online, pairs


if __name__ == "__main__":
    torch.set_num_threads(1)
    online, pairs = motion_audit()
    assert len(pairs) == 6 and all(len(v) == 3 for row in pairs.values()
                                   for k, v in row.items() if k.endswith("vector_mm"))
    actor = list(actor_audit())
    for result in actor:
        assert np.isfinite(result["loss_parts"]).all()
        if result["step"] == 0:
            assert np.array_equal(result["median_log_std"], [-3, -3, -3])
            assert not np.any(result["mean_abs_quantiles"])
    print(json.dumps({"actor": actor, "online": online, "pairs": pairs}, indent=2))
