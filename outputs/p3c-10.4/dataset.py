"""Collect, read and replay Feedback Teacher transitions without training."""
import argparse
from collections import Counter
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
for stage in ("p3c-10.3", "p3c-9"):
    sys.path.insert(0, str(ROOT.parent / stage))
from residual import TrainingChallenge, load_policy, sample_xy
from validate import feedback
from cr3_lab.rl.residual_budget import BudgetedResidualEnv, DEFAULT_BUDGET_M
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation


def plan():
    orientations = tuple(Orientation)
    for seed in range(18000, 18100):
        yield seed, orientations[seed % 2], "train" if seed < 18080 else "validation"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def phase(obs):
    if obs[33] > 0:
        return "hold"
    if obs[34]:
        return "lift"
    if obs[31]:
        return "bilateral"
    if obs[30] > .1:
        return "closing"
    return "approach" if np.linalg.norm(obs[12:15]) >= .065 else "near"


def transition(env, obs, action):
    after, reward, term, trunc, info = env.step(action)
    trace = info["residual_budget"]
    row = dict(observation_40=obs[:40].tolist(), budget_state_3=obs[40:].tolist(),
               observation_rl_43=obs.tolist(), next_observation_rl_43=after.tolist(),
               bc_action_3=trace["bc_action"][:3], teacher_residual_raw_3=trace["raw_residual"],
               teacher_residual_scaled_3=trace["scaled_residual"],
               budget_projected_action_3=trace["final_action"][:3],
               accepted_target_delta_xyz=trace["accepted_residual_target_delta_m"],
               phase=phase(obs), bilateral=bool(obs[31]), confirmed=bool(obs[34]),
               height_m=float(obs[27]), reward=reward, terminated=term, truncated=trunc,
               info=info)
    return after, row


def read_episode(root, case):
    path = root / case["file"]
    if digest(path) != case["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def collect(root, bc):
    root.mkdir(parents=True)
    env = BudgetedResidualEnv(TrainingChallenge(), bc)
    manifest = []
    try:
        for seed, orientation, split in plan():
            obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
            rows = []
            while True:
                obs, row = transition(env, obs, feedback(obs, env.base_action()))
                rows.append(row)
                if row["terminated"] or row["truncated"]:
                    break
            info = row["info"]
            name = f"{seed}.jsonl.gz"
            with gzip.open(root / name, "xt") as stream:
                for record in rows:
                    record["episode_success"] = info["is_success"]
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            manifest.append(dict(seed=seed, orientation=str(orientation), split=split,
                                 spawn_xy_m=sample_xy(.035, seed).tolist(),
                                 settled_xy_m=rows[0]["observation_40"][15:17], file=name,
                                 sha256=digest(root / name), **{k: info[k] for k in
                                 ("steps", "is_success", "reason", "collision",
                                  "first_bilateral_step", "first_lift_step", "first_100mm_step",
                                  "hold_steps", "max_lift_m")}))
    finally:
        env.close()
    (root / "manifest.json").write_text(json.dumps({"schema": 1, "episodes": manifest}, indent=2) + "\n")
    files = [r["file"] for r in manifest] + ["manifest.json"]
    (root / "SHA256SUMS").write_text("".join(f"{digest(root / f)}  {f}\n" for f in files))


def quantiles(values):
    return np.quantile(values, [.5, .95, 1], axis=0).tolist()


def replay_and_audit(root, bc):
    manifest = json.loads((root / "manifest.json").read_text())["episodes"]
    assert [(r["seed"], Orientation(r["orientation"]), r["split"]) for r in manifest] == list(plan())
    assert len({tuple(r["spawn_xy_m"]) for r in manifest}) == 100
    env = BudgetedResidualEnv(TrainingChallenge(), bc)
    phases = {}
    subsets = {"all": manifest}
    total_steps = 0
    signed_edges = np.zeros(3, dtype=int)
    zeros = 0
    labels, accepted, consumption, measured = [], [], [], []
    ik = Counter()
    try:
        for case in manifest:
            seed, orientation = case["seed"], case["orientation"]
            rows = read_episode(root, case)
            assert len(rows) == case["steps"] <= 400
            obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "orientation": orientation})
            for step, row in enumerate(rows, 1):
                action = np.asarray(row["teacher_residual_raw_3"], dtype=np.float32)
                np.testing.assert_array_equal(action, feedback(obs, env.base_action()))
                obs, expected = transition(env, obs, action)
                assert row == {**expected, "episode_success": case["is_success"]}, (seed, step)
                assert (row["terminated"] or row["truncated"]) == (step == len(rows))
                trace = row["info"]["residual_budget"]
                phases.setdefault(row["phase"], []).append(action)
                signed_edges += np.isclose(np.abs(trace["signed_budget_m"]), DEFAULT_BUDGET_M,
                                           rtol=0, atol=1e-12)
                ik[trace["ik_status"]] += 1
                zeros += not np.any(action)
                labels.append(action)
                accepted.append(trace["accepted_residual_target_delta_m"])
                measured.append(trace["measured_tcp_delta_m"])
            assert rows[-1]["info"]["is_success"] == case["is_success"]
            total_steps += len(rows)
            consumption.append(trace["absolute_budget_m"])
            for key in (case["split"], orientation):
                subsets.setdefault(key, []).append(case)
            dx, dy = np.array(case["spawn_xy_m"]) - DEFAULT_SCENE.initial_xy_m
            subsets.setdefault(f"x{'+' if dx >= 0 else '-'}y{'+' if dy >= 0 else '-'}", []).append(case)
            subsets.setdefault("outer" if max(abs(dx), abs(dy)) >= .025 else "inner", []).append(case)
    finally:
        env.close()
    report = {
        "episodes": {key: dict(count=len(rows), successes=sum(r["is_success"] for r in rows),
                               length_p50_p95_max=quantiles([r["steps"] for r in rows]),
                               failures=dict(Counter(r["reason"] for r in rows if not r["is_success"])))
                     for key, rows in subsets.items()},
        "failures": [r for r in manifest if not r["is_success"]],
        "transitions": total_steps, "zero_vectors": zeros, "edge_steps_xyz": signed_edges.tolist(),
        "ik": dict(ik), "raw_signed_min_max_xyz": [np.min(labels, axis=0).tolist(),
                                                    np.max(labels, axis=0).tolist()],
        "raw_abs_p50_p95_max_xyz": quantiles(np.abs(labels)),
        "scaled_abs_p50_p95_max_xyz": quantiles(.5 * np.abs(labels)),
        "accepted_abs_mm_p50_p95_max_xyz": quantiles(1000 * np.abs(accepted)),
        "measured_abs_mm_p50_p95_max_xyz": quantiles(1000 * np.abs(measured)),
        "absolute_episode_mm_p50_p95_max_xyz": quantiles(1000 * np.asarray(consumption)),
        "phases": {key: dict(count=len(values), zero_vectors=sum(not np.any(v) for v in values),
                             raw_abs_p50_p95_max_xyz=quantiles(np.abs(values)))
                   for key, values in phases.items()},
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(total_steps, "transitions exactly replayed;", report["episodes"]["all"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "replay"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    (collect if args.mode == "collect" else replay_and_audit)(args.output, load_policy())


if __name__ == "__main__":
    main()
