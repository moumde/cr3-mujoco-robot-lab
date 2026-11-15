"""Train a residual MLP and evaluate the frozen policy on the development panel."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
for stage in ("p3c-10.3", "p3c-10.4"):
    sys.path.insert(0, str(ROOT.parent / stage))
from dataset import read_episode, digest, plan
from validate import feedback, load_policy, panel, plant_for, ZERO, write_step
from cr3_lab.rl.residual_budget import BudgetedResidualEnv
from cr3_lab.tasks.motor_scene import Orientation

DATA = ROOT.parent / "p3c-10.4/data"
PHASES = ("approach", "near", "closing", "bilateral", "lift", "hold")


class ResidualMLP(nn.Module):
    def __init__(self, center, spread):
        super().__init__()
        self.register_buffer("center", center)
        self.register_buffer("spread", spread)
        self.network = nn.Sequential(
            nn.Linear(43, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Tanh())

    def forward(self, obs):
        return self.network((obs - self.center) / self.spread)


def load_data():
    path = DATA / "manifest.json"
    assert digest(path) == digest(DATA.parent / "manifest.json")
    cases = json.loads(path.read_text())["episodes"]
    assert [(c["seed"], Orientation(c["orientation"]), c["split"]) for c in cases] == list(plan())
    groups = {"train": [], "validation": []}
    for case in cases:
        groups[case["split"]].extend(read_episode(DATA, case))
    return {split: (
        torch.tensor([r["observation_rl_43"] for r in rows]),
        torch.tensor([r["teacher_residual_raw_3"] for r in rows]),
        torch.tensor([PHASES.index(r["phase"]) for r in rows]),
    ) for split, rows in groups.items()}


@torch.no_grad()
def metrics(model, data):
    obs, labels, phases = data
    pred = model(obs)
    error = (labels - pred).abs()
    return dict(loss=error.square().mean().item(), mae=error.mean(dim=0).tolist(),
                saturation=(pred.abs() >= .95).float().mean(dim=0).tolist(),
                phase_mae={p: error[phases == i].mean(dim=0).tolist()
                           for i, p in enumerate(PHASES)})


def train(out, mode):
    out.mkdir(parents=True)
    torch.manual_seed(11)
    started = time.perf_counter()
    groups = load_data()
    obs, labels, phases = groups["train"]
    model = ResidualMLP(obs.mean(dim=0), obs.std(dim=0, unbiased=False).clamp_min(.001))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    weights = phases.bincount(minlength=len(PHASES)).float().reciprocal()[phases]
    best_loss, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, 101):
        order = (torch.randperm(len(obs)) if mode == "uniform" else
                 torch.multinomial(weights, len(obs), replacement=True))
        for idx in order.split(256):
            loss = nn.functional.mse_loss(model(obs[idx]), labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        row = {key: metrics(model, group) for key, group in groups.items()}
        history.append(dict(epoch=epoch, **row))
        if row["validation"]["loss"] < best_loss:
            best_loss, best_epoch = row["validation"]["loss"], epoch
            torch.save(model.state_dict(), out / "model.pt")
        if epoch - best_epoch >= 15:
            break
    result = dict(mode=mode, seed=11, epochs=epoch, best=history[best_epoch - 1],
                  transitions={key: len(group[0]) for key, group in groups.items()},
                  wall_time_s=time.perf_counter() - started, history=history)
    (out / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "curve.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_mse", "validation_mse"])
        for row in history:
            writer.writerow([row["epoch"], row["train"]["loss"], row["validation"]["loss"]])
    print(mode, "best epoch", best_epoch, "validation MSE", best_loss, flush=True)


def rollout(case, bc, method, predictor, stream):
    profile, seed, orientation, xy, historical = case
    env = BudgetedResidualEnv(plant_for(profile, xy), bc)
    options = dict(scene_seed=seed, orientation=orientation)
    try:
        obs, _ = env.reset(seed=seed, options=options)
        initial = obs.copy()
        effective = np.zeros(3, dtype=int)
        requested = np.zeros(3, dtype=int)
        saturation = np.zeros(3, dtype=int)
        peak = np.zeros(3)
        rejected = {}
        while True:
            action = predictor(obs)
            if method == "teacher":
                action = feedback(obs, env.base_action())
            obs, reward, term, trunc, info = env.step(action)
            trace = info["residual_budget"]
            raw = np.asarray(trace["raw_residual"])
            effective += np.abs(trace["accepted_residual_target_delta_m"]) > 1e-12
            requested += raw != 0
            saturation += np.abs(raw) >= .95
            peak = np.maximum(peak, np.abs(trace["signed_budget_m"]))
            status = trace["ik_status"]
            if status != "accepted":
                rejected[status] = rejected.get(status, 0) + 1
            write_step(stream, obs, reward, term, trunc, info, method=method, profile=profile)
            if term or trunc:
                if method == "frozen" and historical:
                    for key in ("is_success", "steps", "reason", "collision"):
                        assert info[key] == historical[key]
                result = {key: value for key, value in info.items() if key != "residual_budget"}
                return initial, dict(result, requested_xyz=requested.tolist(),
                                     effective_xyz=effective.tolist(), saturated_xyz=saturation.tolist(),
                                     peak_signed_m=peak.tolist(), absolute_m=trace["absolute_budget_m"],
                                     ik_rejections=rejected)
    finally:
        env.close()


def gate(rows):
    regular = [r for r in rows if r["profile"] == "35mm"]
    previous = [r for r in regular if r["frozen"]["is_success"]]
    failures = [r for r in regular if not r["frozen"]["is_success"]]
    nominal = [r for r in rows if r["profile"] == "fixed"]
    assert (len(previous), len(failures), len(nominal)) == (34, 6, 2)
    return dict(preserved=sum(r["residual"]["is_success"] for r in previous),
                nominal=sum(r["residual"]["is_success"] for r in nominal),
                rescued=sum(r["residual"]["is_success"] for r in failures),
                new_contact=sum(bool(r["residual"]["collision"]) and
                                r["residual"]["collision"] != r["frozen"]["collision"] for r in rows))


def evaluate(out, compare_baseline=True):
    bc = load_policy()
    state = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model = ResidualMLP(state["center"], state["spread"])
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def predict(obs):
        return model(torch.from_numpy(obs)).numpy()

    cached = None if compare_baseline else json.loads((out.parent / "uniform/evaluation.json").read_text())["rows"]
    rows = []
    started = time.perf_counter()
    with gzip.open(out / "rollouts.jsonl.gz", "xt") as stream:
        cases = [c for c in panel() if c[0] in ("fixed", "35mm")]
        for case in cases:
            profile, seed, orientation, xy, _ = case
            row = dict(profile=profile, seed=seed, orientation=str(orientation), xy=xy)
            original = None
            if cached is not None:
                prior = cached[len(rows)]
                assert all(prior[k] == v for k, v in row.items())
                row.update({key: prior[key] for key in ("frozen", "teacher", "initial_observation")})
                original = np.asarray(prior["initial_observation"], dtype=np.float32)
            methods = [("residual", predict)]
            if compare_baseline:
                methods = [("frozen", lambda _: ZERO), ("teacher", lambda _: ZERO), *methods]
            for method, policy in methods:
                initial, row[method] = rollout(case, bc, method, policy, stream)
                if original is None:
                    original = initial
                np.testing.assert_array_equal(initial, original)
            row["initial_observation"] = original.tolist()
            rows.append(row)
    gates = gate(rows)
    (out / "evaluation.json").write_text(json.dumps(
        dict(gate=gates, rows=rows, wall_time_s=time.perf_counter() - started), indent=2) + "\n")
    print(out.name, gates, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("uniform", "phase_balanced"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.root / args.mode
    train(path, args.mode)
    evaluate(path, compare_baseline=args.mode == "uniform")
