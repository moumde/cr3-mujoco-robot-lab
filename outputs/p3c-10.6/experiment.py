"""Diagnose residual regressions and apply the ordered gain/DAgger gate."""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "p3c-10.5"))
from distill import (BudgetedResidualEnv, ResidualMLP, ZERO, feedback, gate, load_data,
                     load_policy, metrics, panel, plant_for, rollout)
from dataset import phase, plan
from residual import TrainingChallenge
from summarize import summarize

GAINS = (.25, .5, .75, 1.)
DIAGNOSE = (17207, 17218, 17201)


def model_at(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualMLP(state["center"], state["spread"])
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def action(model, obs, gain):
    return (model(torch.from_numpy(obs)).numpy() * gain).astype(np.float32)


def passes(result):
    return result["preserved"] == 34 and result["nominal"] == 2 and (
        result["rescued"] >= 1 and result["new_contact"] == 0)


def diagnostics(out, cases, bc, uniform, balanced):
    ends = []
    for case in cases:
        _, seed, orientation, xy, _ = case
        if seed not in DIAGNOSE or orientation != "flat_facing_camera":
            continue
        initial = None
        for mode, model in (("frozen", None), ("teacher", None),
                            ("uniform", uniform), ("phase_balanced", balanced)):
            env = BudgetedResidualEnv(plant_for(case[0], xy), bc)
            try:
                obs, _ = env.reset(seed=seed, options=dict(scene_seed=seed, orientation=orientation))
                if initial is None:
                    initial = obs.copy()
                np.testing.assert_array_equal(obs, initial)
                with gzip.open(out / f"{seed}-{mode}.jsonl.gz", "xt") as stream:
                    while True:
                        teacher = feedback(obs, env.base_action())
                        student = action(model, obs, 1.) if model is not None else None
                        command = teacher if mode == "teacher" else ZERO if model is None else student
                        after, reward, term, trunc, info = env.step(command)
                        record = dict(obs=obs.tolist(), next_obs=after.tolist(), phase=phase(obs),
                                      next_phase=phase(after), residual_gain=0 if mode == "frozen" else 1,
                                      teacher=teacher.tolist(), student=None if student is None else student.tolist(),
                                      reward=reward, terminated=term, truncated=trunc, **info)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        obs = after
                        if term or trunc:
                            ends.append(dict(mode=mode, **info))
                            break
            finally:
                env.close()
    (out / "diagnostic-ends.json").write_text(json.dumps(ends, indent=2) + "\n")


def evaluate(out, name, cases, reference, bc, model, gain):
    results = []
    with gzip.open(out / f"{name}.jsonl.gz", "xt") as stream:
        for case, prior in zip(cases, reference, strict=True):
            profile, seed, orientation, xy, _ = case
            assert (profile, seed, orientation, xy) == tuple(
                prior[k] for k in ("profile", "seed", "orientation", "xy"))
            obs, info = rollout(case, bc, "residual", lambda obs: action(model, obs, gain), stream)
            np.testing.assert_array_equal(obs, np.asarray(prior["initial_observation"], dtype=np.float32))
            results.append(dict(prior, uniform=prior["residual"], residual=info))
    result = dict(gate=gate(results), metrics=summarize([r["residual"] for r in results]), rows=results)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, result["gate"], flush=True)
    return result


def collect(out, bc, uniform):
    observations, labels = [], []
    episodes = []
    env = BudgetedResidualEnv(TrainingChallenge(), bc)
    try:
        with gzip.open(out / "dagger-data.jsonl.gz", "xt") as stream:
            for seed, orientation, split in plan():
                if split != "train":
                    continue
                obs, _ = env.reset(seed=seed, options=dict(scene_seed=seed, orientation=orientation))
                rows = []
                while True:
                    student = action(uniform, obs, 1.)
                    # The raw teacher has no mutable state. Its snapshot contains
                    # a copy of the student's exposed ledger, with no shadow step.
                    teacher = feedback(obs.copy(), env.base_action())
                    before = obs
                    obs, reward, term, trunc, info = env.step(student)
                    rows.append(dict(obs=before.tolist(), next_obs=obs.tolist(),
                                     teacher=teacher.tolist(), student=student.tolist(),
                                     difference=(teacher - student).tolist(),
                                     phase=phase(before), label_valid=True, reward=reward,
                                     terminated=term, truncated=trunc, **info))
                    observations.append(before)
                    labels.append(teacher)
                    if term or trunc:
                        break
                for row in rows:
                    stream.write(json.dumps(dict(row, episode_success=info["is_success"]),
                                            allow_nan=False) + "\n")
                episodes.append(dict(seed=seed, orientation=str(orientation), steps=len(rows),
                                     success=info["is_success"], reason=info["reason"]))
    finally:
        env.close()
    (out / "dagger-manifest.json").write_text(json.dumps(episodes, indent=2) + "\n")
    return torch.tensor(np.stack(observations)), torch.tensor(np.stack(labels)), episodes


def aggregate(out, uniform, dagger):
    torch.manual_seed(11)
    groups = load_data()
    original_obs, original_labels, _ = groups["train"]
    dagger_obs, dagger_labels, episodes = dagger
    model = ResidualMLP(uniform.center.clone(), uniform.spread.clone())
    model.load_state_dict(uniform.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best_loss, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, 101):
        for _ in range((max(len(original_obs), len(dagger_obs)) + 127) // 128):
            a = torch.randint(len(original_obs), (128,))
            b = torch.randint(len(dagger_obs), (128,))
            obs = torch.cat((original_obs[a], dagger_obs[b]))
            labels = torch.cat((original_labels[a], dagger_labels[b]))
            loss = (model(obs) - labels).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        row = dict(epoch=epoch, **{key: metrics(model, value) for key, value in groups.items()})
        with torch.no_grad():
            row["dagger_mae"] = (model(dagger_obs) - dagger_labels).abs().mean(dim=0).tolist()
        history.append(row)
        if row["validation"]["loss"] < best_loss:
            best_loss, best_epoch = row["validation"]["loss"], epoch
            torch.save(model.state_dict(), out / "dagger.pt")
        if epoch - best_epoch >= 15:
            break
    result = dict(history=history, best_epoch=best_epoch, original_transitions=len(original_obs),
                  dagger_transitions=len(dagger_obs), student_success=sum(e["success"] for e in episodes))
    (out / "dagger-training.json").write_text(json.dumps(result, indent=2) + "\n")
    return model_at(out / "dagger.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    bc = load_policy()
    prior = ROOT.parent / "p3c-10.5"
    uniform = model_at(prior / "uniform/model.pt")
    balanced = model_at(prior / "phase_balanced/model.pt")
    reference = json.loads((prior / "uniform/evaluation.json").read_text())["rows"]
    cases = [c for c in panel() if c[0] in ("fixed", "35mm")]
    diagnostics(out, cases, bc, uniform, balanced)
    summary = {}
    for gain in GAINS:
        result = evaluate(out, f"gain-{gain:.2f}", cases, reference, bc, uniform, gain)
        summary[str(gain)] = {k: result[k] for k in ("metrics", "gate")}
    selected = next((gain for gain in GAINS if passes(summary[str(gain)]["gate"])), None)
    if selected is None:
        student = aggregate(out, uniform, collect(out, bc, uniform))
        result = evaluate(out, "dagger", cases, reference, bc, student, 1.)
        summary["dagger"] = {k: result[k] for k in ("metrics", "gate")}
    (out / "summary.json").write_text(json.dumps(dict(selected_gain=selected, results=summary),
                                                 indent=2) + "\n")


if __name__ == "__main__":
    main()
