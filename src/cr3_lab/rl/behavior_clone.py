"""Supervised Grasp & Lift cloning and closed-loop evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .expert_replay import collect as collect_demos, verify
from .train_grasp_lift import SCALE, TEST_SEEDS, collect, summarize

TRAIN_TRAJECTORIES = 6
EPOCHS = 400


def model_for(clock):
    return nn.Sequential(nn.Linear(40 if clock else 39, 64), nn.Tanh(),
                         nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4), nn.Tanh())


def features(scaled, clock):
    return scaled if clock else np.delete(scaled, 36, axis=-1)


def ambiguity(x, y, offsets, radius=0.01):
    """Find close states with conflicting labels across distinct phase times."""
    times = np.concatenate([np.arange(n) for n in np.diff(offsets)])
    distance = np.maximum(0, (x * x).sum(1)[:, None] + (x * x).sum(1) - 2 * x @ x.T)
    near = (distance <= radius**2) & (times[:, None] != times)
    i, j = np.where(np.triu(near, 1))
    delta = np.max(abs(y[i] - y[j]), axis=1)
    conflicts = np.flatnonzero(delta >= 0.5)
    return {"radius_scaled_l2": radius, "near_pairs": len(i), "conflicts": len(conflicts),
            "examples": [{"steps": [int(times[i[k]]), int(times[j[k]])],
                          "actions": y[[i[k], j[k]]].tolist(),
                          "distance": float(np.sqrt(distance[i[k], j[k]]))}
                         for k in conflicts[:3]]}


def errors(pred, target):
    diff = (pred - target).detach().numpy()
    return {"mae": np.abs(diff).mean(0).tolist(),
            "rmse": np.sqrt(np.square(diff).mean(0)).tolist()}


def fit(data, clock, output):
    torch.manual_seed(11)
    scaled = data["observation"] * SCALE
    x = torch.from_numpy(features(scaled, clock))
    y = torch.from_numpy(data["action"])
    cut = int(data["offsets"][TRAIN_TRAJECTORIES])
    model = model_for(clock)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best = float("inf")
    history = []
    for epoch in range(1, EPOCHS + 1):
        for indexes in torch.randperm(cut).split(128):
            loss = (model(x[indexes]) - y[indexes]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            validation = float((model(x[cut:]) - y[cut:]).square().mean())
        if validation < best:
            best, best_epoch = validation, epoch
            torch.save(model.state_dict(), output / "model.pt")
        if epoch == 1 or epoch % 20 == 0:
            history.append({"epoch": epoch, "validation_mse": validation})
    model.load_state_dict(torch.load(output / "model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    result = {"clock": clock, "best_epoch": best_epoch, "history": history,
              "train": errors(prediction[:cut], y[:cut]),
              "validation": errors(prediction[cut:], y[cut:]),
              "ambiguity_train": ambiguity(x[:cut].numpy().astype(np.float64), y[:cut].numpy(),
                                           data["offsets"][:TRAIN_TRAJECTORIES + 1])}
    (output / "supervised.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--demos", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.demos:
        with np.load(args.demos, allow_pickle=False) as source:
            data = dict(source)
    else:
        data = collect_demos()
    verify(data, SCALE)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"train_seeds": data["seeds"][:TRAIN_TRAJECTORIES].tolist(),
               "validation_seeds": data["seeds"][TRAIN_TRAJECTORIES:].tolist(),
               "test_seeds": list(TEST_SEEDS), "steps": len(data["action"]), "variants": {}}
    for clock, name in ((True, "full"), (False, "without_clock")):
        output = args.output / name
        output.mkdir()
        result = fit(data, clock, output)
        model = model_for(clock)
        model.load_state_dict(torch.load(output / "model.pt", weights_only=True))
        model.eval()

        def policy(obs):
            with torch.no_grad():
                return model(torch.from_numpy(features(obs, clock))).numpy()

        rows = collect(policy, "pregrasp", TEST_SEEDS)
        result["closed_loop"] = summarize(rows)
        result["episodes"] = rows
        summary["variants"][name] = result
    expert = collect(None, "pregrasp", TEST_SEEDS)
    summary["expert"] = {"summary": summarize(expert), "episodes": expert}
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, result in summary["variants"].items():
        print(name, result["best_epoch"], result["validation"], result["closed_loop"])


if __name__ == "__main__":
    main()
