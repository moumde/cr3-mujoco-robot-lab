"""Position-conditioned behavior cloning under the P3c-5 benchmark."""
import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
import torch

from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation, SpawnRange, sample_initial_xy
from .behavior_clone import errors, model_for
from .check_grasp_lift import expert_actions
from .grasp_benchmark import execute, record_reference
from .grasp_lift import MotorGraspLiftEnv
from .train_grasp_lift import SCALE, summarize, write

TRAIN = range(16000, 16064)
VALIDATION = range(16100, 16116)
TEST = range(16200, 16220)
PROFILE = SpawnRange.LARGE
EPOCHS = 100


def xy(seed):
    return sample_initial_xy(DEFAULT_SCENE, seed, PROFILE)


def check_splits():
    # Treat positions within the same micrometer bin as identical.
    groups = [{tuple(np.rint(xy(seed) * 1e6).astype(int)) for seed in split}
              for split in (TRAIN, VALIDATION, TEST)]
    if any(len(group) != len(split) for group, split in zip(groups, (TRAIN, VALIDATION, TEST))):
        raise ValueError("Repeated XY within split")
    if any(a & b for a, b in combinations(groups, 2)):
        raise ValueError("XY leakage between splits")


def capture(output):
    check_splits()
    output.mkdir(parents=True, exist_ok=False)
    env = MotorGraspLiftEnv(PROFILE)
    episodes, observations, actions = [], [], []
    offsets = [0]
    for split, seeds in (("train", TRAIN), ("validation", VALIDATION)):
        for seed, orientation in product(seeds, Orientation):
            def record(obs, action):
                observations.append(obs.copy())
                actions.append(np.asarray(action, dtype=np.float32))

            _, info = execute(env, seed, orientation, lambda: expert_actions(env),
                              on_action=record)
            offset = (xy(seed) - DEFAULT_SCENE.initial_xy_m) * 1000
            episodes.append({**info, "split": split, "spawn_xy_m": xy(seed).tolist(),
                             "offset_mm": offset.tolist()})
            offsets.append(len(actions))
    env.close()
    write(output / "episodes.json", episodes)
    np.savez_compressed(output / "demonstrations.npz", observation=np.asarray(observations),
                        action=np.asarray(actions), offsets=offsets)
    audit(output)


def audit(output):
    episodes = json.loads((output / "episodes.json").read_text())
    with np.load(output / "demonstrations.npz", allow_pickle=False) as data:
        obs, act, offsets = (data[key] for key in ("observation", "action", "offsets"))
    report = {}
    for split in ("train", "validation"):
        indexes = [i for i, row in enumerate(episodes) if row["split"] == split]
        rows = [episodes[i] for i in indexes]
        coordinates = np.asarray([r["offset_mm"] for r in rows])
        phases = dict.fromkeys(("approach_xy", "descend", "close", "lift", "hold"), 0)
        for i in indexes:
            a = act[offsets[i]:offsets[i + 1]]
            phases["approach_xy"] += int(np.any(np.linalg.norm(a[:, :2], axis=1) > 1e-6))
            phases["descend"] += int(np.any(a[:, 2] < 0))
            phases["close"] += int(np.any(a[:, 3] > 0))
            phases["lift"] += int(np.any((a[:, 2] > 0) & (a[:, 3] > 0)))
            phases["hold"] += int(episodes[i]["is_success"])
        report[split] = {"trajectories_with_phase": phases,**summarize(rows), "distinct_xy": len({tuple(r["spawn_xy_m"]) for r in rows}),
                         "offset_min_mm": coordinates.min(0).tolist(),
                         "offset_max_mm": coordinates.max(0).tolist(),
                         "near_boundary_xy": len({tuple(r["spawn_xy_m"]) for r in rows
                                                  if max(map(abs, r["offset_mm"])) >= 20}),
                         "orientations": {o: sum(r["orientation"] == o for r in rows)
                                          for o in Orientation}}
    indexes = [i for i, r in enumerate(episodes) if r["split"] == "train"]
    parts = [slice(offsets[i], offsets[i + 1]) for i in indexes]
    aligned = min(p.stop - p.start for p in parts)
    simultaneous_x = np.stack([obs[p][:aligned] for p in parts])
    simultaneous_y = np.stack([act[p][:aligned] for p in parts])
    exact = [np.all(simultaneous_y[i] == simultaneous_y[j], axis=1)
             for i, j in combinations(range(len(parts)), 2)]
    # Bounded audit of neighboring physical states near action changes.
    sample = np.concatenate([p.start + np.union1d(np.arange(0, p.stop - p.start, 16),
        np.flatnonzero(np.max(abs(np.diff(act[p], axis=0)), axis=1) >= 0.5) + 1) for p in parts])
    x = (obs[sample] * SCALE).astype(np.float64)
    equal = (x * x).sum(1)
    dist = np.maximum(0, equal[:, None] + equal - 2 * x @ x.T)
    near = np.triu(dist <= 0.01**2, 1)
    i, j = np.where(near)
    report["diversity"] = {
        "aligned_steps": int(aligned), "aligned_action_exact_fraction": float(np.mean(exact)),
        "aligned_action_sequences_identical": sum(bool(a.all()) for a in exact),
        "trajectory_pairs": len(exact),
        "max_same_step_observation_span": np.ptp(simultaneous_x, axis=0).max(0).tolist(),
        "sampled_states": len(sample), "near_pairs_scaled_l2_le_0.01": len(i),
        "action_conflicts_ge_0.5": int(np.sum(np.max(abs(act[sample[i]] - act[sample[j]]), axis=1) >= 0.5)),
    }
    write(output / "audit.json", report)
    print(json.dumps(report))


def train(output):
    episodes = json.loads((output / "episodes.json").read_text())
    with np.load(output / "demonstrations.npz", allow_pickle=False) as data:
        obs, act, offsets = (data[key] for key in ("observation", "action", "offsets"))
    datasets = {}
    switches = {}
    for split in ("train", "validation"):
        indices, changes = [], []
        for i, row in enumerate(episodes):
            if row["split"] != split or not row["is_success"]:
                continue
            first, last = offsets[i:i + 2]
            indices.extend(range(first, last))
            change = np.r_[False, np.max(abs(np.diff(act[first:last], axis=0)), axis=1) >= 0.5]
            changes.extend(change)
        datasets[split] = (obs[indices] * SCALE, torch.from_numpy(act[indices]))
        switches[split] = np.asarray(changes)
    train_x, train_y = datasets["train"]
    center = train_x.mean(0)
    spread = np.maximum(train_x.std(0), 0.02)
    x = {s: torch.from_numpy(np.clip((v[0] - center) / spread, -8, 8)) for s, v in datasets.items()}
    val_y = datasets["validation"][1]
    torch.manual_seed(11)
    model = model_for(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    best, chosen = float("inf"), 0
    log = []
    for epoch in range(1, EPOCHS + 1):
        for idx in torch.randperm(len(train_y)).split(256):
            loss = (model(x["train"][idx]) - train_y[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mse = float((model(x["validation"]) - val_y).square().mean())
        log.append({"epoch": epoch, "validation_mse": mse})
        if mse < best:
            best, chosen = mse, epoch
            torch.save({"weights": model.state_dict(), "center": torch.from_numpy(center),
                        "spread": torch.from_numpy(spread)}, output / "model.pt")
        if epoch - chosen >= 15:
            break
    model.load_state_dict(torch.load(output / "model.pt", weights_only=True)["weights"])
    with torch.no_grad():
        metrics = {s: {**errors(model(x[s]), y), "steps": len(y),
                       "transition": errors(model(x[s][switches[s]]), y[switches[s]]),
                       "transition_steps": int(switches[s].sum())}
                   for s, (_, y) in datasets.items()}
    write(output / "training.json", {"best_epoch": chosen, "history": log, "metrics": metrics})
    print(json.dumps({"best_epoch": chosen, "metrics": metrics}))


def evaluate(output):
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    model = model_for(True)
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    center, spread = (checkpoint[k].numpy() for k in ("center", "spread"))
    reference, _ = record_reference()
    env = MotorGraspLiftEnv(PROFILE)
    pairs = []

    @torch.no_grad()
    def predict(obs):
        x = np.clip((obs * SCALE - center) / spread, -8, 8)
        return model(torch.from_numpy(x)).numpy()

    for seed, orientation in product(TEST, Orientation):
        oracle_initial, oracle = execute(env, seed, orientation, lambda: expert_actions(env))
        replay_initial, replay = execute(env, seed, orientation, lambda: reference)
        bc_initial, bc = execute(env, seed, orientation, policy=predict)
        np.testing.assert_array_equal(oracle_initial, replay_initial)
        np.testing.assert_array_equal(oracle_initial, bc_initial)
        offset = (xy(seed) - DEFAULT_SCENE.initial_xy_m) * 1000
        pairs.append({"seed": seed, "orientation": orientation, "offset_mm": offset.tolist(),
                      "region": "outer" if max(abs(offset)) >= 20 else "inner",
                      "oracle": oracle, "replay": replay, "bc": bc})
    env.close()
    groups = {"all": pairs, **{o: [p for p in pairs if p["orientation"] == o] for o in Orientation},
              **{r: [p for p in pairs if p["region"] == r] for r in ("inner", "outer")},
              "oracle_success_replay_failure": [p for p in pairs
                  if p["oracle"]["is_success"] and not p["replay"]["is_success"]]}
    stats = {name: {m: summarize([p[m] for p in group]) for m in ("oracle", "replay", "bc")}
             for name, group in groups.items() if group}
    write(output / "test.json", {"summary": stats, "pairs": pairs})
    print(json.dumps(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "audit", "train", "test"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"capture": capture, "audit": audit, "train": train, "test": evaluate}[args.command](args.output)


if __name__ == "__main__":
    main()
