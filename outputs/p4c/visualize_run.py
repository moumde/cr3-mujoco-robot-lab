"""Plot measured results and render a reference episode from logged actuator controls."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cr3_lab.demonstrations.chunks import CHUNK, PHASES, control_rows
from cr3_lab.demonstrations.collect import INTEGRATION, scene
from cr3_lab.tasks.motor_expert import MotorExpert, DEFAULT_EXPERT


def replay(raw, identifier, output):
    folder = raw / identifier
    model = scene(False)
    robot = MotorExpert(model)
    data = mujoco.MjData(model)
    initial = np.load(folder / "initial.npy")
    mujoco.mj_setState(model, data, initial, INTEGRATION)
    # Restoring the integration state alone leaves derived rendering fields unset.
    # Populate a separate state for the first frame so forward dynamics never
    # changes the state used for the exact command replay.
    initial_render = mujoco.MjData(model)
    mujoco.mj_setState(model, initial_render, initial, INTEGRATION)
    mujoco.mj_forward(model, initial_render)
    state = np.empty_like(initial)
    frames = []
    with mujoco.Renderer(model, height=320, width=480) as renderer:
        for i, row in enumerate(control_rows(folder)):
            action = row["action"]
            if i % CHUNK == 0:
                renderer.update_scene(initial_render if i == 0 else data, "overview")
                image = Image.fromarray(renderer.render())
                ImageDraw.Draw(image).text((8, 8), f"Expert control replay | {row['phase']} | {data.time:.2f}s",
                                           fill="white")
                frames.append(image)
            robot.gripper.command(data, action[6])
            robot.arm.step(data, action[:6])
            mujoco.mj_getState(model, data, state, INTEGRATION)
            np.testing.assert_allclose(state, row["integration_after"], atol=1e-12, rtol=0)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0)


def plots(root):
    train = json.loads((root / "train/training.json").read_text())
    runs = json.loads((root / "eval/results.json").read_text())["rows"]
    manifest = json.loads((root / "dataset/manifest.json").read_text())
    fig, ax = plt.subplots()
    for mode in ("train", "validation"):
        ax.semilogy([r["epoch"] for r in train["history"]],
                    [r[mode]["mse"] for r in train["history"]], label=mode)
    ax.legend()
    ax.set(xlabel="Epoch", ylabel="Normalized MSE")
    fig.savefig(root / "loss.png", dpi=140)
    plt.close(fig)
    values = train["best"]["validation"]["mae"]
    fig, (joints, gripper) = plt.subplots(1, 2, figsize=(9, 4))
    joints.bar(range(1, 7), values[:6])
    joints.set(xlabel="Joint", ylabel="Validation MAE (rad)")
    gripper.bar(["Gripper"], [values[6]])
    gripper.set(ylabel="Closure MAE [0,1]")
    fig.tight_layout()
    fig.savefig(root / "action-error.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    # Scheduled phase-entry is a timing diagnostic. Physical achievements are separate.
    entries = [sum(p in row["phases"] for row in runs) for p in PHASES]
    milestone = [sum(bool(row[key]) for row in runs) for key in ("bilateral", "grasped", "stable", "success")]
    stage_pass = dict(zip(PHASES, entries))
    physical = dict(
        Grasp=milestone[0],
        Lift=sum(row["max_lift_m"] >= DEFAULT_EXPERT.min_lift_m for row in runs),
        Reorient=sum(row["grasped"] and not row["dropped"] and
                     row["angle_error_deg"] <= DEFAULT_EXPERT.angle_tolerance_deg for row in runs),
        Place=sum(row["stable"] and
                  row["position_error_m"] <= DEFAULT_EXPERT.position_tolerance_m for row in runs),
        Release=sum(row["open_seen"] and row["stable"] for row in runs),
        Retreat=milestone[-1],
    )
    stage_pass.update(physical)
    ax.bar(list(PHASES)+["Confirmed", "Stable", "Success"],
           list(stage_pass.values())+milestone[1:])
    ax.tick_params(axis="x", rotation=40)
    ax.set(ylabel="Episodes")
    fig.tight_layout()
    fig.savefig(root / "stages.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, marker in (("train", "o"), ("validation", "^"), ("test", "s")):
        points = [json.loads((root / "raw" / row["id"] / "oracle-debug.json").read_text())["initial_xy_m"]
                  for row in manifest if row["split"] == split]
        ax.scatter(*np.asarray(points).T, label=split, marker=marker)
    ax.legend()
    ax.set(xlabel="World X (m)", ylabel="World Y (m)")
    fig.tight_layout()
    fig.savefig(root / "scene-distribution.png", dpi=140)
    plt.close(fig)
    summary = dict(phase_entries=dict(zip(PHASES, entries)), total=len(runs), stage_metrics=stage_pass,
                   milestones=dict(zip(("bilateral", "grasped", "stable", "success"), milestone)),
                   data=dict(Counter(row["split"] for row in manifest)))
    (root / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args.root / "raw", "episode-000", args.root / "expert-replay.gif")
    else:
        plots(args.root)
