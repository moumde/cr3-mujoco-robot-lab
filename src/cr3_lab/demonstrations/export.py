"""Write and verify local LeRobot v3 datasets with the installed LeRobot API."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION

FPS = 10
CAMERAS = ("observation.images.inspection_rgb", "observation.images.localization_rgb")
REPO = "local/cr3-motor"


def rows(folder):
    return [json.loads(line) for line in (folder / "control.jsonl").read_text().splitlines()]


def export(raw, out):
    if version("lerobot") != "0.7.0" or CODEBASE_VERSION != "v3.0":
        raise RuntimeError("Verified export requires LeRobot 0.7.0 / v3.0")
    schema = json.loads((raw / "schema.json").read_text())
    features = {key: dict(dtype="video", shape=(400, 600, 3), names=["height", "width", "channels"])
                for key in CAMERAS}
    features.update({
        "observation.state": dict(dtype="float32", shape=(20,), names=schema["state"]),
        "action": dict(dtype="float32", shape=(7,), names=schema["action"]),
    })
    manifest = json.loads((raw / "manifest.json").read_text())
    out.mkdir(parents=True)
    for split in ("train", "validation"):
        group = [r for r in manifest if r["split"] == split]
        if not group:
            continue
        # Separate local datasets let the standard loader open either split directly.
        dataset = LeRobotDataset.create(repo_id=f"{REPO}-{split}", fps=FPS, features=features,
            root=out / split, robot_type="dobot_cr3_sim", video_backend="pyav", encoder_threads=2)
        for index, episode in enumerate(group):
            episode["lerobot_episode_index"] = index
            folder = raw / episode["id"]
            control = rows(folder)
            with np.load(folder / "frames.npz") as frames:
                for rgb, control_idx in zip(frames["rgb"], frames["control_index"], strict=True):
                    tick = control[control_idx]
                    dataset.add_frame({"task": episode["task"],
                                       "observation.state": np.asarray(tick["state"], dtype=np.float32),
                                       "action": np.asarray(tick["action"], dtype=np.float32),
                                       **dict(zip(CAMERAS, rgb, strict=True))})
            dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify(raw, out):
    manifest = json.loads((out / "manifest.json").read_text())
    summary = {}
    for split in ("train", "validation"):
        group = [r for r in manifest if r["split"] == split]
        if not group:
            continue
        dataset = LeRobotDataset(f"{REPO}-{split}", root=out / split, video_backend="pyav")
        assert dataset.fps == FPS
        assert len(dataset) == sum(e["frames"] for e in group)
        i = 0
        for episode in group:
            folder = raw / episode["id"]
            control = rows(folder)
            with np.load(folder / "frames.npz") as frames:
                for j, (rgb, control_idx) in enumerate(zip(
                        frames["rgb"], frames["control_index"], strict=True)):
                    frame = dataset[i]
                    assert int(frame["episode_index"]) == episode["lerobot_episode_index"]
                    assert int(frame["frame_index"]) == j
                    assert abs(float(frame["timestamp"]) - control[control_idx]["time_s"]) < 2e-6
                    assert frame["task"] == episode["task"]
                    for k, raw_key in (("action", "action"), ("observation.state", "state")):
                        np.testing.assert_array_equal(frame[k], np.asarray(control[control_idx][raw_key], np.float32))
                    for camera, original in zip(CAMERAS, rgb, strict=True):
                        decoded = frame[camera].permute(1, 2, 0).numpy()
                        assert decoded.shape == (400, 600, 3) and np.isfinite(decoded).all()
                        assert np.mean(np.abs(decoded - original / 255.0)) < .05
                    i += 1
        for path in (out / split / "videos").rglob("*.mp4"):
            with av.open(str(path)) as video:
                stream = video.streams.video[0]
                assert stream.average_rate == FPS and (stream.width, stream.height) == (600, 400)
        summary[split] = dict(episodes=len(group), frames=i, fps=FPS,
                              failures=sum(not r["success"] for r in group))
    (out / "loader-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "verify"))
    parser.add_argument("raw", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    (export if args.mode == "export" else verify)(args.raw, args.dataset)


if __name__ == "__main__":
    main()
