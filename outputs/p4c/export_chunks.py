"""Export complete projected actuator chunks via LeRobot 0.7.0."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.configs.video import RGBEncoderConfig
import numpy as np

from cr3_lab.demonstrations.chunks import CHUNK, FPS, control_rows, labels
from cr3_lab.demonstrations.export import CAMERAS

REPO = "local/cr3-bc"


def run(raw, out):
    assert version("lerobot") == "0.7.0" and CODEBASE_VERSION == "v3.0"
    schema = json.loads((raw / "schema.json").read_text())
    manifest = json.loads((raw / "manifest.json").read_text())
    assert len(manifest) == 38
    xy_splits = {}
    for episode in manifest:
        xyz = json.loads((raw / episode["id"] / "oracle-debug.json").read_text())["initial_xy_m"]
        key = tuple(np.round(xyz, 6))
        assert xy_splits.setdefault(key, episode["split"]) == episode["split"]
    features = {k: dict(dtype="video", shape=(400, 600, 3), names=["height", "width", "channels"])
                for k in CAMERAS}
    features.update({"observation.state": dict(dtype="float32", shape=(20,), names=schema["state"]),
                     "action": dict(dtype="float32", shape=(35,), names=[
                         f"tick{i}_{key}" for i in range(CHUNK) for key in schema["action"]]),
                     "phase_index": dict(dtype="int64", shape=(1,), names=None)})
    out.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        ds = LeRobotDataset.create(f"{REPO}-{split}", FPS, features, root=out / split,
                                   robot_type="dobot_cr3_sim", video_backend="pyav", encoder_threads=2,
                                   rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=20))
        group = [r for r in manifest if r["split"] == split and r["success"]]
        for i, episode in enumerate(group):
            folder = raw / episode["id"]
            control = control_rows(folder)
            with np.load(folder / "frames.npz") as data:
                indices = data["control_index"]
                rgb = data["rgb"]
                schedule = list(labels(control, indices))
                for j, (k, action, phase) in enumerate(schedule):
                    assert k == indices[j] == j * CHUNK
                    ds.add_frame(dict(task=episode["task"], action=action.reshape(35),
                                      phase_index=np.array([phase], np.int64),
                                      **{"observation.state": np.asarray(control[k]["state"], np.float32)},
                                      **dict(zip(CAMERAS, rgb[j], strict=True))))
            ds.save_episode(parallel_encoding=False)
            episode.update(dataset_episode=i, chunk_frames=len(schedule),
                           excluded_tail_frames=len(indices)-len(schedule))
        ds.finalize()
        check = LeRobotDataset(f"{REPO}-{split}", root=out / split, video_backend="pyav")
        assert len(check) == sum(e["chunk_frames"] for e in group)
        assert check.fps == 10
        print(split, len(group), len(check), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.raw, args.output)
