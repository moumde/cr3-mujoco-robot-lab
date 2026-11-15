"""Prepare compact training tensors by decoding the actual LeRobot datasets."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from cr3_lab.demonstrations.export import CAMERAS

SIZE = (48, 32)


def load(root, split):
    dataset = LeRobotDataset(f"local/cr3-bc-{split}", root=root / split, video_backend="pyav",
                              return_uint8=True)
    states, actions, phases, images = [], [], [], []
    for i in range(len(dataset)):
        frame = dataset[i]
        states.append(np.asarray(frame["observation.state"]))
        actions.append(np.asarray(frame["action"]))
        phases.append(int(frame["phase_index"]))
        images.append(np.concatenate([
            np.asarray(Image.fromarray(frame[key].permute(1, 2, 0).numpy()).resize(
                SIZE, Image.Resampling.BILINEAR)).transpose(2, 0, 1) for key in CAMERAS]))
        assert abs(float(frame["timestamp"]) - int(frame["frame_index"]) / 10) < 2e-6
    if split != "test":
        np.savez(root / f"{split}-cache.npz", state=states, action=actions, phase=phases, images=images)
    print(split, len(dataset), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for split in ("train", "validation", "test"):
        load(args.root, split)
