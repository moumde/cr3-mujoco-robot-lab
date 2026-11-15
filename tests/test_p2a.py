from collections import Counter

import mujoco
import numpy as np

from cr3_lab.perception.camera import calibration, project
from cr3_lab.perception.dataset import CAMERAS, HEIGHT, WIDTH, cases
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, SpawnRange, load_scene, sample_initial_xy


def test_camera_extrinsics_and_projection():
    model = load_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in CAMERAS:
        camera = calibration(model, data, name, WIDTH, HEIGHT)
        rotation = np.asarray(camera["R_world_to_optical"])
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotation), 1)
        np.testing.assert_allclose(
            rotation @ data.cam(name).xpos + camera["t_world_to_optical_m"], 0, atol=1e-12)
        lookat = getattr(DEFAULT_SCENE, f"{name}_lookat_m")
        uv, depth = project([lookat], camera)
        assert depth[0] > 0
        np.testing.assert_allclose(uv[0], [(WIDTH - 1) / 2, (HEIGHT - 1) / 2], atol=1e-10)
        right = rotation[0] * 0.01
        displaced, _ = project([np.asarray(lookat) + right], camera)
        assert displaced[0, 0] > uv[0, 0]


def test_dataset_split_and_explicit_boundaries():
    positions = {}
    seeds = {}
    boundary = Counter()
    for split, profile, seed, case, xy in cases():
        if xy is None:
            xy = sample_initial_xy(DEFAULT_SCENE, seed, profile)
        assert positions.setdefault(tuple(xy), split) == split
        assert seeds.setdefault(seed, split) == split
        if case.startswith(("corner", "edge_midpoint")):
            assert split == "test"
            offset = abs(xy - np.asarray(DEFAULT_SCENE.initial_xy_m))
            count = np.isclose(offset, profile.half_width_m).sum()
            assert count == (2 if case.startswith("corner") else 1)
            boundary[profile, count] += 1
    for profile in (SpawnRange.SMALL, SpawnRange.LARGE):
        assert boundary[profile, 1] == boundary[profile, 2] == 4
