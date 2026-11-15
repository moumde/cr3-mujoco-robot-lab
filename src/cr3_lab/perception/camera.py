"""Pinhole calibration and the RGB-to-planning result contract."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class VisualEstimate:
    """Estimated housing center in World meters and Inspection end-face class."""

    position_xy: tuple[float, float] | None
    orientation: Literal["protrusion_facing_camera", "flat_facing_camera"] | None
    confidence: float
    failure: str | None = None

    @property
    def valid(self):
        return self.failure is None


def calibration(model, data, name, width, height):
    camera = data.cam(name)
    # MuJoCo camera axes are right/up/back; optical axes are right/down/forward.
    rotation = np.diag([1, -1, -1]) @ camera.xmat.reshape(3, 3).T
    translation = -rotation @ camera.xpos
    focal = height / (2 * np.tan(np.deg2rad(model.camera(name).fovy[0]) / 2))
    return {
        "width": width, "height": height,
        "K": [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]],
        "R_world_to_optical": rotation.tolist(),
        "t_world_to_optical_m": translation.tolist(),
        "distortion": "none",
    }


def project(world_points, camera):
    points = (np.asarray(world_points) @ np.asarray(camera["R_world_to_optical"]).T
              + camera["t_world_to_optical_m"])
    pixels = points @ np.asarray(camera["K"]).T
    return pixels[:, :2] / pixels[:, 2:], points[:, 2]


def load_calibration(path):
    """Read the undistorted pinhole calibration exported by P2a."""
    cameras = json.loads(Path(path).read_text())
    for name in ("inspection", "localization"):
        camera = cameras[name]
        k = np.asarray(camera["K"])
        rotation = np.asarray(camera["R_world_to_optical"])
        translation = np.asarray(camera["t_world_to_optical_m"])
        if (k.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,)
                or not all(np.isfinite(a).all() for a in (k, rotation, translation))
                or camera["width"] < 1 or camera["height"] < 1 or camera["distortion"] != "none"
                or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
                or not np.allclose(rotation @ rotation.T, np.eye(3))
                or not np.isclose(np.linalg.det(rotation), 1)):
            raise ValueError(f"Invalid pinhole calibration for {name}")
    return cameras


def unproject_to_z(pixel, world_z_m, camera):
    """Intersect a pixel ray with a horizontal World plane."""
    rotation = np.asarray(camera["R_world_to_optical"])
    eye = -rotation.T @ camera["t_world_to_optical_m"]
    ray = rotation.T @ np.linalg.solve(camera["K"], [*pixel, 1])
    if abs(ray[2]) < 1e-10:
        raise ValueError("Pixel ray is parallel to the horizontal plane")
    distance = (world_z_m - eye[2]) / ray[2]
    if distance <= 0:
        raise ValueError("Plane intersection is behind the camera")
    return eye + distance * ray
