"""Color and calibrated cylinder geometry for the P2a synthetic RGB domain."""
import argparse
from dataclasses import asdict
import json

import numpy as np
from PIL import Image

from .camera import VisualEstimate, load_calibration, project, unproject_to_z


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class RgbMotorEstimator:
    def __init__(self, cameras, scene=None):
        if scene is None:
            from cr3_lab.tasks.motor_scene import DEFAULT_SCENE
            scene = DEFAULT_SCENE
        self.cameras = cameras
        self.center = np.asarray(scene.initial_xy_m)
        self.z = scene.rest_z_m
        angle = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        motor = scene.motor

        def ring(radius, y):
            return np.column_stack((radius * np.cos(angle), np.full_like(angle, y),
                                    self.z + radius * np.sin(angle)))

        half = motor.length_m / 2
        self.shape = np.vstack((ring(motor.radius_m, -half), ring(motor.radius_m, half),
                                ring(motor.boss_radius_m, half + motor.boss_length_m)))
        # Search the left workcell around the nominal spawn, including room for the silhouette.
        corners = np.array([[self.center[0] + x, self.center[1] + y, self.z + z]
                            for x in (-0.065, 0.065) for y in (-0.085, 0.085)
                            for z in (-0.028, 0.028)])
        self.rois = {}
        for name, camera in cameras.items():
            pixels, depth = project(corners, camera)
            if np.any(depth <= 0):
                raise ValueError(f"Workcell behind {name} camera")
            low = np.maximum(0, np.floor(pixels.min(axis=0))).astype(int)
            high = np.minimum([camera["width"], camera["height"]],
                              np.ceil(pixels.max(axis=0)) + 1).astype(int)
            self.rois[name] = (*low, *high)

    def _observe(self, image, name):
        camera = self.cameras[name]
        image = np.asarray(image)
        if image.shape != (camera["height"], camera["width"], 3) or image.dtype != np.uint8:
            return None
        x0, y0, x1, y1 = self.rois[name]
        red, green, blue = image[y0:y1, x0:x1].astype(np.int16).transpose(2, 0, 1)
        motor = (red > 45) & (red - green > 20) & (green - blue > 8)
        y, x = np.where(motor)
        if len(x) < 500:
            return None
        if min(x.min(), y.min(), x1 - x0 - 1 - x.max(), y1 - y0 - 1 - y.max()) == 0:
            return None
        box = np.array([x.min() + x0, y.min() + y0, x.max() + x0, y.max() + y0])
        if len(x) < 0.35 * np.prod(box[2:] - box[:2] + 1):
            return None
        return box, float(np.count_nonzero(motor & (green < 0.46 * red)) / len(x))

    def _fit(self, observed, direction):
        camera = self.cameras["localization"]
        shape = self.shape.copy()
        shape[:, 1] *= direction
        xy = unproject_to_z((observed[:2] + observed[2:]) / 2, self.z, camera)[:2]

        def bounds(position):
            pixels, _ = project(shape + [*position, 0], camera)
            return np.r_[pixels.min(axis=0), pixels.max(axis=0)]

        for _ in range(8):
            predicted = bounds(xy)
            jacobian = np.column_stack([(bounds(xy + step) - predicted) / 0.0001
                                        for step in np.eye(2) * 0.0001])
            delta = np.linalg.lstsq(jacobian, observed - predicted, rcond=None)[0]
            xy += delta
            if np.linalg.norm(delta) < 1e-6:
                break
        return xy, float(np.linalg.norm(bounds(xy) - observed) / 2)

    def predict(self, inspection_rgb, localization_rgb):
        inspection = self._observe(inspection_rgb, "inspection")
        if inspection is None:
            return VisualEstimate(None, None, 0.0, "inspection_missing_or_invalid")
        localization = self._observe(localization_rgb, "localization")
        if localization is None:
            return VisualEstimate(None, None, 0.0, "localization_missing_or_invalid")
        ratio = inspection[1]
        if 0.02 <= ratio <= 0.10:
            return VisualEstimate(None, None, 0.0, "ambiguous_end_face")
        protrusion = ratio > 0.10
        label = "protrusion_facing_camera" if protrusion else "flat_facing_camera"
        camera = self.cameras["inspection"]
        eye = -np.asarray(camera["R_world_to_optical"]).T @ camera["t_world_to_optical_m"]
        direction = (1 if protrusion else -1) * np.sign(eye[1] - self.center[1])
        try:
            xy, residual = self._fit(localization[0], direction)
        except (ValueError, np.linalg.LinAlgError):
            return VisualEstimate(None, label, 0.0, "localization_geometry_failed")
        if (not np.isfinite(xy).all() or not np.isfinite(residual)
                or residual > 6 or np.any(abs(xy - self.center) > 0.035)):
            return VisualEstimate(None, label, 0.0, "localization_geometry_failed")
        # Heuristic quality score based on distance from the rejection thresholds.
        classification_score = min(1.0, (ratio - 0.10) / 0.10 if protrusion else (0.02 - ratio) / 0.02)
        return VisualEstimate(tuple(map(float, xy)), label,
                              float(min(classification_score, 1 - residual / 6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration")
    parser.add_argument("inspection")
    parser.add_argument("localization")
    args = parser.parse_args()
    result = RgbMotorEstimator(load_calibration(args.calibration)).predict(
        read_rgb(args.inspection), read_rgb(args.localization))
    print(json.dumps({**asdict(result), "valid": result.valid}, indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
