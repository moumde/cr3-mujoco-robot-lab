import json

import numpy as np
import pytest

from cr3_lab.perception.camera import load_calibration, project, unproject_to_z
from cr3_lab.perception.evaluate_rgb import LABELS, metrics
from cr3_lab.perception.rgb_estimator import RgbMotorEstimator


@pytest.fixture
def cameras(tmp_path):
    camera = {"width": 600, "height": 400, "distortion": "none",
              "K": [[700, 0, 299.5], [0, 700, 199.5], [0, 0, 1]],
              "R_world_to_optical": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
              "t_world_to_optical_m": [-0.45, 0.10, 0.65]}
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps({"inspection": camera, "localization": camera}))
    return load_calibration(path)


def test_unproject_roundtrip(cameras):
    camera = cameras["localization"]
    point = [0.46, 0.076, 0.146]
    pixels, _ = project([point], camera)
    np.testing.assert_allclose(unproject_to_z(pixels[0], point[2], camera), point)


@pytest.mark.parametrize("protrusion", [True, False])
def test_rgb_detection_and_geometry(cameras, protrusion):
    estimator = RgbMotorEstimator(cameras)
    point = np.array([0.464, 0.081])
    direction = 1 if protrusion else -1
    shape = estimator.shape.copy()
    shape[:, 1] *= direction
    pixels, _ = project(shape + [*point, 0], cameras["localization"])
    (x0, y0), (x1, y1) = np.rint([pixels.min(axis=0), pixels.max(axis=0)]).astype(int)
    localization = np.zeros((400, 600, 3), dtype=np.uint8)
    localization[y0:y1 + 1, x0:x1 + 1] = [150, 90, 25]
    inspection = localization.copy()
    if protrusion:
        inspection[y0:(y0 + y1) // 2, x0:(x0 + x1) // 2] = [150, 53, 20]
    result = estimator.predict(inspection, localization)
    assert result.valid and result.orientation == LABELS[not protrusion]
    np.testing.assert_allclose(result.position_xy, point, atol=0.001)
    assert 0 <= result.confidence <= 1


def test_invalid_prediction_is_counted(cameras):
    blank = np.zeros((400, 600, 3), dtype=np.uint8)
    result = RgbMotorEstimator(cameras).predict(blank, blank)
    assert not result.valid and result.position_xy is None and result.confidence == 0
    summary = metrics([{"truth": LABELS[0], "predicted_class": "invalid", "xy_error_mm": None}])
    assert summary["accuracy"] == 0 and summary["invalid_count"] == 1
    assert summary["confusion_matrix"] == [[0, 0, 1], [0, 0, 0]]
