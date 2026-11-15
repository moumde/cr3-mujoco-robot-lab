"""Evaluate RGB predictions against a selected, unchanged P2a split."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .camera import load_calibration
from .rgb_estimator import RgbMotorEstimator, read_rgb

LABELS = ("protrusion_facing_camera", "flat_facing_camera")
PREDICTIONS = (*LABELS, "invalid")


def metrics(rows):
    matrix = [[sum(row["truth"] == truth and row["predicted_class"] == prediction
                   for row in rows) for prediction in PREDICTIONS] for truth in LABELS]
    errors = [row["xy_error_mm"] for row in rows if row["xy_error_mm"] is not None]
    return {
        "count": len(rows),
        "accuracy": sum(matrix[i][i] for i in range(2)) / len(rows),
        "confusion_matrix": matrix,
        "invalid_count": len(rows) - len(errors),
        "xy_mm": {"mean": float(np.mean(errors)), "median": float(np.median(errors)),
                  "p95": float(np.percentile(errors, 95)), "max": max(errors)} if errors else None,
    }


def plot(output, summary, rows):
    canvas = Image.new("RGB", (840, 380), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), "End-face confusion | rows: truth", fill="#152433")
    draw.text((20, 49), "Predicted: protrusion / flat / invalid", fill="#152433")
    for i, values in enumerate(summary["confusion_matrix"]):
        draw.text((20, 91 + i * 70), ("Protrusion", "Flat")[i], fill="#152433")
        for j, value in enumerate(values):
            x, y = 100 + j * 69, 75 + i * 70
            draw.rectangle((x, y, x + 59, y + 59), fill="#c9e4f5" if i == j else "#edf0f3")
            draw.text((x + 20, y + 20), str(value), fill="#152433")
    draw.text((350, 18), "XY error distribution (mm) | valid estimates", fill="#152433")
    draw.text((350, 42), "Blue: protrusion front   Orange: flat front", fill="#152433")
    errors = [row["xy_error_mm"] for row in rows if row["xy_error_mm"] is not None]
    if errors:
        xmax = max(1, max(errors))
        ymax = max(1, len(errors))
        draw.line([(380, 310), (800, 310)], fill="#152433")
        for i in range(6):
            x = 380 + i * 84
            draw.text((x - 8, 321), f"{i * xmax / 5:.2f}", fill="#152433")
        for label, color in zip(LABELS, ("#156bab", "#d16829")):
            values = sorted(row["xy_error_mm"] for row in rows
                            if row["truth"] == label and row["xy_error_mm"] is not None)
            for i, value in enumerate(values):
                x, y = 380 + 420 * value / xmax, 303 - 225 * i / ymax
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((380, 348), "Each point is one scene, stacked by rank within its class.",
                  fill="#152433")
    canvas.save(output / "metrics.png")


def examples(dataset, output, rows):
    failures = [row for row in rows if not row["prediction"]["valid"]
                or row["predicted_class"] != row["truth"] or row["xy_error_mm"] > 5]
    worst = sorted((row for row in rows if row["xy_error_mm"] is not None),
                   key=lambda row: row["xy_error_mm"], reverse=True)
    selected = (failures or worst)[:6]
    if selected:
        sheet = Image.new("RGB", (720, 265 * len(selected)), "#22303e")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            for j, name in enumerate(("inspection", "localization")):
                with Image.open(dataset / row["rgb"][name]) as image:
                    sheet.paste(image.resize((360, 240)), (j * 360, i * 265))
            error = row["xy_error_mm"]
            status = row["prediction"]["failure"] or f"{error:.3f} mm"
            draw.text((6, i * 265 + 244),
                      f"{row['profile']} {row['seed']} | {row['truth'].split('_')[0]} -> "
                      f"{row['predicted_class'].split('_')[0]} | {status}", fill="white")
        sheet.save(output / "examples.png")
    return len(failures)


def evaluate(dataset, output, split):
    output.mkdir(parents=True, exist_ok=True)
    estimator = RgbMotorEstimator(load_calibration(dataset / "cameras.json"))
    rows = []
    for line in (dataset / "manifest.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if sample["split"] != split:
            continue
        images = sample["rgb"]
        result = estimator.predict(read_rgb(dataset / images["inspection"]),
                                   read_rgb(dataset / images["localization"]))
        oracle = sample["oracle"]
        xy = oracle["position_xy_m"]
        error = float(1000 * np.linalg.norm(np.asarray(result.position_xy) - xy)) if result.valid else None
        rows.append({"seed": sample["seed"], "profile": sample["profile"], "case": sample["case"],
                     "rgb": images, "truth": oracle["orientation"], "truth_xy_m": xy,
                     "predicted_class": result.orientation if result.valid else "invalid",
                     "prediction": {**asdict(result), "valid": result.valid}, "xy_error_mm": error})
    if not rows:
        raise ValueError(f"No samples in split {split}")
    groups = {"all": rows}
    for name, profiles in {"fixed": {"fixed"}, "random": {"small", "large"},
                           "small": {"small"}, "large": {"large"}}.items():
        groups[name] = [row for row in rows if row["profile"] in profiles]
    for label in LABELS:
        groups[label] = [row for row in rows if row["truth"] == label]
    stats = {name: metrics(group) for name, group in groups.items() if group}
    plot(output, stats["all"], rows)
    failure_count = examples(dataset, output, rows)
    report = {"split": split, "true_classes": LABELS, "predicted_classes": PREDICTIONS,
              "groups": stats, "example_failure_count": failure_count, "samples": rows}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()
    evaluate(args.dataset, args.output, args.split)


if __name__ == "__main__":
    main()
