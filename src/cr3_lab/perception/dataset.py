"""Capture paired RGB, offline Oracle labels, and deterministic coverage checks."""
import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cr3_lab.tasks.motor_expert import MotorExpert
from cr3_lab.tasks.motor_scene import (
    DEFAULT_SCENE, MOTOR_GEOMS, Orientation, SpawnRange, load_scene, oracle_pose, reset_scene,
)
from .camera import calibration, project

WIDTH, HEIGHT = 600, 400
CAMERAS = ("inspection", "localization")
SPLIT_SEEDS = {"train": range(2000, 2012), "validation": range(3000, 3004),
               "test": range(4000, 4006)}


def cases():
    yield "test", SpawnRange.FIXED, 4100, "center", None
    for split, seeds in SPLIT_SEEDS.items():
        for profile in (SpawnRange.SMALL, SpawnRange.LARGE):
            for seed in seeds:
                yield split, profile, seed, "random", None
    for profile in (SpawnRange.SMALL, SpawnRange.LARGE):
        for index, offset in enumerate(product((-1, 0, 1), repeat=2)):
            if offset == (0, 0):
                continue
            xy = np.asarray(DEFAULT_SCENE.initial_xy_m) + profile.half_width_m * np.array(offset)
            name = "corner" if all(offset) else "edge_midpoint"
            yield "test", profile, 4200 + index, f"{name}_{index}", tuple(xy)


def coverage(renderer, data, name, camera, model, motor_ids, cap_world):
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[5] = 1
    renderer.update_scene(data, name, scene_option=scene_option)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    try:
        seg = renderer.render()
        isolated = mujoco.MjvOption()
        isolated.geomgroup[:] = 0
        isolated.geomgroup[5] = 1
        isolated.sitegroup[:] = 0
        renderer.update_scene(data, name, scene_option=isolated)
        reference = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()

    def pixels(image, geom):
        return (image[..., 0] == geom) & (image[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)

    mask = np.logical_or.reduce([pixels(seg, geom) for geom in motor_ids])
    full = np.logical_or.reduce([pixels(reference, geom) for geom in motor_ids])
    y, x = np.where(full)
    margin = int(min(x.min(), y.min(), WIDTH - 1 - x.max(), HEIGHT - 1 - y.max())) if len(x) else -1
    cap_pixels, depth = project(cap_world, camera)
    # Sample the exposed near-end disk (an annulus when the boss is on this end).
    eye = data.cam(name).xpos
    geom_group = np.ones(6, dtype=np.uint8)
    geom_group[3] = 0  # hidden arm collision meshes
    geom_id = np.empty(1, dtype=np.int32)
    visible = 0
    for point, uv, z in zip(cap_world, cap_pixels, depth):
        if z > 0 and 0 <= uv[0] < WIDTH and 0 <= uv[1] < HEIGHT:
            ray = point - eye
            ray_length = np.linalg.norm(ray)
            hit = mujoco.mj_ray(model, data, eye, ray / ray_length, geom_group, True, -1, geom_id)
            visible += geom_id[0] == motor_ids[0] and abs(hit - ray_length) < 0.0002
    return rgb, {
        "bbox_xyxy_px": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None,
        "frame_margin_px": margin,
        "visible_silhouette_fraction": float((mask & full).sum() / max(1, full.sum())),
        "boss_visible_pixels": int(pixels(seg, motor_ids[1]).sum()),
        "near_face_visible_fraction": float(visible / len(cap_world)),
    }


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    model = load_scene()
    data = mujoco.MjData(model)
    expert = MotorExpert(model)
    motor_ids = [model.geom(name).id for name in MOTOR_GEOMS]
    # Render group used only for the offline isolated silhouette reference.
    model.geom_group[motor_ids] = 5
    records = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for split, profile, seed, case, xy in cases():
            scene = replace(DEFAULT_SCENE, initial_xy_m=xy) if xy is not None else DEFAULT_SCENE
            for orientation in Orientation:
                reset_scene(model, data, orientation, seed, scene,
                            spawn_range=SpawnRange.FIXED if xy is not None else profile)
                expert.grasp_ready(data)
                center, rotation = oracle_pose(data)
                normal = rotation[:, 2]
                front = bool(normal @ (data.cam("inspection").xpos - center) > 0)
                label = Orientation.PROTRUSION if front else Orientation.FLAT
                angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
                radii = [0.016, 0.021, 0.024] if front else [0.006, 0.015, 0.024]
                cap_local = np.array([(r * np.cos(a), r * np.sin(a),
                                       (1 if front else -1) * scene.motor.length_m / 2)
                                      for r in radii for a in angles])
                cap_world = center + cap_local @ rotation.T
                row = {
                    "split": split, "profile": profile, "seed": seed, "case": case,
                    "oracle": {"orientation": label, "position_xy_m": center[:2].tolist(),
                               "initial_xyz_m": center.tolist(),
                               "initial_quaternion_wxyz": data.joint("motor_free").qpos[3:].tolist()},
                    "rgb": {}, "coverage": {},
                }
                for name in CAMERAS:
                    camera = calibration(model, data, name, WIDTH, HEIGHT)
                    rgb, metrics = coverage(renderer, data, name, camera, model, motor_ids, cap_world)
                    path = Path(split) / f"{profile}-{seed}-{case}-{label}-{name}.png"
                    (output / path.parent).mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(output / path)
                    row["rgb"][name] = path.as_posix()
                    row["coverage"][name] = metrics
                records.append(row)
    (output / "cameras.json").write_text(json.dumps({
        name: calibration(model, data, name, WIDTH, HEIGHT) for name in CAMERAS}, indent=2) + "\n")
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return records


def inspect(output):
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("Dataset manifest is empty")
    groups = {}
    labels = {}
    cameras = json.loads((output / "cameras.json").read_text())
    inspection = cameras["inspection"]
    eye = -np.asarray(inspection["R_world_to_optical"]).T @ inspection["t_world_to_optical_m"]
    for row in rows:
        key = tuple(row["oracle"]["position_xy_m"])
        groups.setdefault(key, set()).add(row["split"])
        annotation = row["oracle"]
        quaternion = np.asarray(annotation["initial_quaternion_wxyz"])
        xyz = np.asarray(annotation["initial_xyz_m"])
        if not np.isclose(np.linalg.norm(quaternion), 1) or tuple(xyz[:2]) != key:
            raise ValueError("Invalid Oracle pose annotation")
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        normal = matrix.reshape(3, 3)[:, 2]
        expected = Orientation.PROTRUSION if normal @ (eye - xyz) > 0 else Orientation.FLAT
        if annotation["orientation"] != expected:
            raise ValueError("End-face label disagrees with the camera-relative pose")
        labels.setdefault(key, Counter())[expected] += 1
    if any(counts != Counter(Orientation) for counts in labels.values()):
        raise ValueError("Expected one RGB pair per end-face class at each XY")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("An XY group spans multiple splits")
    for row in rows:
        for name in CAMERAS:
            with Image.open(output / row["rgb"][name]) as image:
                if image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                    raise ValueError("Invalid RGB image")
                image.load()
    samples = [row for row in rows if row["profile"] == "fixed"
               or (row["profile"] == "large" and row["case"] in {"corner_0", "corner_8"})]
    sheet = Image.new("RGB", (720, 270 * len(samples)), "#222222")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(samples):
        for j, name in enumerate(CAMERAS):
            with Image.open(output / row["rgb"][name]) as image:
                sheet.paste(image.resize((360, 240)), (j * 360, i * 270))
            x, y = row["oracle"]["position_xy_m"]
            label = row["oracle"]["orientation"].split("_")[0]
            draw.text((j * 360 + 5, i * 270 + 245), f"{name} | {label} | ({x:.3f}, {y:.3f})",
                      fill="white")
    sheet.save(output / "samples.png")
    report = {"pairs_by_split": dict(Counter(row["split"] for row in rows)),
              "distinct_xy": len(groups),
              "pairs_by_profile": dict(Counter(row["profile"] for row in rows)), "coverage": {}}
    for name in CAMERAS:
        metrics = [row["coverage"][name] for row in rows]
        report["coverage"][name] = {
            "min_margin_px": min(item["frame_margin_px"] for item in metrics),
            "min_silhouette_visible": min(item["visible_silhouette_fraction"] for item in metrics),
            "min_near_face_visible": min(item["near_face_visible_fraction"] for item in metrics),
            "min_front_boss_pixels": min(row["coverage"][name]["boss_visible_pixels"] for row in rows
                                         if row["oracle"]["orientation"] == Orientation.PROTRUSION),
            "max_back_boss_pixels": max(row["coverage"][name]["boss_visible_pixels"] for row in rows
                                        if row["oracle"]["orientation"] == Orientation.FLAT),
        }
    for profile in SpawnRange:
        selected = [row for row in rows if row["profile"] == profile]
        report.setdefault("by_profile", {})[profile] = {
            name: {
                "min_margin_px": min(row["coverage"][name]["frame_margin_px"] for row in selected),
                "min_silhouette_visible": min(row["coverage"][name]["visible_silhouette_fraction"]
                                              for row in selected),
                "min_near_face_visible": min(row["coverage"][name]["near_face_visible_fraction"]
                                             for row in selected),
            } for name in CAMERAS
        }
    (output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "inspect"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        capture(args.output)
    else:
        inspect(args.output)


if __name__ == "__main__":
    main()
