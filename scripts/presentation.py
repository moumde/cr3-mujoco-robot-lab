"""Build local review media and publishable numeric charts from frozen artifacts."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIENTATIONS = ("protrusion_facing_camera", "flat_facing_camera")


def read(path):
    return json.loads(path.read_text())


def charts(out, archive):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r2 = read(ROOT / "outputs/p5a-r2/summary.json")
    r3 = read(ROOT / "outputs/p5a-r3/summary.json")["strategies"]
    out.mkdir(parents=True, exist_ok=True)
    colors = {"ppo": "#1864ab", "sac": "#c05621"}
    labels = list(r3)
    positions = np.arange(len(labels))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, run in r2.items():
        for ax, panel in zip(axes, ("history", "validation"), strict=True):
            records = run["records"]
            ax.plot([r["step"] for r in records],
                    [r[panel]["success"] / r[panel]["count"] for r in records],
                    label=name, color=colors[name.split("-")[0]],
                    linestyle={"11": "-", "22": "--", "33": ":"}[name.split("-")[1]],
                    marker=".", alpha=.85)
            ax.set(title=f"R2 {panel} panel", xlabel="Online environment steps",
                   ylabel="Stable grasp success rate", ylim=(0, 1.05))
    axes[0].legend(ncol=2, fontsize=8)
    save(fig, "learning")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (title, key) in zip(axes.flat, (
        ("Stable successes / 80", "success"), ("Successful mean steps", "successful_mean"),
        ("Paired step change vs BC", "paired_mean"), ("IK rejects", "ik"),
        ("Peak absolute cumulative correction (mm)", "absolute_max_mm"),
        ("Successful P95 steps", "p95"),
    ), strict=True):
        if key == "absolute_max_mm":
            for i, axis in enumerate("XYZ"):
                ax.bar(positions + (i - 1) * .24,
                       [row[key][i] for row in r3.values()], .24, label=axis)
            ax.legend()
        elif key == "ik":
            bottom = np.zeros(len(labels))
            for metric in ("solver_nonconverged", "joint_step_limit"):
                count = [row[key][metric] for row in r3.values()]
                ax.bar(positions, count, bottom=bottom, label=metric)
                bottom += count
            ax.legend(fontsize=7)
        else:
            ax.bar(positions, [row["successful_p50_p95"][1] if key == "p95"
                               else row[key] for row in r3.values()])
        ax.set(title=title)
        ax.set_xticks(positions, labels, rotation=30)
    fig.suptitle("R3 frozen independent panel")
    save(fig, "independent")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, run in r2.items():
        if name.startswith("ppo"):
            updates = [json.loads(line) for line in
                       (archive / "p5a-r2" / name / "updates.jsonl").read_text().splitlines()]
            for ax, key in zip(axes[:2], ("approx_kl", "clip_fraction"), strict=True):
                points = [(r["step"], r[key]) for r in updates if key in r]
                ax.plot(*zip(*points), label=name)
                ax.set(title=key)
        else:
            data = sorted((int(step), max(row["actor_drift_p95"]))
                          for step, row in run["online"].items())
            axes[2].plot(*zip(*data), label=name)
            axes[2].set(title="SAC max axis actor drift P95")
    for ax in axes:
        ax.set(xlabel="Online environment steps")
        ax.legend(fontsize=8)
    save(fig, "optimization")

    training = read(archive / "p4c/train/training.json")
    vision = read(ROOT / "outputs/p4c/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].semilogy([r["epoch"] for r in training["history"]],
                        [r[split]["mse"] for r in training["history"]], label=split)
    axes[0].set(xlabel="Epoch", ylabel="Normalized action MSE", title="P4c offline")
    axes[0].legend()
    stages = vision["stage_metrics"]
    axes[1].bar(list(stages), list(stages.values()))
    axes[1].set(title=f"P4c closed loop: {vision['milestones']['success']}/{vision['total']}",
                ylabel="Stage successes", ylim=(0, vision["total"]))
    axes[1].tick_params(axis="x", labelrotation=55)
    save(fig, "vision-bc")


def write_video(path, frames, fps):
    import av
    from itertools import chain

    frames = iter(frames)
    first = next(frames)
    if path.exists():
        raise FileExistsError(path)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=Fraction(fps))
        stream.width, stream.height = first.size
        stream.pix_fmt = "yuv420p"
        for image in chain((first,), frames):
            frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def visual_frames(base, short):
    from PIL import Image, ImageDraw, ImageSequence

    for mode in ("expert",) if short else ("expert", "learned"):
        results = {r["orientation"]: r for r in read(base / mode / "results.json")}
        for orientation in ORIENTATIONS:
            folder = base / mode / orientation
            rgb = Image.open(folder / "rgb.png").convert("RGB")
            with Image.open(folder / "motion.gif") as motion:
                for i, image in enumerate(ImageSequence.Iterator(motion)):
                    if short and i % 3:
                        continue
                    canvas = Image.new("RGB", (1000, 500), "#111827")
                    canvas.paste(image.convert("RGB"), (0, 52))
                    for y, label in ((0, "Inspection"), (400, "Localization")):
                        panel = rgb.crop((0, y, 600, y + 400)).resize((240, 160))
                        canvas.paste(panel, (616, 68 + y * 9 // 20))
                        ImageDraw.Draw(canvas).text((864, 73 + y * 9 // 20), label,
                                                    fill="white", font_size=16)
                    text = ImageDraw.Draw(canvas)
                    text.text((14, 13), f"Complete visual manipulation | {mode.upper()} | {orientation}",
                              fill="white", font_size=20)
                    predicted = results[orientation]["estimate"]["orientation"]
                    verdict = "SUCCESS" if results[orientation]["success"] else "FAILURE"
                    text.text((616, 419), f"RGB: {predicted}\nArchived outcome: {verdict}",
                              fill="white", font_size=16)
                    text.text((14, 466), "Approach > grasp > lift > reorient > transfer > place > retreat",
                              fill="white", font_size=16)
                    if short:
                        text.text((760, 466), "3x playback", fill="white", font_size=16)
                    yield canvas


def comparison(base, name):
    from PIL import Image, ImageDraw, ImageSequence
    selection = read(ROOT / "outputs/p5a-r3/media-selection.json")[name]
    with (ROOT / "outputs/p5a-r3/episodes.csv").open() as stream:
        episodes = {r["candidate"]: r for r in csv.DictReader(stream)
                    if int(r["seed"]) == selection["seed"] and r["orientation"] == selection["orientation"]}
    steps = {r["name"]: r["step"] for r in
             read(ROOT / "outputs/p5a-r3/frozen_candidates.json")["candidates"]}
    folder = base / "archive"
    plot = Image.open(folder / f"{name}-tcp.png").convert("RGB").resize((544, 354))
    with Image.open(folder / f"{name}.gif") as gif:
        for image in ImageSequence.Iterator(gif):
            frame = Image.new("RGB", (1080, 696), "#111827")
            frame.paste(image.convert("RGB"), (0, 42))
            frame.paste(plot, (0, 324))
            draw = ImageDraw.Draw(frame)
            draw.text((12, 10), f"Scene {selection['seed']} | {selection['orientation']} | frozen R3 playback",
                      fill="white", font_size=19)
            for i, policy in enumerate(selection["strategies"]):
                row = episodes[policy]
                status = "SUCCESS" if row["success"] == "True" else "FAILURE / timeout"
                draw.text((360 * i + 8, 294), f"{policy} @{steps[policy]} | {status} | {row['steps']} steps",
                          fill="white", font_size=16)
            draw.text((565, 395), "TCP X/Z path and IK rejections\n\nDots mark rejected IK steps.\n"
                      "Last frames hold after an episode ends.", fill="white", font_size=19)
            yield frame


def dataset_frames(base):
    from itertools import islice
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageSequence

    episode = base / "raw/episode-000"
    rows = [json.loads(line) for line in (episode / "control.jsonl").read_text().splitlines()]
    result = next(row for row in read(base / "raw/manifest.json") if row["id"] == episode.name)
    curves = [[r["state"][0] for r in rows], [r["action"][0] for r in rows],
              [r["action"][6] for r in rows]]
    videos = base / "lerobot/train/videos"
    file = next((videos / "observation.images.inspection_rgb").rglob("*.mp4"))
    verified = read(base / "lerobot/loader-check.json")["train"]["frames"]
    with np.load(episode / "frames.npz") as data, av.open(str(file)) as decoded, \
            Image.open(base / "collection-replay.gif") as replay:
        indices = data["control_index"]
        frames = data["rgb"]
        images = ImageSequence.Iterator(replay)
        stream = islice(decoded.decode(video=0), len(indices))
        for idx, pair, frame, compressed in zip(indices, frames, images, stream, strict=True):
            t = rows[idx]["time_s"]
            assert abs(compressed.time - t) < 1e-6
            canvas = Image.new("RGB", (1200, 650), "#111827")
            canvas.paste(frame.convert("RGB").resize((390, 260)), (0, 56))
            canvas.paste(Image.fromarray(pair[0]).resize((390, 260)), (402, 56))
            canvas.paste(Image.fromarray(pair[1]).resize((390, 260)), (804, 56))
            canvas.paste(compressed.to_image().resize((270, 180)), (900, 406))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 14), f"{episode.name} | {t:05.2f}s | {rows[idx]['phase']} | success={result['success']}",
                      fill="white", font_size=21)
            for x, label in zip((10, 412, 814), ("50 Hz physical replay", "Inspection RGB", "Localization RGB")):
                draw.text((x, 325), label, fill="white", font_size=17)
            for i, (curve, color, label) in enumerate(zip(curves, ("#56c8ff", "#ffd166", "#70e090"),
                    ("J1 state (rad)", "J1 executed target (rad)", "Gripper command [0,1]"))):
                y = 406 + 70 * i
                low, high = min(curve), max(curve)
                draw.rectangle((15, y, 840, y + 49), outline="#495773")
                points = [(15 + j * 825 / (len(curve) - 1), y + 46 - 42 * (v - low) / max(high-low, 1e-9))
                          for j, v in enumerate(curve)]
                draw.line(points, fill=color, width=2)
                cursor = points[idx][0]
                draw.line((cursor, y, cursor, y + 49), fill="white", width=2)
                draw.text((16, y - 17), f"{label} · {curve[idx]:.3f}", fill=color, font_size=14)
            draw.text((881, 369), f"Decoded LeRobot v3 frame\n{verified} pairs verified at 10 FPS",
                      fill="white", font_size=16)
            yield canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("charts", "videos"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, required=True,
                        help="P5a/P4c outputs root for charts or P5b-1 validation root for video")
    args = parser.parse_args()
    if args.mode == "charts":
        charts(args.output, args.archive)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    write_video(args.output / "a-complete.mp4", visual_frames(args.archive, False), Fraction(50, 3))
    write_video(args.output / "a-quick.mp4", visual_frames(args.archive, True), Fraction(50, 3))
    write_video(args.output / "b-learned-comparison.mp4", comparison(args.archive, "shared-success"), 10)
    write_video(args.output / "c-dataset.mp4", dataset_frames(args.archive), 10)
    write_video(args.output / "d-sac-failure-17721.mp4", comparison(args.archive, "regression"), 10)


if __name__ == "__main__":
    main()
