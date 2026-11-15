"""Standalone paired evaluation. Only the reset sampler is replaced."""
import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from cr3_lab.rl.behavior_clone import model_for
from cr3_lab.rl.check_grasp_lift import expert_actions
from cr3_lab.rl.grasp_benchmark import execute
from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.train_grasp_lift import SCALE, summarize
from cr3_lab.tasks.motor_scene import DEFAULT_SCENE, Orientation, reset_scene

DEV = {25: range(17000, 17020), 30: range(17100, 17120), 35: range(17200, 17220)}
TEST = {25: range(17500, 17540), 30: range(17600, 17640), 35: range(17700, 17740)}
CHECKPOINT = Path("outputs/p3c-6/run/model.pt")
ROOT = Path(__file__).resolve().parent


def sample_xy(width, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    return np.asarray(DEFAULT_SCENE.initial_xy_m) + rng.uniform(-width, width, 2)


def clearance(width):
    scene = DEFAULT_SCENE
    extent = np.array([scene.motor.radius_m, scene.motor.length_m / 2 + scene.motor.boss_length_m])
    return np.asarray(scene.table_half_size_m[:2]) - abs(
        np.asarray(scene.initial_xy_m) - scene.table_xy_m) - width - extent


class ChallengeEnv(MotorGraspLiftEnv):
    def __init__(self):
        super().__init__()
        self.xy = None

    def reset(self, *, seed=None, options=None):
        scene = replace(DEFAULT_SCENE, initial_xy_m=tuple(self.xy))

        def spawn(model, data, orientation, seed, *_args, **_kwargs):
            return reset_scene(model, data, orientation, seed, scene)

        # Keep the accepted full reset and nominal grasp_ready IK; only choose object XY.
        with patch("cr3_lab.rl.grasp_lift.reset_scene", spawn):
            return super().reset(seed=seed, options=options)


def load_policy():
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = model_for(True)
    model.load_state_dict(state["weights"])
    model.eval()
    center, spread = (state[key].numpy() for key in ("center", "spread"))

    @torch.no_grad()
    def predict(obs):
        return model(torch.from_numpy(np.clip((obs * SCALE - center) / spread, -8, 8))).numpy()

    return predict


def paired(env, seed, orientation, policy):
    before, oracle = execute(env, seed, orientation, lambda: expert_actions(env))
    after, bc = execute(env, seed, orientation, policy=policy)
    np.testing.assert_array_equal(before, after)
    return {"seed": seed, "orientation": orientation, "spawn_xy_m": env.xy.tolist(),
            "oracle": oracle, "bc": bc}


def run():
    torch.set_num_threads(1)
    policy = load_policy()
    env = ChallengeEnv()
    results = {"ranges": {}, "reserved_test_seeds": {k: list(v) for k, v in TEST.items()}}
    for mm, seeds in DEV.items():
        width = mm / 1000
        margins = clearance(width)
        if np.any(margins <= 0) or DEFAULT_SCENE.initial_xy_m[1] - width <= 0.053:
            raise ValueError("Object footprint crosses table edge or workcell centerline")
        # Run unchanged physics at all four corners first to check the full expert path.
        corners = []
        for signs, orientation in product(product((-1, 1), repeat=2), Orientation):
            env.xy = np.asarray(DEFAULT_SCENE.initial_xy_m) + width * np.asarray(signs)
            _, row = execute(env, 16999, orientation, lambda: expert_actions(env))
            corners.append({**row, "spawn_xy_m": env.xy.tolist()})
        pairs = []
        for seed, orientation in product(seeds, Orientation):
            env.xy = sample_xy(width, seed)
            pairs.append(paired(env, seed, orientation, policy))
        stats = {method: summarize([row[method] for row in pairs]) for method in ("oracle", "bc")}
        results["ranges"][mm] = {"clearance_m": margins.tolist(), "corners": corners,
                                 "summary": stats, "pairs": pairs}
        print(mm, "corner success", sum(r["is_success"] for r in corners), stats, flush=True)
    env.close()
    with (ROOT / "results.json").open("x") as stream:
        json.dump(results, stream, indent=2)


def media(mm, seed, orientation):
    import mujoco
    from PIL import Image, ImageDraw

    torch.set_num_threads(1)
    env = ChallengeEnv()
    env.xy = sample_xy(mm / 1000, seed)
    policy = load_policy()
    with mujoco.Renderer(env.model, height=320, width=480) as renderer:
        for name, predictor in (("oracle", None), ("bc", policy)):
            frames = []

            def frame(_obs, _action):
                if env.steps % 4:
                    return
                renderer.update_scene(env.data, "overview")
                img = Image.fromarray(renderer.render())
                ImageDraw.Draw(img).text((8, 8), f"{name.upper()} | ±{mm} mm | step {env.steps}",
                                         fill="white", stroke_width=2, stroke_fill="black")
                frames.append(img)

            source = (lambda: expert_actions(env)) if predictor is None else None
            _, result = execute(env, seed, orientation, source, policy=predictor, on_action=frame)
            renderer.update_scene(env.data, "overview")
            final = Image.fromarray(renderer.render())
            ImageDraw.Draw(final).text((8, 8), f"{name.upper()} | {result['reason']}",
                                       fill="white", stroke_width=2, stroke_fill="black")
            frames.append(final)
            frames[0].save(ROOT / f"{mm}-{seed}-{name}.gif", save_all=True,
                           append_images=frames[1:], duration=[80] * (len(frames) - 1) + [1000], loop=0)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", nargs=3, metavar=("MM", "SEED", "ORIENTATION"))
    args = parser.parse_args()
    if args.gif:
        mm, seed, orientation = args.gif
        media(int(mm), int(seed), Orientation(orientation))
    else:
        run()
