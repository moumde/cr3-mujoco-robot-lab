"""Validate timing and replay the exact actuator sequence in MuJoCo."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from cr3_lab.core.control import STEPS_PER_COMMAND
from cr3_lab.core.model import CONTROL_HZ, TIMESTEP_S
from cr3_lab.tasks.motor_expert import MotorExpert
from .collect import FPS, INTEGRATION, scene


def check(root, episode):
    folder = root / episode["id"]
    control = [json.loads(line) for line in (folder / "control.jsonl").read_text().splitlines()]
    assert len(control) == episode["controls"]
    assert episode["terminal"] and (episode["success"] == (episode["failure_stage"] is None))
    model = scene(episode["dim"])
    robot = MotorExpert(model)
    data = mujoco.MjData(model)
    initial = np.load(folder / "initial.npy")
    mujoco.mj_setState(model, data, initial, INTEGRATION)
    state = np.empty_like(initial)
    joint_lower, joint_upper = model.jnt_range[[model.joint(f"joint{i}").id for i in range(1, 7)]].T
    with np.load(folder / "frames.npz") as frames:
        np.testing.assert_array_equal(frames["control_index"], np.arange(0, len(control), CONTROL_HZ // FPS))
        rgb = frames["rgb"]
        assert rgb.shape == (episode["frames"], 2, 400, 600, 3) and rgb.dtype == np.uint8
    for i, row in enumerate(control):
        assert row["tick"] == i * STEPS_PER_COMMAND
        assert row["time_s"] == row["observed_time_s"]
        assert abs(row["time_s"] - i / CONTROL_HZ) < 1e-12
        assert abs(row["action_end_s"] - (i + 1) / CONTROL_HZ) < 1e-12
        action, local = np.asarray(row["action"]), np.asarray(row["state"])
        assert action.shape == (7,) and local.shape == (20,)
        assert np.isfinite(action).all() and np.isfinite(local).all()
        assert np.all(local[:6] >= joint_lower - 1e-4) and np.all(local[:6] <= joint_upper + 1e-4)
        assert -.001 <= local[12] <= .075 and abs(np.linalg.norm(local[-4:]) - 1) < 1e-6
        assert 0 <= action[6] <= 1
        np.testing.assert_array_equal(action[:6], np.clip(row["arm_request_with_bias_rad"],
                                                           *robot.arm.limits_rad.T))
        np.testing.assert_array_equal(data.qpos[robot.arm.qpos], local[:6])
        np.testing.assert_array_equal(data.qvel[robot.arm.dofs], local[6:12])
        robot.gripper.command(data, action[6])
        robot.arm.step(data, action[:6])
        mujoco.mj_getState(model, data, state, INTEGRATION)
        np.testing.assert_allclose(state, row["integration_after"], atol=1e-12, rtol=0)
    assert abs(data.time - episode["duration_s"]) < TIMESTEP_S / 100
    return dict(id=episode["id"], controls=len(control), frames=episode["frames"], replay="verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--episode")
    args = parser.parse_args()
    episodes = json.loads((args.raw / "manifest.json").read_text())
    selected = [r for r in episodes if args.episode is None or r["id"] == args.episode]
    if not selected:
        parser.error("unknown episode")
    result = [check(args.raw, row) for row in selected]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
