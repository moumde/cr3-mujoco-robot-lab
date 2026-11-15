"""Collect and verify fixed-workcell expert transitions, then seed SAC replay."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .check_grasp_lift import rollout
from .grasp_lift import MotorGraspLiftEnv

DEMO_SEEDS = range(11000, 11008)
FIELDS = ("observation", "action", "reward", "next_observation", "terminated", "truncated")


def collect():
    env = MotorGraspLiftEnv(start="pregrasp")
    rows, offsets = [], [0]
    started = perf_counter()
    for seed in DEMO_SEEDS:
        info = rollout(env, seed, True, on_transition=lambda *row: rows.append(row))
        if not info["is_success"]:
            raise RuntimeError(f"Expert failed on seed {seed}: {info}")
        offsets.append(len(rows))
    env.close()
    columns = {key: np.asarray(values) for key, values in zip(FIELDS, zip(*rows))}
    return {**columns, "offsets": np.asarray(offsets), "seeds": np.asarray(DEMO_SEEDS),
            "collection_wall_s": np.asarray(perf_counter() - started)}


def verify(data, scale):
    """Replay recorded actions from their seeds through the training observation wrapper."""
    from .train_grasp_lift import ScaledGrasp

    env = ScaledGrasp(MotorGraspLiftEnv(start="pregrasp"))
    actions, offsets, seeds = data["action"], data["offsets"], data["seeds"]
    if (not np.array_equal(seeds, np.asarray(DEMO_SEEDS))
            or len(offsets) != len(seeds) + 1 or offsets[0] != 0
            or offsets[-1] != len(actions) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Unexpected demonstration episode layout")
    for episode, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed), options={"scene_seed": int(seed)})
        for i in range(offsets[episode], offsets[episode + 1]):
            np.testing.assert_array_equal(obs, data["observation"][i] * scale)
            if not env.action_space.contains(actions[i]):
                raise ValueError("Demonstration action is outside the normalized Box")
            obs, reward, terminated, truncated, info = env.step(actions[i])
            np.testing.assert_array_equal(obs, data["next_observation"][i] * scale)
            np.testing.assert_equal((reward, terminated, truncated),
                                    tuple(data[key][i] for key in ("reward", "terminated", "truncated")))
            if (terminated or truncated) != (i + 1 == offsets[episode + 1]):
                raise ValueError("Demonstration boundary disagrees with termination")
        if not info["is_success"]:
            raise ValueError("Demonstration replay did not finish successfully")
    env.close()


def prefill(model, data, scale):
    buffer = model.replay_buffer
    count = len(data["action"])
    if buffer.size() or count > buffer.buffer_size:
        raise ValueError("Demonstrations require an empty buffer with adequate capacity")
    for obs, action, reward, next_obs, terminated, truncated in zip(*(data[key] for key in FIELDS)):
        # Both policy and environment use [-1,1], so the recorded action is already normalized.
        buffer.add((obs * scale)[None], (next_obs * scale)[None], action[None],
                   np.asarray([reward]), np.asarray([terminated or truncated]),
                   [{"TimeLimit.truncated": bool(truncated and not terminated)}])
    return count


def main():
    from .train_grasp_lift import SCALE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = collect()
    verify(data, SCALE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **data)
    print(f"Verified {len(data['seeds'])} episodes and {len(data['action'])} transitions")


if __name__ == "__main__":
    main()
