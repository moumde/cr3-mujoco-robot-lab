import numpy as np
import torch

from research import Episodes, anchor, schedule
from cr3_lab.tasks.motor_scene import Orientation


def test_schedule_and_anchor():
    assert schedule(11) == schedule(11) and schedule(11) != schedule(22)
    assert all(18000 <= seed <= 18079 and side in (0, 1) for seed, side in schedule(33))
    assert anchor(torch.zeros(1,3), torch.ones(1,3)).item() == 1.
    env = Episodes([(18000, 0), (18079, 1)])
    try:
        for seed, side in [(18000,0),(18079,1)]:
            obs, info = env.reset()
            assert obs.shape == (40,)
            assert info["scene_seed"] == seed
            assert info["orientation"] == list(Orientation)[side]
    finally:
        env.close()
