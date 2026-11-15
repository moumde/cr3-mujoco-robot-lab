from unittest.mock import patch
from gymnasium.error import ResetNeeded

import numpy as np
import pytest

from cr3_lab.rl.grasp_lift import MotorGraspLiftEnv
from cr3_lab.rl.residual_budget import BudgetedResidualEnv
from cr3_lab.tasks.ik import Unreachable

ZERO = np.zeros(3, dtype=np.float32)


def constant_bc(obs):
    assert obs.shape == (40,)
    return np.array([0, 0, 0, -1], dtype=np.float32)


@pytest.mark.parametrize("direction", (1, -1))
def test_signed_limit_and_reversal(direction):
    plant = MotorGraspLiftEnv(max_steps=3)
    env = BudgetedResidualEnv(plant, constant_bc, budget_m=(.0002, .003, .012))
    try:
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs[40:], ZERO)
        command = np.array([direction, 0, 0])
        with patch.object(plant.robot.ik, "solve", side_effect=lambda p, r, seed: seed.copy()):
            obs, _, term, trunc, info = env.step(command)
            assert not term and not trunc
            assert obs[40] == direction
            first = info["residual_budget"]
            assert first["accepted_residual_target_delta_m"][0] == pytest.approx(direction * .0002)
            obs, _, _, _, info = env.step(command)
            assert obs[40] == direction
            np.testing.assert_array_equal(info["residual_budget"]["accepted_residual_target_delta_m"], ZERO)
            obs, _, term, trunc, info = env.step(-command)
        assert trunc and not term
        assert obs[40] == -direction
        assert info["residual_budget"]["signed_budget_m"][0] == -direction * .0002
        assert info["residual_budget"]["absolute_budget_m"][0] == pytest.approx(.0006)
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs[40:], ZERO)
    finally:
        env.close()


@pytest.mark.parametrize("status", ("solver_nonconverged", "joint_step_limit"))
def test_rejected_action_spends_no_budget(status):
    plant = MotorGraspLiftEnv(max_steps=1)
    env = BudgetedResidualEnv(plant, constant_bc)
    try:
        env.reset(seed=1)
        def reject(p, r, seed):
            if status == "solver_nonconverged":
                raise Unreachable("test")
            return seed + .04
        with patch.object(plant.robot.ik, "solve", side_effect=reject):
            obs, _, _, trunc, info = env.step([1, 1, 1])
        trace = info["residual_budget"]
        assert trunc and trace["ik_status"] == status
        for key in ("accepted_residual_target_delta_m", "signed_budget_m", "absolute_budget_m"):
            np.testing.assert_array_equal(trace[key], ZERO)
        np.testing.assert_array_equal(obs[40:], ZERO)
    finally:
        env.close()


def test_workspace_projection_precedes_budget():
    plant = MotorGraspLiftEnv(max_steps=1)
    env = BudgetedResidualEnv(plant, constant_bc)
    try:
        env.reset(seed=1)
        plant.tcp_target[0] = plant.robot.scene.initial_xy_m[0] + .06
        with patch.object(plant.robot.ik, "solve", side_effect=lambda p, r, seed: seed.copy()):
            obs, _, _, _, info = env.step([1, 0, 0])
        trace = info["residual_budget"]
        assert trace["mixed_requested_target_m"][0] > trace["mixed_workspace_target_m"][0]
        assert trace["budget_target_m"] == trace["bc_workspace_target_m"]
        np.testing.assert_array_equal(obs[40:], ZERO)
    finally:
        env.close()


@pytest.mark.parametrize("budget,command", (((0, 0, 0), [1, -1, 1]),
                                            ((.006, .003, .012), ZERO)))
def test_zero_residual_or_zero_budget_is_exact(budget, command):
    raw, plant = MotorGraspLiftEnv(max_steps=2), MotorGraspLiftEnv(max_steps=2)
    env = BudgetedResidualEnv(plant, constant_bc, budget_m=budget)
    try:
        obs, _ = raw.reset(seed=2)
        env.reset(seed=2)
        for _ in range(2):
            action = constant_bc(obs)
            obs, reward, term, trunc, expected_info = raw.step(action)
            actual, other_reward, other_term, other_trunc, info = env.step(command)
            trace = info.pop("residual_budget")
            np.testing.assert_array_equal(actual, np.r_[obs, ZERO])
            np.testing.assert_array_equal(trace["final_action"], action)
            np.testing.assert_array_equal(trace["tcp_target_m"], raw.tcp_target)
            assert (other_reward, other_term, other_trunc, info) == (reward, term, trunc, expected_info)
    finally:
        raw.close()
        env.close()


def test_terminal_observation_keeps_accepted_budget():
    plant = MotorGraspLiftEnv(max_steps=4)
    env = BudgetedResidualEnv(plant, constant_bc)
    try:
        env.reset(seed=1)
        observed, _, _ = plant._observe()
        with (patch.object(plant.robot.ik, "solve", side_effect=lambda p, r, seed: seed.copy()),
              patch.object(plant, "_observe", return_value=(observed, "test contact", 0.0))):
            terminal, _, term, trunc, info = env.step([1, 0, 0])
        assert term and not trunc
        assert terminal[40] == pytest.approx(.125)
        assert info["residual_budget"]["absolute_budget_m"][0] == pytest.approx(.00075)
        with pytest.raises(ResetNeeded, match="Reset before"):
            env.step(ZERO)
    finally:
        env.close()
