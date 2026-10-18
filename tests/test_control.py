import numpy as np
import pytest

from wbc.control import PostureHold, Tracker, ankle_stability_margin
from wbc.types import NDX, NU, NV, ControlConfig


@pytest.fixture
def hold(plant, robot):
    state = plant.reset()
    return state, PostureHold(robot, ControlConfig(), state.q, plant.dt), Tracker(robot)


def test_plan_is_one_knot_with_zero_base_gains(hold):
    _, planner, _ = hold
    plan, _ = planner(None)
    assert plan.ts.shape == (1,)
    assert plan.Ks.shape == (1, NU, NDX)
    assert np.allclose(plan.Ks[0][:, :6], 0.0)
    assert np.allclose(plan.Ks[0][:, NV : NV + 6], 0.0)
    assert np.all(np.diag(plan.Ks[0][:, 6:NV]) < 0.0)


def test_tracker_returns_the_feedforward_at_the_reference(hold):
    state, planner, tracker = hold
    plan, _ = planner(state)
    assert np.allclose(tracker(plan, state), plan.taus[0], atol=1e-10)


def test_damping_respects_the_explicit_integration_bound(hold):
    _, planner, _ = hold
    assert np.all(planner.kd <= planner.kd_ceiling + 1e-12)
    assert planner.kd_clipped.any()


def test_posture_hold_settles_at_body_weight(plant, robot, hold):
    state, planner, tracker = hold
    plan, _ = planner(state)
    for _ in range(500):
        state = plant.step(tracker(plan, state))

    total = sum(plant.grf().values())
    assert total[2] == pytest.approx(robot.mass * 9.81, rel=0.05)
    assert state.q[2] == pytest.approx(0.7837, abs=0.02)
    assert np.abs(plant.last_torque()[1]).max() < 1.0


def test_posture_hold_tracks_its_reference(plant, robot, hold):
    state, planner, tracker = hold
    plan, _ = planner(state)
    for _ in range(500):
        state = plant.step(tracker(plan, state))

    error = robot.tangent_error(state.q, state.v, plan.q_ref(0), plan.v_ref(0))
    assert np.abs(error[6:NV]).max() < 0.05


def test_default_gains_clear_the_toppling_threshold(plant, robot, hold):
    _, planner, _ = hold
    margin = ankle_stability_margin(
        robot, planner.kp, float(plant.centroidal().com[2])
    )
    assert margin["stable"]
    assert margin["k_ankle_total"] > 2.0 * margin["m_g_l"] * 0.5
