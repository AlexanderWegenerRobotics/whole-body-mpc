import numpy as np
import pinocchio as pin
import pytest

from helpers import random_q

from wbc.types import NQ, NU, NV


def test_sizes_and_mass(robot):
    assert (robot.nq, robot.nv, robot.nu) == (NQ, NV, NU)
    assert len(robot.joint_names) == NU
    assert robot.mass == pytest.approx(33.341, abs=1e-3)


def test_geometry_loaded(robot):
    assert len(robot.collision_model.geometryObjects) > 0
    assert len(robot.visual_model.geometryObjects) > 0


def test_limits_are_sane(robot):
    assert robot.tau_limit.shape == (NU,)
    assert np.all(robot.tau_limit > 0.0)
    assert np.all(robot.q_lower < robot.q_upper)
    assert robot.tau_limit.max() == pytest.approx(139.0)
    assert robot.tau_limit.min() == pytest.approx(5.0)


def test_gravity_torque_drops_the_base_rows(robot):
    q = robot.neutral()
    tau = robot.gravity_torque(q)
    full = pin.computeGeneralizedGravity(robot.model, robot.data, q)
    assert tau.shape == (NU,)
    assert np.allclose(tau, full[6:])


def test_tangent_error_is_zero_at_the_reference(robot, rng):
    q = random_q(robot, rng)
    v = rng.normal(size=NV)
    assert np.allclose(robot.tangent_error(q, v, q, v), 0.0, atol=1e-12)


def test_tangent_error_integrates_back(robot, rng):
    q_ref = random_q(robot, rng)
    q = random_q(robot, rng)
    v = rng.normal(size=NV)
    v_ref = rng.normal(size=NV)

    error = robot.tangent_error(q, v, q_ref, v_ref)
    assert error.shape == (2 * NV,)

    back = pin.integrate(robot.model, q_ref, error[:NV])
    assert np.allclose(back[:3], q[:3], atol=1e-10)
    assert np.allclose(back[7:], q[7:], atol=1e-10)
    assert np.allclose(pin.difference(robot.model, back, q), 0.0, atol=1e-10)
    assert np.allclose(error[NV:], v - v_ref)


def test_quaternion_double_cover_is_handled(robot, rng):
    """q and -q are the same rotation, so configurations must be compared on
    the manifold. A component-wise comparison of q is not a valid test."""
    q = random_q(robot, rng)
    flipped = q.copy()
    flipped[3:7] *= -1.0

    assert not np.allclose(q, flipped)
    assert np.allclose(pin.difference(robot.model, q, flipped), 0.0, atol=1e-12)


def test_mass_diagonal_is_positive(robot):
    diag = robot.mass_diagonal(robot.neutral())
    assert diag.shape == (NU,)
    assert np.all(diag > 0.0)
