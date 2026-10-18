import mujoco
import numpy as np
import pytest

from wbc.sim import MjPinMap
from wbc.types import NQ, NU, NV


def _random_mj_state(rng, nq, nv):
    qpos = np.empty(nq)
    qpos[0:3] = rng.normal(size=3)
    quat = rng.normal(size=4)
    qpos[3:7] = quat / np.linalg.norm(quat)
    qpos[7:] = rng.normal(size=nq - 7)
    return qpos, rng.normal(size=nv)


def test_index_map_is_derived_not_assumed(plant):
    m = plant.map
    assert np.array_equal(m.qpos_adr, np.arange(7, NQ))
    assert np.array_equal(m.dof_adr, np.arange(6, NV))
    assert np.array_equal(m.actuator_adr, np.arange(NU))


def test_conversion_round_trip(plant, rng):
    for _ in range(200):
        qpos, qvel = _random_mj_state(rng, plant.model.nq, plant.model.nv)
        q, v = plant.map.mj_to_pin(qpos, qvel)
        back_qpos, back_qvel = plant.map.pin_to_mj(q, v)
        assert np.allclose(qpos, back_qpos, atol=1e-14)
        assert np.allclose(qvel, back_qvel, atol=1e-12)


def test_base_velocity_frame_against_mujoco(plant, rng):
    """pinocchio v[0:6] is the LOCAL spatial velocity of the body FRAME.

    mjOBJ_BODY asks mj_objectVelocity for the body's INERTIAL frame, which is
    offset to the link centre of mass; mjOBJ_XBODY is the frame that qpos
    describes. The difference is omega x r_com, a few centimetres per rad/s.
    """
    qpos, qvel = _random_mj_state(rng, plant.model.nq, plant.model.nv)
    plant.data.qpos[:] = qpos
    plant.data.qvel[:] = qvel
    mujoco.mj_forward(plant.model, plant.data)

    q, v = plant.map.mj_to_pin(plant.data.qpos, plant.data.qvel)
    bid = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    local = np.empty(6)
    mujoco.mj_objectVelocity(
        plant.model, plant.data, mujoco.mjtObj.mjOBJ_XBODY, bid, local, 1
    )

    assert np.allclose(v[0:3], local[3:6], atol=1e-10)
    assert np.allclose(v[3:6], local[0:3], atol=1e-10)


def test_forward_kinematics_agree(plant, robot):
    """Gate G2 — pinocchio and MuJoCo must place the same links identically."""
    state = plant.reset()
    robot.forward(state.q)

    mj_names = {
        mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(plant.model.nbody)
    } - {None}
    checked = 0
    for name in sorted(mj_names):
        fid = robot.model.getFrameId(name)
        if fid >= robot.model.nframes:
            continue
        bid = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(
            robot.data.oMf[fid].translation, plant.data.xpos[bid], atol=1e-5
        ), name
        checked += 1
    assert checked >= 20


def test_centroidal_cross_check(plant, robot, rng):
    """MuJoCo's subtree sensors against pinocchio's centroidal momentum."""
    state = plant.reset()
    v = rng.normal(scale=0.3, size=NV)
    plant.reset(q=state.q, v=v)

    mj = plant.centroidal()
    pn = robot.centroidal(state.q, v)

    assert np.allclose(mj.com, pn.com, atol=1e-6)
    assert np.allclose(mj.linmom, pn.linmom, atol=1e-5)
    assert np.allclose(mj.angmom, pn.angmom, atol=1e-5), (
        f"angular momentum disagrees: mujoco {mj.angmom} vs pinocchio {pn.angmom}. "
        f"A difference shaped like com x linmom = {np.cross(mj.com, mj.linmom)} "
        f"means the two use different reference points."
    )


def test_reset_is_deterministic(plant):
    a = plant.reset()
    b = plant.reset()
    assert np.array_equal(a.q, b.q)
    assert np.array_equal(a.v, b.v)
    assert a.t == b.t == 0.0


def test_home_keyframe_touches_down(plant):
    """At reset the contacts carry load but not yet full weight: MuJoCo's
    contact force comes from penetration depth, which needs a few steps to
    develop. The settled weight check lives in test_control."""
    plant.reset(keyframe="home")
    total = sum(plant.grf().values())
    assert total[2] > 100.0
    assert total[2] > 3.0 * np.hypot(total[0], total[1])


def test_zero_torque_falls(plant):
    state = plant.reset()
    start = state.q[2]
    for _ in range(250):
        state = plant.step(np.zeros(NU))
    assert state.q[2] < start - 0.05
    assert np.all(np.isfinite(state.q)) and np.all(np.isfinite(state.v))


def test_torque_is_clipped_to_ctrlrange(plant, robot):
    plant.reset()
    plant.step(np.full(NU, 1e4))
    applied, saturation = plant.last_torque()
    assert np.allclose(applied, robot.tau_limit)
    assert np.allclose(saturation, 1.0)


def test_state_arrays_do_not_alias_mjdata(plant):
    a = plant.reset()
    plant.step(np.zeros(NU))
    b = plant.state()
    assert not np.shares_memory(a.q, plant.data.qpos)
    assert not np.array_equal(a.q, b.q)
