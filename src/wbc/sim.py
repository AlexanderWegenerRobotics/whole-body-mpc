import time
from pathlib import Path

import numpy as np
import mujoco
import pinocchio as pin

from wbc.model import RobotModel
from wbc.types import SimConfig, RobotState, Centroidal, ViewerConfig, NQ, NV, NU

class MjPinMap:
    """Index and convention translation between MuJoCo and Pinocchio."""

    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_adr: np.ndarray
    nq_mj: int
    nv_mj: int

    def __init__(self, mj_model: mujoco.MjModel, robot: RobotModel) -> None:
        if (mj_model.nq, mj_model.nv, mj_model.nu) != (NQ, NV, NU):
            raise ValueError(
                f"MuJoCo model has nq/nv/nu = "
                f"{mj_model.nq}/{mj_model.nv}/{mj_model.nu}, expected {NQ}/{NV}/{NU}"
            )

        free = [j for j in range(mj_model.njnt) if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"expected exactly one free joint, found {len(free)}")
        if mj_model.jnt_qposadr[free[0]] != 0 or mj_model.jnt_dofadr[free[0]] != 0:
            raise ValueError("the free joint must occupy the first qpos/qvel slots")

        joint_to_actuator: dict[int, int] = {}
        for a in range(mj_model.nu):
            if mj_model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"actuator {a} is not a joint transmission")
            joint_to_actuator[int(mj_model.actuator_trnid[a, 0])] = a

        qpos_adr = np.empty(NU, dtype=np.int64)
        dof_adr = np.empty(NU, dtype=np.int64)
        actuator_adr = np.empty(NU, dtype=np.int64)

        for i, name in enumerate(robot.joint_names):
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"pinocchio joint {name!r} is not in the MuJoCo model")
            if mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"MuJoCo joint {name!r} is not a hinge")
            if jid not in joint_to_actuator:
                raise ValueError(f"MuJoCo joint {name!r} has no actuator")
            qpos_adr[i] = mj_model.jnt_qposadr[jid]
            dof_adr[i] = mj_model.jnt_dofadr[jid]
            actuator_adr[i] = joint_to_actuator[jid]

        if len(set(actuator_adr.tolist())) != NU:
            raise ValueError("two pinocchio joints map to the same MuJoCo actuator")

        self.qpos_adr = qpos_adr
        self.dof_adr = dof_adr
        self.actuator_adr = actuator_adr
        self.nq_mj = int(mj_model.nq)
        self.nv_mj = int(mj_model.nv)

    @staticmethod
    def _rotation(quat_wxyz: np.ndarray) -> np.ndarray:
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, np.ascontiguousarray(quat_wxyz, dtype=np.float64))
        return mat.reshape(3, 3)

    def mj_to_pin(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.empty(NQ)
        v = np.empty(NV)

        q[0:3] = qpos[0:3]
        q[3] = qpos[4]
        q[4] = qpos[5]
        q[5] = qpos[6]
        q[6] = qpos[3]
        q[7:] = qpos[self.qpos_adr]

        rot = self._rotation(qpos[3:7])
        v[0:3] = rot.T @ qvel[0:3]
        v[3:6] = qvel[3:6]
        v[6:] = qvel[self.dof_adr]

        return q, v

    def pin_to_mj(self, q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.empty(self.nq_mj)
        qvel = np.empty(self.nv_mj)

        qpos[0:3] = q[0:3]
        qpos[3] = q[6]
        qpos[4] = q[3]
        qpos[5] = q[4]
        qpos[6] = q[5]
        qpos[self.qpos_adr] = q[7:]

        rot = self._rotation(qpos[3:7])
        qvel[0:3] = rot @ v[0:3]
        qvel[3:6] = v[3:6]
        qvel[self.dof_adr] = v[6:]

        return qpos, qvel


class MujocoPlant:
    """The plant. It never sleeps, never renders and never logs."""

    model: mujoco.MjModel
    data: mujoco.MjData
    map: MjPinMap
    dt: float

    def __init__(self, cfg: SimConfig, robot: RobotModel) -> None:
        scene = Path(cfg.scene).resolve()
        if not scene.is_file():
            raise FileNotFoundError(f"scene not found: {scene}")

        self.cfg = cfg
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.map = MjPinMap(self.model, robot)
        self.dt = float(self.model.opt.timestep) * cfg.n_substeps

        self._ctrl_range = self.model.actuator_ctrlrange[self.map.actuator_adr].copy()
        self._ctrl_bound = np.maximum(np.abs(self._ctrl_range[:, 0]), np.abs(self._ctrl_range[:, 1]))

        if not np.allclose(robot.tau_limit, self._ctrl_range[:, 1], rtol=1e-6):
            worst = int(np.argmax(np.abs(robot.tau_limit - self._ctrl_range[:, 1])))
            raise ValueError(
                f"torque limits disagree at joint {robot.joint_names[worst]!r}: "
                f"URDF {robot.tau_limit[worst]} vs MJCF {self._ctrl_range[worst, 1]}"
            )

        self._sensor: dict[str, slice] = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name is not None:
                adr = int(self.model.sensor_adr[i])
                self._sensor[name] = slice(adr, adr + int(self.model.sensor_dim[i]))

        self._site: dict[str, int] = {}
        for i in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name is not None:
                self._site[name] = i

        self._contact_geom: dict[str, int] = {}
        for name in cfg.contact_geoms:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"contact geom {name!r} is not in the scene")
            self._contact_geom[name] = gid
        self._geom_to_contact = {g: n for n, g in self._contact_geom.items()}

        self._tau_applied = np.zeros(NU)
        self.reset()

    def reset(self, *, q=None, v=None, keyframe: str | None = None) -> RobotState:
        mujoco.mj_resetData(self.model, self.data)

        if q is None:
            key = self.cfg.keyframe if keyframe is None else keyframe
            if key is not None:
                kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
                if kid < 0:
                    raise ValueError(f"keyframe {key!r} is not in the model")
                mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        else:
            qpos, qvel = self.map.pin_to_mj(q, np.zeros(NV) if v is None else v)
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel

        self.data.time = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self._tau_applied[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def step(self, tau: np.ndarray) -> RobotState:
        tau = np.asarray(tau, dtype=np.float64)
        if tau.shape != (NU,):
            raise ValueError(f"tau must have shape ({NU},), got {tau.shape}")

        if self.cfg.clip_torque:
            tau = np.clip(tau, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

        self._tau_applied[:] = tau
        self.data.ctrl[self.map.actuator_adr] = tau

        for _ in range(self.cfg.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.state()

    def state(self) -> RobotState:
        q, v = self.map.mj_to_pin(self.data.qpos, self.data.qvel)
        return RobotState(t=float(self.data.time), q=q, v=v)

    def qpos_raw(self) -> np.ndarray:
        return self.data.qpos.copy()

    def grf(self) -> dict[str, np.ndarray]:
        out = {name: np.zeros(3) for name in self._contact_geom}
        wrench = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            name = self._geom_to_contact.get(g1, self._geom_to_contact.get(g2))
            if name is None:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            force = con.frame.reshape(3, 3).T @ wrench[:3]
            if g1 in self._geom_to_contact:
                force = -force
            out[name] += force

        return out

    def centroidal(self) -> Centroidal:
        return Centroidal(
            com=self.sensor("com"),
            linmom=self.robot.mass * self.sensor("com_linvel"),
            angmom=self.sensor("com_angmom"),
        )

    def site_pose(self, name: str) -> pin.SE3:
        sid = self._site.get(name)
        if sid is None:
            raise KeyError(f"no site named {name!r}")
        return pin.SE3(
            self.data.site_xmat[sid].reshape(3, 3).copy(),
            self.data.site_xpos[sid].copy(),
        )

    def sensor(self, name: str) -> np.ndarray:
        span = self._sensor.get(name)
        if span is None:
            raise KeyError(f"no sensor named {name!r}")
        return self.data.sensordata[span].copy()

    def last_torque(self) -> tuple[np.ndarray, np.ndarray]:
        return self._tau_applied.copy(), np.abs(self._tau_applied) / self._ctrl_bound

    def apply_wrench(self, body: str, wrench: np.ndarray) -> None:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise KeyError(f"no body named {body!r}")
        w = np.asarray(wrench, dtype=np.float64)
        if w.shape != (6,):
            raise ValueError("wrench must be (6,): force then torque, world frame")
        self.data.xfrc_applied[bid] = w

    def clear_wrench(self) -> None:
        self.data.xfrc_applied[:] = 0.0


class Viewer:
    """Optional passive viewer. A no-op when disabled, so the loop has no branch.

    On macOS the passive viewer needs the UI on the main thread, which means
    running under `mjpython`, not `python`.
    """

    def __init__(self, plant: MujocoPlant, cfg: ViewerConfig) -> None:
        self.plant = plant
        self.cfg = cfg
        self._handle = None
        self._wall_offset: float | None = None

    def __enter__(self) -> "Viewer":
        if not self.cfg.enabled:
            return self

        import mujoco.viewer

        try:
            self._handle = mujoco.viewer.launch_passive(self.plant.model, self.plant.data)
        except RuntimeError as exc:
            raise RuntimeError(
                "could not open the passive viewer — on macOS run this with "
                "mjpython instead of python"
            ) from exc

        if self.cfg.camera is not None:
            cid = mujoco.mj_name2id(
                self.plant.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cfg.camera
            )
            if cid < 0:
                raise ValueError(f"no camera named {self.cfg.camera!r} in the scene")
            self._handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._handle.cam.fixedcamid = cid

        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def sync(self, sim_time: float) -> None:
        if self._handle is None:
            return

        rtf = self.cfg.realtime_factor
        if rtf > 0.0:
            if self._wall_offset is None:
                self._wall_offset = time.perf_counter() - sim_time / rtf
            delay = self._wall_offset + sim_time / rtf - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

        self._handle.sync()

    @property
    def is_running(self) -> bool:
        if self._handle is None:
            return True
        return self._handle.is_running()


if __name__ == "__main__":
    import argparse

    from wbc.types import REPO_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--keyframe", default="home")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--rtf", type=float, default=1.0)
    args = parser.parse_args()

    robot = RobotModel(REPO_ROOT / "model" / "g1_29dof.urdf", [REPO_ROOT / "model"])
    plant = MujocoPlant(
        SimConfig(scene=REPO_ROOT / "model" / "scene_mpc.xml", keyframe=args.keyframe),
        robot,
    )

    state = plant.reset()
    tau = np.zeros(NU)
    viewer_cfg = ViewerConfig(enabled=args.viewer, realtime_factor=args.rtf)

    with Viewer(plant, viewer_cfg) as viewer:
        while state.t < args.duration and viewer.is_running:
            state = plant.step(tau)
            viewer.sync(state.t)

    grf = plant.grf()
    print(f"t        {state.t:.3f} s")
    print(f"base z   {state.q[2]:.4f} m")
    print(f"GRF sum  {np.round(sum(grf.values()), 2)}")
    print(f"com      {np.round(plant.centroidal().com, 4)}")
