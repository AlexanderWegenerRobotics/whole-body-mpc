from __future__ import annotations

import argparse
from dataclasses import asdict

import mujoco
import numpy as np

from wbc.model import RobotModel
from wbc.recording import (
    HDF5Backend,
    NullRecorder,
    Recorder,
    make_run_dir,
    run_metadata,
    write_meta,
)
from wbc.sim import MujocoPlant, Viewer
from wbc.types import NQ, NU, NV, REPO_ROOT, SimConfig, ViewerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="run the plant open loop and log it")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--keyframe", default="home")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--camera", default=None)
    p.add_argument("--rtf", type=float, default=1.0)
    p.add_argument("--name", default="run")
    p.add_argument("--no-log", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = REPO_ROOT / "model"
    urdf = model_dir / "g1_29dof.urdf"
    scene = model_dir / "scene_mpc.xml"

    robot = RobotModel(urdf, [model_dir])
    sim_cfg = SimConfig(scene=scene, keyframe=args.keyframe)
    plant = MujocoPlant(sim_cfg, robot)
    viewer_cfg = ViewerConfig(
        enabled=args.viewer, realtime_factor=args.rtf, camera=args.camera
    )

    if args.no_log:
        run_dir = None
        rec = NullRecorder()
    else:
        run_dir = make_run_dir(REPO_ROOT / "runs", args.name)
        try:
            mujoco.mj_saveLastXML(str(run_dir / "model.xml"), plant.model)
        except Exception as exc:
            print(f"could not save flattened model xml: {exc}")
        rec = Recorder(HDF5Backend(run_dir / "log.h5"))

    contacts = list(sim_cfg.contact_geoms)
    rec.register("sim.state", q=NQ, v=NV)
    rec.register("sim.qpos_raw", qpos=plant.model.nq)
    rec.register("sim.grf", **{name: 3 for name in contacts})
    rec.register("sim.centroidal", com=3, linmom=3, angmom=3)
    rec.register("ctrl.tau", star=NU, applied=NU, saturation=NU)

    def log(state, tau_star):
        applied, saturation = plant.last_torque()
        centroidal = plant.centroidal()
        rec.record("sim.state", state.t, q=state.q, v=state.v)
        rec.record("sim.qpos_raw", state.t, qpos=plant.qpos_raw())
        rec.record("sim.grf", state.t, **plant.grf())
        rec.record(
            "sim.centroidal",
            state.t,
            com=centroidal.com,
            linmom=centroidal.linmom,
            angmom=centroidal.angmom,
        )
        rec.record("ctrl.tau", state.t, star=tau_star, applied=applied, saturation=saturation)

    tau_star = np.zeros(NU)
    state = plant.reset()
    log(state, tau_star)

    with Viewer(plant, viewer_cfg) as viewer:
        while state.t < args.duration and viewer.is_running:
            state = plant.step(tau_star)
            log(state, tau_star)
            viewer.sync(state.t)

    meta = run_metadata(
        config={
            "sim": asdict(sim_cfg),
            "viewer": asdict(viewer_cfg),
            "args": vars(args),
            "dt": plant.dt,
            "mass": robot.mass,
        },
        files=[urdf, scene, model_dir / "g1_29dof_mpc.xml"],
        repo_root=REPO_ROOT,
    )
    rec.close(meta)
    if run_dir is not None:
        write_meta(run_dir / "meta.yaml", meta)

    print(f"t        {state.t:.3f} s")
    print(f"base z   {state.q[2]:.4f} m")
    print(f"GRF sum  {np.round(sum(plant.grf().values()), 2)}")
    if run_dir is not None:
        print(f"run dir  {run_dir}")


if __name__ == "__main__":
    main()
