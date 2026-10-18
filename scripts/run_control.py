from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from wbc.control import PostureHold, Tracker, ankle_stability_margin
from wbc.model import RobotModel
from wbc.recording import (
    HDF5Backend,
    NullRecorder,
    Recorder,
    make_run_dir,
    run_metadata,
    write_meta,
)
from wbc.render import load_model, render_run, save_flat_model
from wbc.sim import MujocoPlant, Viewer
from wbc.types import NQ, NU, NV, NX, REPO_ROOT, ControlConfig, SimConfig, ViewerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="posture hold under the plan/tracker path")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--keyframe", default="home")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--camera", default=None)
    p.add_argument("--rtf", type=float, default=1.0)
    p.add_argument("--name", default="hold")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--video", action="store_true")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--video-speed", type=float, default=1.0)
    p.add_argument("--kp-scale", type=float, default=6.0)
    p.add_argument("--kd-scale", type=float, default=0.08)
    p.add_argument("--push", type=float, default=0.0, help="forward force on the pelvis [N]")
    p.add_argument("--push-at", type=float, default=2.0)
    p.add_argument("--push-for", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = REPO_ROOT / "model"
    urdf = model_dir / "g1_29dof.urdf"
    scene = model_dir / "scene_mpc.xml"

    robot = RobotModel(urdf, [model_dir])
    sim_cfg = SimConfig(scene=scene, keyframe=args.keyframe)
    plant = MujocoPlant(sim_cfg, robot)
    ctrl_cfg = ControlConfig(kp_scale=args.kp_scale, kd_scale=args.kd_scale)

    state = plant.reset()
    planner = PostureHold(robot, ctrl_cfg, state.q, plant.dt)
    tracker = Tracker(robot)

    print(planner.report())
    margin = ankle_stability_margin(robot, planner.kp, float(plant.centroidal().com[2]))
    print(
        f"ankle k {margin['k_ankle_total']:.0f} vs m*g*l {margin['m_g_l']:.0f} Nm/rad"
        f"  -> {'stable' if margin['stable'] else 'TOPPLES'}"
        f", feet tip near {np.degrees(margin['lean_at_cop_limit_rad']):.1f} deg"
    )
    if not margin["stable"]:
        print(
            f"  the ankle spring is weaker than gravity: use "
            f"--kp-scale above {margin['min_kp_scale']:.1f} (2x that for margin)"
        )

    if args.no_log:
        run_dir, rec = None, NullRecorder()
    else:
        run_dir = make_run_dir(REPO_ROOT / "runs", args.name)
        try:
            save_flat_model(run_dir / "model.xml", plant.model, model_dir / "meshes")
        except Exception as exc:
            print(f"could not save flattened model xml: {exc}")
        rec = Recorder(HDF5Backend(run_dir / "log.h5"))

    contacts = list(sim_cfg.contact_geoms)
    rec.register("sim.state", q=NQ, v=NV)
    rec.register("sim.qpos_raw", qpos=plant.model.nq)
    rec.register("sim.grf", **{name: 3 for name in contacts})
    rec.register("sim.centroidal", com=3, linmom=3, angmom=3)
    rec.register("ctrl.tau", star=NU, applied=NU, saturation=NU)
    rec.register("ctrl.error", dq=NV, dv=NV)
    rec.register("mpc.plan", tau0=NU, x0=NX)
    rec.register("mpc.solve", iterations=(), cost=(), wall_ms=())

    def log(state, tau_star, plan):
        applied, saturation = plant.last_torque()
        centroidal = plant.centroidal()
        i = plan.index_at(state.t)
        error = robot.tangent_error(state.q, state.v, plan.q_ref(i), plan.v_ref(i))
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
        rec.record("ctrl.error", state.t, dq=error[:NV], dv=error[NV:])

    viewer_cfg = ViewerConfig(
        enabled=args.viewer, realtime_factor=args.rtf, camera=args.camera
    )
    plan, stats = planner(state)
    pushing = False
    step = 0

    with Viewer(plant, viewer_cfg) as viewer:
        while state.t < args.duration and viewer.is_running:
            if step % ctrl_cfg.decimation == 0:
                plan, stats = planner(state)
                rec.record("mpc.plan", state.t, tau0=plan.taus[0], x0=plan.xs[0])
                rec.record(
                    "mpc.solve",
                    state.t,
                    iterations=stats.iterations,
                    cost=stats.cost,
                    wall_ms=stats.wall_ms,
                )

            if args.push > 0.0:
                active = args.push_at <= state.t < args.push_at + args.push_for
                if active and not pushing:
                    plant.apply_wrench("pelvis", np.array([args.push, 0, 0, 0, 0, 0]))
                    pushing = True
                elif pushing and not active:
                    plant.clear_wrench()
                    pushing = False

            tau = tracker(plan, state)
            state = plant.step(tau)
            log(state, tau, plan)
            viewer.sync(state.t)
            step += 1

    meta = run_metadata(
        config={
            "sim": asdict(sim_cfg),
            "control": asdict(ctrl_cfg),
            "viewer": asdict(viewer_cfg),
            "args": vars(args),
            "dt": plant.dt,
            "mass": robot.mass,
            "ankle_margin": margin,
        },
        files=[urdf, scene, model_dir / "g1_29dof_mpc.xml"],
        repo_root=REPO_ROOT,
    )
    rec.close(meta)
    if run_dir is not None:
        write_meta(run_dir / "meta.yaml", meta)

    i = plan.index_at(state.t)
    error = robot.tangent_error(state.q, state.v, plan.q_ref(i), plan.v_ref(i))
    print(f"t          {state.t:.3f} s")
    print(f"base z     {state.q[2]:.4f} m  (start 0.7837)")
    print(f"joint err  max {np.abs(error[6:NV]).max():.4f} rad")
    print(f"saturation max {plant.last_torque()[1].max():.3f}")
    print(f"GRF sum    {np.round(sum(plant.grf().values()), 1)}")
    if run_dir is not None:
        print(f"run dir    {run_dir}")

    if args.video and run_dir is not None:
        path, frames = render_run(
            log_path=run_dir / "log.h5",
            out_path=run_dir / "video.mp4",
            model=load_model(run_dir, scene),
            fps=args.fps,
            speed=args.video_speed,
        )
        print(f"video      {path} ({frames} frames)")


if __name__ == "__main__":
    main()
