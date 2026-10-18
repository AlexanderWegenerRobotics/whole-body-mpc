from __future__ import annotations

import argparse
from pathlib import Path

from wbc.render import load_model, render_run
from wbc.types import REPO_ROOT


def main() -> None:
    p = argparse.ArgumentParser(description="render a logged run to video")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--camera", default=None,
                   help="named model camera; omit to follow the CoM in xy")
    p.add_argument("--distance", type=float, default=4.0)
    p.add_argument("--lookat-z", type=float, default=0.6)
    p.add_argument("--azimuth", type=float, default=None)
    p.add_argument("--elevation", type=float, default=None)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args()

    run_dir = args.run_dir
    out = args.out or run_dir / "video.mp4"

    path, frames = render_run(
        log_path=run_dir / "log.h5",
        out_path=out,
        model=load_model(run_dir, REPO_ROOT / "model" / "scene_mpc.xml"),
        camera=args.camera,
        fps=args.fps,
        speed=args.speed,
        width=args.width,
        height=args.height,
        distance=args.distance,
        lookat_z=args.lookat_z,
        azimuth=args.azimuth,
        elevation=args.elevation,
    )
    print(f"{frames} frames -> {path}")


if __name__ == "__main__":
    main()
