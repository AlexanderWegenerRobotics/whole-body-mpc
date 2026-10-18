from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from wbc.recording import RunReader


def _ensure_offscreen(model: mujoco.MjModel, width: int, height: int) -> None:
    try:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    except AttributeError:
        pass


def tracking_camera(
    model: mujoco.MjModel,
    distance: float = 4.0,
    lookat_z: float = 0.6,
    azimuth: float | None = None,
    elevation: float | None = None,
) -> mujoco.MjvCamera:
    """Free camera that follows the robot in xy at a fixed height.

    A trackcom camera follows the centre of mass in z as well, so a fallen
    robot drags the camera down to the floor plane. This keeps the eye level
    fixed and only pans horizontally.
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.lookat[:] = (0.0, 0.0, lookat_z)
    try:
        default_az = float(model.vis.global_.azimuth)
        default_el = float(model.vis.global_.elevation)
    except AttributeError:
        default_az, default_el = 130.0, -20.0
    cam.azimuth = default_az if azimuth is None else azimuth
    cam.elevation = default_el if elevation is None else elevation
    return cam


def render_run(
    log_path: Path,
    out_path: Path,
    model: mujoco.MjModel,
    camera: str | int | None = None,
    fps: int = 60,
    speed: float = 1.0,
    width: int = 1280,
    height: int = 720,
    distance: float = 4.0,
    lookat_z: float = 0.6,
    azimuth: float | None = None,
    elevation: float | None = None,
    channel: str = "sim.qpos_raw",
    field: str = "qpos",
) -> tuple[Path, int]:
    """Render a recorded run to video, offline, from its logged qpos.

    camera=None follows the centre of mass in xy at a fixed height; a string
    or int selects a camera declared in the model instead.
    """
    import imageio.v2 as imageio

    if fps <= 0 or speed <= 0.0:
        raise ValueError("fps and speed must be positive")

    log_path, out_path = Path(log_path), Path(out_path)

    with RunReader(log_path) as run:
        view = run[channel]
        if len(view) < 2:
            raise ValueError(f"{channel} has {len(view)} samples, nothing to render")
        t0, t1 = float(view.t[0]), float(view.t[-1])
        times = np.arange(t0, t1, speed / fps)
        qpos = run.at(channel, times, field=field)

    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"logged qpos has width {qpos.shape[1]} but the model has nq={model.nq}"
        )

    _ensure_offscreen(model, width, height)
    data = mujoco.MjData(model)

    if isinstance(camera, str):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]
            raise ValueError(f"no camera {camera!r} in the model (have: {names})")

    tracking = camera is None
    cam = (
        tracking_camera(model, distance, lookat_z, azimuth, elevation)
        if tracking
        else camera
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height, width)
    writer = imageio.get_writer(
        out_path, fps=fps, macro_block_size=None, quality=8
    )

    try:
        for row in qpos:
            data.qpos[:] = row
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if tracking:
                cam.lookat[0] = data.subtree_com[0, 0]
                cam.lookat[1] = data.subtree_com[0, 1]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        if hasattr(renderer, "close"):
            renderer.close()

    return out_path, len(qpos)


def save_flat_model(path: Path, model: mujoco.MjModel, meshdir: Path) -> None:
    """Archive the flattened model XML with an absolute meshdir.

    mj_saveLastXML keeps the original relative meshdir, which would resolve
    against the run directory instead of the model package.
    """
    path = Path(path)
    mujoco.mj_saveLastXML(str(path), model)
    text = path.read_text()
    absolute = str(Path(meshdir).resolve())
    patched, n = re.subn(
        r'(<compiler\b[^>]*\bmeshdir=")[^"]*(")',
        lambda m: m.group(1) + absolute + m.group(2),
        text,
        count=1,
    )
    if n == 0:
        patched = re.sub(
            r"(<mujoco\b[^>]*>)",
            lambda m: m.group(1) + f'\n  <compiler meshdir="{absolute}"/>',
            text,
            count=1,
        )
    path.write_text(patched)


def load_model(run_dir: Path, fallback: Path) -> mujoco.MjModel:
    """Prefer the flattened model archived with the run; fall back to the scene."""
    archived = Path(run_dir) / "model.xml"
    if archived.is_file():
        try:
            return mujoco.MjModel.from_xml_path(str(archived))
        except ValueError as exc:
            print(f"archived model unusable ({exc}), falling back to {fallback}")
    return mujoco.MjModel.from_xml_path(str(Path(fallback)))
