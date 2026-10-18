from dataclasses import dataclass
import numpy as np
from pathlib import Path

NQ, NV, NU = 36, 35, 29

FOOT_CONTACT_GEOMS = (
    "left_foot_heel_in", "left_foot_heel_out",
    "left_foot_toe_in", "left_foot_toe_out",
    "right_foot_heel_in", "right_foot_heel_out",
    "right_foot_toe_in", "right_foot_toe_out",
)
REPO_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True, slots=True)
class RobotState:
    """Robot state, PINOCCHIO convention.

    q: [pos(3, world), quat_xyzw(4), joints(29)]
        NOTE MuJoCo stores the quaternion wxyz — converted on the way in.
    v: [linear(3, LOCAL/body frame), angular(3, LOCAL), joints(29)]
        NOTE MuJoCo's freejoint qvel is linear in WORLD, angular in body.
    t: simulation time [s]. Not wall clock.
    """
    t: float
    q: np.ndarray
    v: np.ndarray

    def __post_init__(self):
        assert self.q.shape == (NQ,) and self.v.shape == (NV,)
        assert self.q.dtype == np.float64 and self.v.dtype == np.float64
        assert abs(np.linalg.norm(self.q[3:7]) - 1.0) < 1e-6
        self.q.flags.writeable = False
        self.v.flags.writeable = False

    @property
    def x(self) -> np.ndarray:
        """Crocoddyl state vector, (71,)."""
        return np.concatenate([self.q, self.v])


@dataclass(frozen=True, slots=True)
class Centroidal:
    com: np.ndarray             # center of mass
    linmom: np.ndarray          # total linear momentum
    angmom: np.ndarray          # total angular momentumx

@dataclass(frozen=True)
class SimConfig:
    scene: Path
    keyframe: str = "home"
    n_substeps: int = 1
    clip_torque: bool = True
    contact_geoms: tuple[str, ...] = FOOT_CONTACT_GEOMS

@dataclass(frozen=True)
class ViewerConfig:
    enabled: bool = False
    realtime_factor: float = 1.0
    camera: str | None = None