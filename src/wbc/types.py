from dataclasses import dataclass
import numpy as np
from pathlib import Path

NQ, NV, NU = 36, 35, 29
NX = NQ + NV
NDX = 2 * NV

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

@dataclass(frozen=True, slots=True)
class ControlPlan:
    """A time-stamped feedforward trajectory with local feedback gains.

    t0 is absolute simulation time; ts are absolute knot times. The tracker
    evaluates the plan by time, never by index, so a slow or stale plan
    degrades gracefully instead of silently shifting.

    xs:   (N, NX)        reference states, pinocchio convention
    taus: (N, NU)        feedforward joint torques
    Ks:   (N, NU, NDX)   feedback gains, tau = taus + Ks @ (x - xs)
    """
    t0: float
    ts: np.ndarray
    xs: np.ndarray
    taus: np.ndarray
    Ks: np.ndarray

    def __post_init__(self):
        n = len(self.ts)
        assert n >= 1
        assert self.xs.shape == (n, NX), self.xs.shape
        assert self.taus.shape == (n, NU), self.taus.shape
        assert self.Ks.shape == (n, NU, NDX), self.Ks.shape
        assert np.all(np.diff(self.ts) > 0.0) if n > 1 else True

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 1))

    def q_ref(self, i: int) -> np.ndarray:
        return self.xs[i, :NQ]

    def v_ref(self, i: int) -> np.ndarray:
        return self.xs[i, NQ:]


@dataclass(frozen=True, slots=True)
class SolveStats:
    iterations: int = 0
    cost: float = 0.0
    regularization: float = 0.0
    feasibility: float = 0.0
    wall_ms: float = 0.0


@dataclass(frozen=True)
class ControlConfig:
    """Posture-hold gains, scaled by each joint's torque authority.

    kp_i = kp_scale * tau_limit_i, so a 139 Nm knee is stiffer than a 5 Nm
    wrist. kd is scaled the same way but then capped at kd_safety * 2*M_ii/dt,
    the explicit-integration stability bound for a motor actuator with no
    velocity term for implicitfast to absorb.
    """
    kp_scale: float = 6.0
    kd_scale: float = 0.08
    kd_safety: float = 0.5
    decimation: int = 5
