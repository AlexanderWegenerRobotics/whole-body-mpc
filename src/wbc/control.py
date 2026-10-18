from __future__ import annotations

import numpy as np

from wbc.model import RobotModel
from wbc.types import (
    NDX,
    NQ,
    NU,
    NV,
    ControlConfig,
    ControlPlan,
    RobotState,
    SolveStats,
)


class Tracker:
    """Evaluates a ControlPlan at the current time and produces joint torques.

    tau = tau*(t) + K(t) @ (x - x*(t))

    The state error is a manifold operation, and the plan is indexed by
    absolute time rather than by step count, so a stale plan degrades into a
    slightly old reference instead of silently sliding out of phase.
    """

    def __init__(self, robot: RobotModel) -> None:
        self.robot = robot

    def __call__(self, plan: ControlPlan, state: RobotState) -> np.ndarray:
        i = plan.index_at(state.t)
        error = self.robot.tangent_error(
            state.q, state.v, plan.q_ref(i), plan.v_ref(i)
        )
        return plan.taus[i] + plan.Ks[i] @ error


class PostureHold:
    """Gravity-compensated joint-space PD, expressed as a one-knot ControlPlan.

    Holds a posture. It cannot balance: K has zero columns for the floating
    base, so there is no feedback on where the body is. The robot stays up
    because stiff ankles make it an inverted pendulum on a torsional spring,
    which is stable only while k_ankle > m*g*l and the centre of pressure
    stays inside the feet.
    """

    def __init__(
        self,
        robot: RobotModel,
        cfg: ControlConfig,
        q_ref: np.ndarray,
        dt: float,
    ) -> None:
        self.robot = robot
        self.cfg = cfg
        self.q_ref = np.array(q_ref, dtype=np.float64)

        self.kp = cfg.kp_scale * robot.tau_limit
        requested = cfg.kd_scale * robot.tau_limit
        self.kd_ceiling = cfg.kd_safety * 2.0 * robot.mass_diagonal(self.q_ref) / dt
        self.kd = np.minimum(requested, self.kd_ceiling)
        self.kd_clipped = requested > self.kd_ceiling

        gains = np.zeros((NU, NDX))
        gains[:, 6:NV] = -np.diag(self.kp)
        gains[:, NV + 6 :] = -np.diag(self.kd)

        state_ref = np.concatenate([self.q_ref, np.zeros(NV)])
        self._plan = ControlPlan(
            t0=0.0,
            ts=np.zeros(1),
            xs=state_ref[None, :],
            taus=robot.gravity_torque(self.q_ref)[None, :],
            Ks=gains[None, :, :],
        )
        self._stats = SolveStats()

    def __call__(self, state: RobotState) -> tuple[ControlPlan, SolveStats]:
        return self._plan, self._stats

    def report(self) -> str:
        lines = [
            f"kp  min {self.kp.min():7.1f}  max {self.kp.max():7.1f}",
            f"kd  min {self.kd.min():7.2f}  max {self.kd.max():7.2f}",
            f"tau* norm {np.linalg.norm(self._plan.taus[0]):.2f} Nm",
        ]
        if self.kd_clipped.any():
            names = [
                n for n, c in zip(self.robot.joint_names, self.kd_clipped) if c
            ]
            lines.append(
                f"kd capped by the explicit-integration bound on {len(names)} joints: "
                + ", ".join(names[:6])
                + (" ..." if len(names) > 6 else "")
            )
        return "\n".join(lines)


def ankle_stability_margin(robot: RobotModel, kp: np.ndarray, com_height: float) -> dict:
    """Compare the ankle pitch stiffness against m*g*l and the foot geometry."""
    idx = [i for i, n in enumerate(robot.joint_names) if n.endswith("ankle_pitch_joint")]
    k_total = float(sum(kp[i] for i in idx))
    tau_total = float(sum(robot.tau_limit[i] for i in idx))
    toppling = robot.mass * 9.81 * com_height
    return {
        "k_ankle_total": k_total,
        "m_g_l": toppling,
        "stable": k_total > toppling,
        "min_kp_scale": toppling / tau_total if tau_total else 0.0,
        "lean_at_cop_limit_rad": (robot.mass * 9.81 * 0.12) / k_total if k_total else 0.0,
    }
