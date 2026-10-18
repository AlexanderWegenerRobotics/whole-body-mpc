from pathlib import Path
import numpy as np
import pinocchio as pin

from wbc.types import Centroidal, NQ, NV, NU


class RobotModel:
    model: pin.Model
    data: pin.Data
    collision_model: pin.GeometryModel
    collision_data: pin.GeometryData
    visual_model: pin.GeometryModel
    visual_data: pin.GeometryData
    nq: int; nv: int; nu: int
    mass: float
    joint_names: list[str]
    q_lower: np.ndarray; q_upper: np.ndarray
    tau_limit: np.ndarray
    v_limit: np.ndarray

    def __init__(self, urdf_path: Path, package_dirs: list[Path]) -> None:
        urdf_path = Path(urdf_path).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.package_dirs = [Path(d).resolve() for d in package_dirs]
        dirs = [str(d) for d in self.package_dirs]

        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        self.collision_model = pin.buildGeomFromUrdf(
            self.model,
            str(urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=dirs,
        )
        self.collision_data = self.collision_model.createData()

        self.visual_model = pin.buildGeomFromUrdf(
            self.model,
            str(urdf_path),
            pin.GeometryType.VISUAL,
            package_dirs=dirs,
        )
        self.visual_data = self.visual_model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.nv - 6

        if (self.nq, self.nv, self.nu) != (NQ, NV, NU):
            raise ValueError(
                f"expected nq/nv/nu = {NQ}/{NV}/{NU}, "
                f"got {self.nq}/{self.nv}/{self.nu} — "
                f"missing JointModelFreeFlyer?"
            )

        self.joint_names = list(self.model.names)[2:]
        self.mass = pin.computeTotalMass(self.model)

        self.q_lower = self.model.lowerPositionLimit[7:].copy()
        self.q_upper = self.model.upperPositionLimit[7:].copy()
        self.tau_limit = self.model.effortLimit[6:].copy()
        self.v_limit = self.model.velocityLimit[6:].copy()

    def frame_id(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid >= self.model.nframes:
            raise KeyError(f"no frame named {name!r}")
        return int(fid)

    def forward(self, q: np.ndarray, v: np.ndarray | None = None) -> None:
        if v is None:
            pin.forwardKinematics(self.model, self.data, q)
        else:
            pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)

    def frame_pose(self, q: np.ndarray, name: str) -> pin.SE3:
        self.forward(q)
        return self.data.oMf[self.frame_id(name)].copy()

    def centroidal(self, q: np.ndarray, v: np.ndarray) -> Centroidal:
        pin.centerOfMass(self.model, self.data, q, v)
        pin.computeCentroidalMomentum(self.model, self.data, q, v)
        return Centroidal(
            com=np.array(self.data.com[0]),
            linmom=np.array(self.data.hg.linear),
            angmom=np.array(self.data.hg.angular),
        )

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Actuated rows of g(q). The first six are the base wrench the
        ground supplies through the feet, and no actuator can produce them."""
        return np.array(pin.computeGeneralizedGravity(self.model, self.data, q)[6:])

    def mass_diagonal(self, q: np.ndarray) -> np.ndarray:
        """Diagonal of the joint-space mass matrix, actuated rows only.

        Sets the explicit-integration damping ceiling: kd < 2 * M_ii / dt.
        """
        mass = pin.crba(self.model, self.data, q)
        return np.array(np.diag(mass)[6:])

    def tangent_error(
        self,
        q: np.ndarray,
        v: np.ndarray,
        q_ref: np.ndarray,
        v_ref: np.ndarray,
    ) -> np.ndarray:
        """x - x_ref on the manifold, (NDX,).

        The configuration part is pin.difference, not a subtraction: q holds a
        unit quaternion and subtracting two of them is meaningless.
        """
        return np.concatenate([pin.difference(self.model, q_ref, q), v - v_ref])

    def neutral(self) -> np.ndarray:
        return pin.neutral(self.model)


if __name__ == "__main__":
    from wbc.types import REPO_ROOT

    model_dir = REPO_ROOT / "model"
    robot = RobotModel(model_dir / "g1_29dof.urdf", [REPO_ROOT / "model"])

    print(robot.nq, robot.nv, robot.mass)
    print(len(robot.collision_model.geometryObjects), "collision objects")
    print(len(robot.visual_model.geometryObjects), "visual objects")

    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    q0 = pin.neutral(robot.model)
    q0[2] = 0.7837
    viz.display(q0)

    input("press enter to close")
