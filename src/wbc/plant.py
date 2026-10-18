import mujoco
import time
from dataclasses import dataclass
import yaml
import numpy as np



@dataclass(frozen=True)
class SimConfig:
    scene: str
    keyframe: str = "home"
    n_substeps: int = 1
    clip_torque: bool = True
    override_timestep: float | None = None

class MujocoPlant:
    def __init__(self, cfg: SimConfig, robot: RobotModel):
        dt: float
        nq: int; nv: int; nu: int


    def reset(self, *, q=None, v=None, keyframe=None) -> RobotState:
        pass
    def step(self, tau: np.ndarray) -> RobotState:
        pass


    def run(self):
        pass