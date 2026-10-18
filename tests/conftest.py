import numpy as np
import pytest

from wbc.model import RobotModel
from wbc.sim import MujocoPlant
from wbc.types import REPO_ROOT, SimConfig

MODEL_DIR = REPO_ROOT / "model"
URDF = MODEL_DIR / "g1_29dof.urdf"
SCENE = MODEL_DIR / "scene_mpc.xml"


@pytest.fixture(scope="session")
def robot() -> RobotModel:
    return RobotModel(URDF, [MODEL_DIR])


@pytest.fixture(scope="session")
def sim_cfg() -> SimConfig:
    return SimConfig(scene=SCENE)


@pytest.fixture
def plant(sim_cfg, robot) -> MujocoPlant:
    return MujocoPlant(sim_cfg, robot)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261017)
