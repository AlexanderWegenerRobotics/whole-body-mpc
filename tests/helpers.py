import numpy as np

from wbc.types import NQ


def random_q(robot, rng) -> np.ndarray:
    """A random configuration with a valid free-flyer.

    pin.randomConfiguration samples between the model's position limits, and a
    free-flyer's translation limits are +-inf, so it returns inf for the base.
    """
    q = np.empty(NQ)
    q[0:3] = rng.normal(size=3)
    quat = rng.normal(size=4)
    q[3:7] = quat / np.linalg.norm(quat)
    lower = np.where(np.isfinite(robot.q_lower), robot.q_lower, -np.pi)
    upper = np.where(np.isfinite(robot.q_upper), robot.q_upper, np.pi)
    q[7:] = rng.uniform(lower, upper)
    return q
