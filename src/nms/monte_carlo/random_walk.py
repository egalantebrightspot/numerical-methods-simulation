"""
Random walk generators for Monte Carlo simulation.

Provides symmetric random walks in one and two dimensions with support
for custom step distributions, reproducible seeding, and ensemble
generation for statistical analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def random_walk_1d(
    n_steps: int,
    n_walks: int = 1,
    step_size: float = 1.0,
    *,
    p_right: float = 0.5,
    start: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate 1-D symmetric (or biased) random walks.

    Parameters
    ----------
    n_steps : int
        Number of steps per walk.
    n_walks : int
        Number of independent walks to simulate.
    step_size : float
        Magnitude of each step.
    p_right : float
        Probability of stepping in the positive direction.
    start : float
        Starting position for every walk.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    paths : np.ndarray, shape (n_walks, n_steps + 1)
        Each row is one walk, starting from *start*.
    """
    rng = np.random.default_rng(seed)
    steps = rng.choice(
        [step_size, -step_size],
        size=(n_walks, n_steps),
        p=[p_right, 1 - p_right],
    )
    paths = np.empty((n_walks, n_steps + 1))
    paths[:, 0] = start
    np.cumsum(steps, axis=1, out=paths[:, 1:])
    paths[:, 1:] += start
    return paths


def random_walk_2d(
    n_steps: int,
    n_walks: int = 1,
    step_size: float = 1.0,
    *,
    start: ArrayLike = (0.0, 0.0),
    seed: int | None = None,
) -> np.ndarray:
    """Generate 2-D random walks with uniformly random direction.

    Parameters
    ----------
    n_steps : int
        Number of steps per walk.
    n_walks : int
        Number of independent walks.
    step_size : float
        Fixed step length.
    start : (x0, y0)
        Starting coordinates.
    seed : int or None
        Random seed.

    Returns
    -------
    paths : np.ndarray, shape (n_walks, n_steps + 1, 2)
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=float)

    angles = rng.uniform(0, 2 * np.pi, size=(n_walks, n_steps))
    dx = step_size * np.cos(angles)
    dy = step_size * np.sin(angles)

    paths = np.empty((n_walks, n_steps + 1, 2))
    paths[:, 0, :] = start
    paths[:, 1:, 0] = np.cumsum(dx, axis=1) + start[0]
    paths[:, 1:, 1] = np.cumsum(dy, axis=1) + start[1]
    return paths
