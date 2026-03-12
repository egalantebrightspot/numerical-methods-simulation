"""
Poisson process generators.

A homogeneous Poisson process with rate lambda has:
    - inter-arrival times ~ Exp(lambda)
    - N(t) - N(s) ~ Poisson(lambda * (t - s))

The compound Poisson process adds i.i.d. jump sizes:
    X(t) = sum_{i=1}^{N(t)} Y_i
"""

from __future__ import annotations

import numpy as np


def poisson_process(
    rate: float,
    T: float,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Simulate arrival times of a homogeneous Poisson process.

    Parameters
    ----------
    rate : float
        Intensity lambda > 0.
    T : float
        Time horizon.
    n_paths : int
        Number of independent realizations.
    seed : int or None
        Random seed.

    Returns
    -------
    arrivals : list[np.ndarray]
        Each element is a 1-D array of arrival times in [0, T].
    """
    if rate <= 0:
        raise ValueError("rate must be positive")

    rng = np.random.default_rng(seed)
    arrivals: list[np.ndarray] = []

    for _ in range(n_paths):
        times = []
        t = 0.0
        while True:
            t += rng.exponential(1.0 / rate)
            if t > T:
                break
            times.append(t)
        arrivals.append(np.array(times))

    return arrivals


def compound_poisson_process(
    rate: float,
    T: float,
    jump_dist: str = "normal",
    jump_params: dict | None = None,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Simulate a compound Poisson process.

    Parameters
    ----------
    rate : float
        Poisson intensity.
    T : float
        Time horizon.
    jump_dist : {"normal", "exponential", "uniform"}
        Distribution family for jump sizes.
    jump_params : dict or None
        Parameters forwarded to the jump distribution (e.g. ``{"loc": 0, "scale": 1}``
        for normal).  Defaults are chosen per distribution.
    n_paths : int
        Number of independent realizations.
    seed : int or None
        Random seed.

    Returns
    -------
    paths : list[tuple[times, cumulative_jumps]]
        Each element is a pair of arrays: arrival times and the running
        cumulative sum of jump sizes.
    """
    rng = np.random.default_rng(seed)
    jump_params = jump_params or {}

    def _sample_jumps(n: int) -> np.ndarray:
        if jump_dist == "normal":
            return rng.normal(jump_params.get("loc", 0), jump_params.get("scale", 1), n)
        if jump_dist == "exponential":
            return rng.exponential(jump_params.get("scale", 1), n)
        if jump_dist == "uniform":
            return rng.uniform(jump_params.get("low", 0), jump_params.get("high", 1), n)
        raise ValueError(f"Unsupported jump_dist: {jump_dist!r}")

    arrival_lists = poisson_process(rate, T, n_paths, seed=None)

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for times in arrival_lists:
        if len(times) == 0:
            paths.append((times, np.array([])))
        else:
            jumps = _sample_jumps(len(times))
            paths.append((times, np.cumsum(jumps)))
    return paths
