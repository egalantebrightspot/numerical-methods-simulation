"""
Brownian motion and geometric Brownian motion generators.

Standard Brownian motion (Wiener process) satisfies:
    W(0) = 0,  W(t) - W(s) ~ N(0, t - s),  independent increments.

Geometric Brownian motion (GBM) satisfies the SDE:
    dS = mu*S*dt + sigma*S*dW
with closed-form solution:
    S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
"""

from __future__ import annotations

import numpy as np


def brownian_motion(
    T: float,
    n_steps: int,
    n_paths: int = 1,
    *,
    dim: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate standard Brownian motion paths.

    Parameters
    ----------
    T : float
        Terminal time.
    n_steps : int
        Number of discrete time steps.
    n_paths : int
        Number of independent paths.
    dim : int
        Spatial dimension of each path.
    seed : int or None
        Random seed.

    Returns
    -------
    t : np.ndarray, shape (n_steps + 1,)
    W : np.ndarray, shape (n_paths, n_steps + 1) if dim == 1,
        or (n_paths, n_steps + 1, dim) otherwise.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    if dim == 1:
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
        W = np.zeros((n_paths, n_steps + 1))
        np.cumsum(dW, axis=1, out=W[:, 1:])
    else:
        dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps, dim))
        W = np.zeros((n_paths, n_steps + 1, dim))
        np.cumsum(dW, axis=1, out=W[:, 1:, :])

    return t, W


def geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate geometric Brownian motion paths.

    Parameters
    ----------
    s0 : float
        Initial value S(0).
    mu : float
        Drift coefficient.
    sigma : float
        Volatility coefficient.
    T : float
        Terminal time.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of independent paths.
    seed : int or None
        Random seed.

    Returns
    -------
    t : np.ndarray, shape (n_steps + 1,)
    S : np.ndarray, shape (n_paths, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(s0)
    np.cumsum(log_increments, axis=1, out=log_S[:, 1:])
    log_S[:, 1:] += np.log(s0)

    return t, np.exp(log_S)
