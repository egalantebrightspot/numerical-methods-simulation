"""
Stability analysis for ODE solvers.

The stability function R(z) of a one-step method applied to the test
equation y' = lambda*y determines the amplification factor per step.
The method is absolutely stable for |R(z)| <= 1 where z = h*lambda.

Euler:  R(z) = 1 + z
RK4:    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
"""

from __future__ import annotations

import numpy as np


def stability_function_euler(z: np.ndarray | complex) -> np.ndarray:
    """Stability function for the forward Euler method: R(z) = 1 + z."""
    return 1 + z


def stability_function_rk4(z: np.ndarray | complex) -> np.ndarray:
    """Stability function for the classical RK4 method."""
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def stability_region(
    R: callable,
    *,
    x_range: tuple[float, float] = (-5, 2),
    y_range: tuple[float, float] = (-3.5, 3.5),
    n_points: int = 500,
) -> dict:
    """Compute the boundary of the absolute stability region |R(z)| <= 1.

    Parameters
    ----------
    R : callable
        Stability function mapping complex *z* to complex *R(z)*.
    x_range : (float, float)
        Real-axis range for the grid.
    y_range : (float, float)
        Imaginary-axis range for the grid.
    n_points : int
        Grid resolution in each direction.

    Returns
    -------
    dict with keys:
        X, Y : np.ndarray, shape (n_points, n_points)
            Real and imaginary parts of the grid.
        magnitude : np.ndarray, shape (n_points, n_points)
            |R(z)| on the grid.
        is_stable : np.ndarray (bool)
            True where |R(z)| <= 1.
    """
    x = np.linspace(*x_range, n_points)
    y = np.linspace(*y_range, n_points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    magnitude = np.abs(R(Z))

    return {
        "X": X,
        "Y": Y,
        "magnitude": magnitude,
        "is_stable": magnitude <= 1.0,
    }


def max_stable_step(
    R: callable,
    lam: complex,
    *,
    h_range: tuple[float, float] = (1e-6, 10.0),
    n_samples: int = 10_000,
) -> float:
    """Estimate the largest step size *h* for which the method is stable.

    For a given eigenvalue *lam* of the ODE system, the method is stable
    when |R(h * lam)| <= 1.

    Parameters
    ----------
    R : callable
        Stability function.
    lam : complex
        Eigenvalue of the linearized system (typically negative real for
        dissipative problems, purely imaginary for oscillatory ones).
    h_range : (float, float)
        Search interval for h.
    n_samples : int
        Number of candidate h values to test.

    Returns
    -------
    float
        Approximate maximum stable step size.
    """
    h_vals = np.linspace(*h_range, n_samples)
    magnitudes = np.abs(R(h_vals * lam))
    stable = h_vals[magnitudes <= 1.0]
    return float(stable[-1]) if len(stable) > 0 else 0.0
