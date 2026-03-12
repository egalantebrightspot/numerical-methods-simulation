"""
Adaptive step-size ODE solver using the Runge-Kutta-Fehlberg (RKF45) pair.

An embedded 4(5) pair provides a fourth-order solution and a fifth-order
error estimate, allowing the integrator to adapt *h* so the local truncation
error stays within a user-specified tolerance.

Butcher tableau coefficients follow the classical Fehlberg embedding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# --- Fehlberg coefficients ---------------------------------------------------
_A = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]

_B = [
    [],
    [1 / 4],
    [3 / 32, 9 / 32],
    [1932 / 2197, -7200 / 2197, 7296 / 2197],
    [439 / 216, -8, 3680 / 513, -845 / 4104],
    [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
]

# 4th-order weights
_C4 = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
# 5th-order weights
_C5 = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])


def rkf45_step(
    f: callable,
    t: float,
    y: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one RKF45 step, returning the 4th-order result and error estimate.

    Returns
    -------
    y4 : np.ndarray
        Fourth-order solution at t + h.
    y5 : np.ndarray
        Fifth-order solution at t + h (used only for error estimation).
    error : np.ndarray
        Elementwise difference |y5 - y4|.
    """
    y = np.asarray(y, dtype=float)
    k = [np.asarray(f(t, y), dtype=float)]

    for i in range(1, 6):
        dy = sum(b * k[j] for j, b in enumerate(_B[i]))
        k.append(np.asarray(f(t + _A[i] * h, y + h * dy), dtype=float))

    k = np.array(k)
    y4 = y + h * _C4 @ k
    y5 = y + h * _C5 @ k
    return y4, y5, np.abs(y5 - y4)


def adaptive_solve(
    f: callable,
    t_span: tuple[float, float],
    y0: ArrayLike,
    *,
    tol: float = 1e-6,
    h0: float | None = None,
    h_min: float = 1e-12,
    h_max: float | None = None,
    max_steps: int = 100_000,
    safety: float = 0.84,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE system with adaptive RKF45 stepping.

    Uses local extrapolation: the 5th-order solution is propagated while
    the 4th/5th-order difference drives step-size control.  This makes the
    actual global error substantially smaller than *tol*.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : (t0, tf)
        Integration interval.
    y0 : array_like
        Initial state.
    tol : float
        Local error tolerance (applied per-component).
    h0 : float or None
        Initial step size.  Defaults to (tf - t0) / 100.
    h_min, h_max : float
        Hard bounds on step size.
    max_steps : int
        Guard against infinite loops.
    safety : float
        Safety factor (< 1) applied when proposing a new *h*.

    Returns
    -------
    t_values : np.ndarray
    y_values : np.ndarray
    """
    t0, tf = t_span
    y = np.atleast_1d(np.asarray(y0, dtype=float))
    h = h0 if h0 is not None else (tf - t0) / 100
    if h_max is None:
        h_max = (tf - t0) / 4

    t_values = [t0]
    y_values = [y.copy()]

    t = t0
    for _ in range(max_steps):
        if t >= tf - 1e-12 * abs(h):
            break

        h = min(h, tf - t)
        h = np.clip(h, h_min, h_max)

        y4, y5, err = rkf45_step(f, t, y, h)
        err_norm = np.max(err) if err.size > 0 else 0.0

        if err_norm <= tol or h <= h_min:
            t += h
            y = y5  # local extrapolation: advance with the higher-order solution
            t_values.append(t)
            y_values.append(y.copy())

        if err_norm > 0:
            h_new = safety * h * (tol / err_norm) ** 0.2
        else:
            h_new = h * 2
        h = np.clip(h_new, h_min, h_max)

    return np.array(t_values), np.array(y_values)
