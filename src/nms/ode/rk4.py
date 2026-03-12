"""
Classical fourth-order Runge-Kutta method (RK4).

Solves dy/dt = f(t, y) using the four-stage scheme:
    k1 = f(t_n,       y_n)
    k2 = f(t_n + h/2, y_n + h*k1/2)
    k3 = f(t_n + h/2, y_n + h*k2/2)
    k4 = f(t_n + h,   y_n + h*k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

The global truncation error is O(h^4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rk4_step(
    f: callable,
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Advance one RK4 step.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t : float
        Current time.
    y : np.ndarray
        Current state vector.
    h : float
        Step size.

    Returns
    -------
    np.ndarray
        State at t + h.
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + h / 2, y + h * k1 / 2), dtype=float)
    k3 = np.asarray(f(t + h / 2, y + h * k2 / 2), dtype=float)
    k4 = np.asarray(f(t + h, y + h * k3), dtype=float)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_solve(
    f: callable,
    t_span: tuple[float, float],
    y0: ArrayLike,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE system over *t_span* with fixed step size *h*.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    t_span : (t0, tf)
        Start and end times.
    y0 : array_like
        Initial state (scalar or vector).
    h : float
        Fixed step size.

    Returns
    -------
    t_values : np.ndarray, shape (N,)
    y_values : np.ndarray, shape (N, d)
        Solution trajectory.
    """
    t0, tf = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    t_values = [t0]
    y_values = [y0.copy()]

    t, y = t0, y0.copy()
    while t < tf - 1e-12 * h:
        h_actual = min(h, tf - t)
        y = rk4_step(f, t, y, h_actual)
        t += h_actual
        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)
