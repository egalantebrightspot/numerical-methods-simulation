"""
Forward Euler method for first-order ODE initial-value problems.

Solves dy/dt = f(t, y) with y(t0) = y0 using the recurrence
    y_{n+1} = y_n + h * f(t_n, y_n)

This is a first-order method: the global truncation error is O(h).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def euler_step(
    f: callable,
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """Advance one Euler step.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) returning an array of the same shape as *y*.
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
    return y + h * np.asarray(f(t, y), dtype=float)


def euler_solve(
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
        Solution trajectory.  *d* is the dimension of the state vector.
    """
    t0, tf = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))

    t_values = [t0]
    y_values = [y0.copy()]

    t, y = t0, y0.copy()
    while t < tf - 1e-12 * h:
        h_actual = min(h, tf - t)
        y = euler_step(f, t, y, h_actual)
        t += h_actual
        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)
