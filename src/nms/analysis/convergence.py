"""
Convergence analysis utilities.

Tools for measuring how quickly a numerical solution approaches the true
solution as the discretization parameter (step size, grid spacing, etc.)
is refined.

The empirical convergence rate *p* is estimated from:
    error(h) ~ C * h^p   =>   p ~ log(e1/e2) / log(h1/h2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def convergence_rate(
    errors: ArrayLike,
    step_sizes: ArrayLike,
) -> np.ndarray:
    """Estimate the empirical convergence rate from successive refinements.

    Parameters
    ----------
    errors : array_like, shape (n,)
        Error norms at each refinement level.
    step_sizes : array_like, shape (n,)
        Corresponding step sizes.

    Returns
    -------
    rates : np.ndarray, shape (n - 1,)
        Estimated order between each consecutive pair.
    """
    errors = np.asarray(errors, dtype=float)
    step_sizes = np.asarray(step_sizes, dtype=float)

    if len(errors) != len(step_sizes):
        raise ValueError("errors and step_sizes must have the same length")
    if len(errors) < 2:
        raise ValueError("Need at least two data points")

    rates = np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])
    return rates


def refinement_study(
    solver: callable,
    reference: callable,
    step_sizes: ArrayLike,
    *,
    norm: callable = None,
) -> dict:
    """Run a grid-refinement study for a given solver.

    Parameters
    ----------
    solver : callable
        ``solver(h) -> (t_values, y_values)`` returning the numerical solution
        for step size *h*.
    reference : callable
        ``reference(t_values) -> y_exact`` returning the exact solution
        evaluated at the given time points.
    step_sizes : array_like
        Sequence of step sizes to test (from coarse to fine).
    norm : callable or None
        Error norm function ``norm(error_array) -> float``.  Defaults to the
        max absolute error at the final time point.

    Returns
    -------
    dict with keys:
        step_sizes, errors, rates, log_step_sizes, log_errors
    """
    if norm is None:
        norm = lambda e: float(np.max(np.abs(e)))

    step_sizes = np.asarray(step_sizes, dtype=float)
    errors = np.empty(len(step_sizes))

    for i, h in enumerate(step_sizes):
        t_vals, y_vals = solver(h)
        y_exact = reference(t_vals)
        errors[i] = norm(y_vals[-1] - y_exact[-1])

    rates = convergence_rate(errors, step_sizes)

    return {
        "step_sizes": step_sizes,
        "errors": errors,
        "rates": rates,
        "log_step_sizes": np.log10(step_sizes),
        "log_errors": np.log10(errors),
    }


def richardson_extrapolation(
    f_h: float,
    f_h2: float,
    p: int,
) -> float:
    """Apply Richardson extrapolation to improve an O(h^p) approximation.

    Given two approximations computed at step sizes h and h/2:
        f_exact ~ (2^p * f_{h/2} - f_h) / (2^p - 1)

    Parameters
    ----------
    f_h : float
        Approximation at step size h.
    f_h2 : float
        Approximation at step size h/2.
    p : int
        Leading-order exponent of the truncation error.

    Returns
    -------
    float
        Extrapolated estimate.
    """
    factor = 2**p
    return (factor * f_h2 - f_h) / (factor - 1)
