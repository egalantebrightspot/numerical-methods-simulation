"""
Gradient descent optimizers with optional momentum and backtracking line search.

Minimizes a scalar objective f(x) given its gradient grad_f(x).

Vanilla update:       x_{k+1} = x_k - alpha * grad_f(x_k)
Momentum update:      v_{k+1} = beta * v_k + grad_f(x_k)
                      x_{k+1} = x_k - alpha * v_{k+1}

The Armijo backtracking line search adaptively selects the step size so that
a sufficient-decrease condition is satisfied at every iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class OptimizationResult:
    """Result container for optimization runs."""

    x: np.ndarray
    f_val: float
    n_iter: int
    converged: bool
    trajectory: list[np.ndarray] = field(default_factory=list)
    f_history: list[float] = field(default_factory=list)


def _backtracking_line_search(
    f: callable,
    x: np.ndarray,
    grad: np.ndarray,
    *,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 40,
) -> float:
    """Armijo backtracking line search.

    Returns the largest alpha = alpha0 * rho^k satisfying the sufficient-decrease
    (Armijo) condition:  f(x - alpha*grad) <= f(x) - c*alpha*||grad||^2.
    """
    alpha = alpha0
    f0 = f(x)
    grad_sq = float(np.dot(grad, grad))

    for _ in range(max_iter):
        if f(x - alpha * grad) <= f0 - c * alpha * grad_sq:
            return alpha
        alpha *= rho

    return alpha


def gradient_descent(
    f: callable,
    grad_f: callable,
    x0: ArrayLike,
    *,
    lr: float = 0.01,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    line_search: bool = False,
    track: bool = False,
) -> OptimizationResult:
    """Vanilla gradient descent.

    Parameters
    ----------
    f : callable
        Objective function f(x) -> float.
    grad_f : callable
        Gradient function grad_f(x) -> array.
    x0 : array_like
        Starting point.
    lr : float
        Learning rate (ignored when *line_search* is True).
    tol : float
        Stop when ||grad|| < tol.
    max_iter : int
        Maximum iterations.
    line_search : bool
        Use Armijo backtracking to select step size.
    track : bool
        If True, record trajectory and f_history.
    """
    x = np.asarray(x0, dtype=float).copy()
    trajectory: list[np.ndarray] = []
    f_history: list[float] = []

    for i in range(max_iter):
        g = np.asarray(grad_f(x), dtype=float)

        if track:
            trajectory.append(x.copy())
            f_history.append(float(f(x)))

        if np.linalg.norm(g) < tol:
            return OptimizationResult(x, float(f(x)), i, True, trajectory, f_history)

        alpha = _backtracking_line_search(f, x, g) if line_search else lr
        x -= alpha * g

    return OptimizationResult(x, float(f(x)), max_iter, False, trajectory, f_history)


def gradient_descent_momentum(
    f: callable,
    grad_f: callable,
    x0: ArrayLike,
    *,
    lr: float = 0.01,
    beta: float = 0.9,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    track: bool = False,
) -> OptimizationResult:
    """Gradient descent with classical momentum.

    Parameters
    ----------
    f, grad_f, x0, lr, tol, max_iter, track
        Same as :func:`gradient_descent`.
    beta : float
        Momentum coefficient in [0, 1).
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.zeros_like(x)
    trajectory: list[np.ndarray] = []
    f_history: list[float] = []

    for i in range(max_iter):
        g = np.asarray(grad_f(x), dtype=float)

        if track:
            trajectory.append(x.copy())
            f_history.append(float(f(x)))

        if np.linalg.norm(g) < tol:
            return OptimizationResult(x, float(f(x)), i, True, trajectory, f_history)

        v = beta * v + g
        x -= lr * v

    return OptimizationResult(x, float(f(x)), max_iter, False, trajectory, f_history)
