"""
Newton's method for optimization and root finding.

**Root finding** (1-D and n-D):
    x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)

**Optimization** (unconstrained minimization):
    x_{k+1} = x_k - H(x_k)^{-1} * grad_f(x_k)

where H is the Hessian matrix.  When the Hessian is unavailable, a
finite-difference approximation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class NewtonResult:
    """Result container for Newton iterations."""

    x: np.ndarray
    f_val: float
    n_iter: int
    converged: bool
    trajectory: list[np.ndarray] = field(default_factory=list)


def _finite_difference_hessian(
    grad_f: callable,
    x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Approximate the Hessian via central finite differences of the gradient."""
    n = len(x)
    H = np.empty((n, n))
    for i in range(n):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += eps
        x_bwd[i] -= eps
        H[:, i] = (np.asarray(grad_f(x_fwd)) - np.asarray(grad_f(x_bwd))) / (2 * eps)
    return 0.5 * (H + H.T)


def newton_optimize(
    f: callable,
    grad_f: callable,
    x0: ArrayLike,
    *,
    hess_f: callable | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    track: bool = False,
) -> NewtonResult:
    """Minimize f(x) using Newton's method.

    Parameters
    ----------
    f : callable
        Objective function.
    grad_f : callable
        Gradient of f.
    x0 : array_like
        Starting point.
    hess_f : callable or None
        Hessian of f.  If None, the Hessian is estimated via finite differences
        of *grad_f*.
    tol : float
        Convergence tolerance on ||grad||.
    max_iter : int
        Maximum iterations.
    track : bool
        Record the full trajectory.
    """
    x = np.asarray(x0, dtype=float).copy()
    trajectory: list[np.ndarray] = []

    for i in range(max_iter):
        g = np.asarray(grad_f(x), dtype=float)

        if track:
            trajectory.append(x.copy())

        if np.linalg.norm(g) < tol:
            return NewtonResult(x, float(f(x)), i, True, trajectory)

        if hess_f is not None:
            H = np.asarray(hess_f(x), dtype=float)
        else:
            H = _finite_difference_hessian(grad_f, x)

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = g  # fallback to gradient step if Hessian is singular

        x -= delta

    return NewtonResult(x, float(f(x)), max_iter, False, trajectory)


def newton_root(
    F: callable,
    x0: ArrayLike,
    *,
    jac: callable | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
    track: bool = False,
) -> NewtonResult:
    """Find a root of F(x) = 0 using Newton-Raphson.

    Parameters
    ----------
    F : callable
        Vector function F(x) -> array of same shape as x.
    x0 : array_like
        Starting point.
    jac : callable or None
        Jacobian J(x).  If None, forward finite differences are used.
    tol : float
        Convergence tolerance on ||F(x)||.
    max_iter : int
        Maximum iterations.
    track : bool
        Record the full trajectory.
    """
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    trajectory: list[np.ndarray] = []

    def _fd_jacobian(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        f0 = np.asarray(F(x), dtype=float)
        n = len(x)
        J = np.empty((len(f0), n))
        for i in range(n):
            x_eps = x.copy()
            x_eps[i] += eps
            J[:, i] = (np.asarray(F(x_eps)) - f0) / eps
        return J

    for i in range(max_iter):
        Fx = np.asarray(F(x), dtype=float)

        if track:
            trajectory.append(x.copy())

        if np.linalg.norm(Fx) < tol:
            return NewtonResult(x, float(np.linalg.norm(Fx)), i, True, trajectory)

        J = np.asarray(jac(x), dtype=float) if jac is not None else _fd_jacobian(x)

        try:
            delta = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            delta = Fx

        x -= delta

    Fx = np.asarray(F(x), dtype=float)
    return NewtonResult(x, float(np.linalg.norm(Fx)), max_iter, False, trajectory)
