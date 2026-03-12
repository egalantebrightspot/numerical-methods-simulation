"""Tests for the optimization module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nms.optimization import (
    gradient_descent,
    gradient_descent_momentum,
    newton_optimize,
    newton_root,
)


# Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2,  minimum at (1,1)
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def rosenbrock_grad(x):
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dfdy = 200 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy])


# Simple quadratic: f(x) = x^T x / 2
def quadratic(x):
    return 0.5 * np.dot(x, x)

def quadratic_grad(x):
    return x.copy()

def quadratic_hess(x):
    return np.eye(len(x))


class TestGradientDescent:
    def test_quadratic(self):
        result = gradient_descent(quadratic, quadratic_grad, [5.0, -3.0], lr=0.1)
        assert result.converged
        assert_allclose(result.x, [0, 0], atol=1e-6)

    def test_with_line_search(self):
        result = gradient_descent(
            quadratic, quadratic_grad, [10.0, -7.0], line_search=True,
        )
        assert result.converged
        assert_allclose(result.x, [0, 0], atol=1e-6)

    def test_tracking(self):
        result = gradient_descent(
            quadratic, quadratic_grad, [1.0, 1.0], lr=0.1, track=True, max_iter=50,
        )
        assert len(result.trajectory) > 0
        assert len(result.f_history) == len(result.trajectory)


class TestGradientDescentMomentum:
    def test_quadratic(self):
        result = gradient_descent_momentum(
            quadratic, quadratic_grad, [5.0, -3.0], lr=0.05, beta=0.9,
        )
        assert result.converged
        assert_allclose(result.x, [0, 0], atol=1e-5)


class TestNewtonOptimize:
    def test_quadratic_exact_hessian(self):
        result = newton_optimize(
            quadratic, quadratic_grad, [10.0, -7.0], hess_f=quadratic_hess,
        )
        assert result.converged
        assert result.n_iter == 1  # Newton converges in 1 step for quadratics
        assert_allclose(result.x, [0, 0], atol=1e-10)

    def test_quadratic_fd_hessian(self):
        result = newton_optimize(quadratic, quadratic_grad, [10.0, -7.0])
        assert result.converged
        assert_allclose(result.x, [0, 0], atol=1e-6)

    def test_rosenbrock(self):
        result = newton_optimize(
            rosenbrock, rosenbrock_grad, [0.0, 0.0], max_iter=500,
        )
        assert result.converged
        assert_allclose(result.x, [1.0, 1.0], atol=1e-4)


class TestNewtonRoot:
    def test_scalar_root(self):
        result = newton_root(lambda x: x**2 - 4, [3.0])
        assert result.converged
        assert_allclose(np.abs(result.x[0]), 2.0, atol=1e-10)

    def test_system(self):
        """Solve x^2 + y^2 = 1, x - y = 0  =>  (1/sqrt(2), 1/sqrt(2))."""
        def F(xy):
            return np.array([xy[0] ** 2 + xy[1] ** 2 - 1, xy[0] - xy[1]])

        result = newton_root(F, [0.5, 0.5])
        assert result.converged
        expected = 1.0 / np.sqrt(2)
        assert_allclose(result.x, [expected, expected], atol=1e-10)

    def test_nonconvergence(self):
        result = newton_root(lambda x: np.array([x[0] ** 2 + 1]), [0.0], max_iter=10)
        assert not result.converged
