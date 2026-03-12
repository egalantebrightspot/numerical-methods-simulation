"""Tests for the ODE solver module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nms.ode import euler_step, euler_solve, rk4_step, rk4_solve, adaptive_solve


# dy/dt = -y  =>  y(t) = y0 * exp(-t)
def exp_decay(t, y):
    return -y


class TestEuler:
    def test_single_step(self):
        y_next = euler_step(exp_decay, 0.0, np.array([1.0]), 0.1)
        expected = np.array([1.0 + 0.1 * (-1.0)])
        assert_allclose(y_next, expected)

    def test_solve_exponential_decay(self):
        t, y = euler_solve(exp_decay, (0, 1), [1.0], h=0.001)
        exact = np.exp(-t)
        assert_allclose(y[:, 0], exact, atol=2e-3)

    def test_scalar_initial_condition(self):
        t, y = euler_solve(exp_decay, (0, 0.5), 1.0, h=0.01)
        assert y.ndim == 2
        assert y.shape[1] == 1


class TestRK4:
    def test_single_step_accuracy(self):
        y0 = np.array([1.0])
        h = 0.1
        y_next = rk4_step(exp_decay, 0.0, y0, h)
        exact = np.exp(-h)
        assert_allclose(y_next[0], exact, atol=1e-8)

    def test_solve_exponential_decay(self):
        t, y = rk4_solve(exp_decay, (0, 2), [1.0], h=0.1)
        exact = np.exp(-t)
        assert_allclose(y[:, 0], exact, atol=1e-6)

    def test_system(self):
        """Solve the harmonic oscillator: y'' + y = 0 as a 2-D system."""
        def harmonic(t, y):
            return np.array([y[1], -y[0]])

        t, y = rk4_solve(harmonic, (0, 2 * np.pi), [1.0, 0.0], h=0.01)
        assert_allclose(y[-1, 0], 1.0, atol=1e-5)
        assert_allclose(y[-1, 1], 0.0, atol=1e-5)


class TestAdaptive:
    def test_exponential_decay(self):
        t, y = adaptive_solve(exp_decay, (0, 3), [1.0], tol=1e-8)
        exact = np.exp(-t)
        assert_allclose(y[:, 0], exact, atol=1e-6)

    def test_respects_tolerance(self):
        t1, y1 = adaptive_solve(exp_decay, (0, 1), [1.0], tol=1e-4)
        t2, y2 = adaptive_solve(exp_decay, (0, 1), [1.0], tol=1e-8)
        # tighter tolerance should produce more steps
        assert len(t2) >= len(t1)

    def test_stiff_equation(self):
        """Mildly stiff problem: dy/dt = -50*y."""
        def stiff(t, y):
            return -50 * y

        t, y = adaptive_solve(stiff, (0, 0.5), [1.0], tol=1e-6)
        exact = np.exp(-50 * t)
        assert_allclose(y[:, 0], exact, atol=1e-4)
