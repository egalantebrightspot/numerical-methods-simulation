"""
Convergence and stability analysis for Euler and RK4 on a stiff ODE.

Test equation: y' = λy,  y(0) = 1,  exact solution y(t) = exp(λt).

Using λ = -25 so that the stability boundaries produce the desired behavior:
    Euler stability limit:  h_max = 2/|λ| = 0.08
    RK4   stability limit:  h_max ≈ 2.785/|λ| ≈ 0.111

Tested step sizes h = [0.1, 0.05, 0.01]:
    h = 0.1  → Euler UNSTABLE (h > 0.08), RK4 STABLE (h < 0.111)
    h = 0.05 → Both stable
    h = 0.01 → Both stable

Convergence tests use a shorter interval [0, 0.1] (where exp(-2.5) ≈ 0.08 is
still appreciable) so that truncation error dominates over the near-zero exact
solution.
"""

import numpy as np
import pytest

from nms.ode import euler_solve, rk4_solve
from nms.analysis.convergence import convergence_rate, refinement_study
from nms.analysis.stability import (
    stability_function_euler,
    stability_function_rk4,
    max_stable_step,
)

LAMBDA = -25
T_SPAN = (0.0, 1.0)
T_SPAN_SHORT = (0.0, 0.1)
Y0 = [1.0]
STEP_SIZES = [0.1, 0.05, 0.01]


def stiff_rhs(t, y):
    return LAMBDA * y


def analytical(t):
    return np.exp(LAMBDA * np.asarray(t))


# ---------------------------------------------------------------------------
# Euler stability
# ---------------------------------------------------------------------------

class TestEulerStability:
    def test_unstable_at_h_0_1(self):
        """h=0.1 exceeds Euler stability limit; amplification factor > 1."""
        z = LAMBDA * 0.1
        assert abs(stability_function_euler(z)) > 1.0

        t, y = euler_solve(stiff_rhs, T_SPAN, Y0, h=0.1)
        exact = analytical(t)
        assert abs(y[-1, 0] - exact[-1]) > 1.0

    def test_stable_at_h_0_05(self):
        z = LAMBDA * 0.05
        assert abs(stability_function_euler(z)) < 1.0

        t, y = euler_solve(stiff_rhs, T_SPAN, Y0, h=0.05)
        exact = analytical(t)
        assert abs(y[-1, 0] - exact[-1]) < 0.1

    def test_stable_at_h_0_01(self):
        z = LAMBDA * 0.01
        assert abs(stability_function_euler(z)) < 1.0

        t, y = euler_solve(stiff_rhs, T_SPAN, Y0, h=0.01)
        exact = analytical(t)
        assert abs(y[-1, 0] - exact[-1]) < 0.01


# ---------------------------------------------------------------------------
# RK4 stability
# ---------------------------------------------------------------------------

class TestRK4Stability:
    @pytest.mark.parametrize("h", STEP_SIZES)
    def test_stable_at_all_step_sizes(self, h):
        z = LAMBDA * h
        assert abs(stability_function_rk4(z)) <= 1.0

        t, y = rk4_solve(stiff_rhs, T_SPAN, Y0, h=h)
        exact = analytical(t)
        assert abs(y[-1, 0] - exact[-1]) < 1.0, (
            f"RK4 diverged at h={h}: error={abs(y[-1, 0] - exact[-1]):.2e}"
        )

    def test_rk4_much_more_accurate_than_euler(self):
        """At h=0.02 (both stable), RK4 error is orders of magnitude smaller."""
        h = 0.02
        t_e, y_e = euler_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)
        t_r, y_r = rk4_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)

        err_euler = abs(y_e[-1, 0] - analytical(t_e[-1]))
        err_rk4 = abs(y_r[-1, 0] - analytical(t_r[-1]))
        assert err_rk4 < err_euler / 100


# ---------------------------------------------------------------------------
# Stability boundaries
# ---------------------------------------------------------------------------

class TestStabilityBoundaries:
    def test_euler_stability_boundary(self):
        h_max = max_stable_step(stability_function_euler, LAMBDA)
        expected = 2.0 / abs(LAMBDA)
        assert abs(h_max - expected) / expected < 0.02

    def test_rk4_wider_than_euler(self):
        h_euler = max_stable_step(stability_function_euler, LAMBDA)
        h_rk4 = max_stable_step(stability_function_rk4, LAMBDA)
        assert h_rk4 > h_euler


# ---------------------------------------------------------------------------
# Convergence rates (step sizes within the stable region)
# ---------------------------------------------------------------------------

CONV_STEPS = [0.02, 0.01, 0.005, 0.0025]


class TestConvergenceRates:
    """Convergence tests use the short interval where the solution is
    appreciable so that truncation error — not the vanishing exact value —
    governs the measured error."""

    def test_euler_first_order(self):
        """Error slope ≈ 1 in log-log space."""
        errors = []
        for h in CONV_STEPS:
            t, y = euler_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)
            errors.append(abs(y[-1, 0] - analytical(t[-1])))

        rates = convergence_rate(errors, CONV_STEPS)
        for r in rates:
            assert 0.8 < r < 1.5, f"Euler rate {r:.2f} outside [0.8, 1.5]"

    def test_rk4_fourth_order(self):
        """Error slope ≈ 4 in log-log space."""
        errors = []
        for h in CONV_STEPS:
            t, y = rk4_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)
            errors.append(abs(y[-1, 0] - analytical(t[-1])))

        rates = convergence_rate(errors, CONV_STEPS)
        for r in rates:
            assert 3.5 < r < 5.0, f"RK4 rate {r:.2f} outside [3.5, 5.0]"

    def test_error_decreases_euler(self):
        errors = []
        for h in CONV_STEPS:
            t, y = euler_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)
            errors.append(abs(y[-1, 0] - analytical(t[-1])))

        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]

    def test_error_decreases_rk4(self):
        errors = []
        for h in CONV_STEPS:
            t, y = rk4_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)
            errors.append(abs(y[-1, 0] - analytical(t[-1])))

        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1]


# ---------------------------------------------------------------------------
# Refinement study (integration test for analysis utilities)
# ---------------------------------------------------------------------------

class TestRefinementStudy:
    def _reference(self, t):
        return analytical(t).reshape(-1, 1)

    def test_euler_refinement(self):
        def solver(h):
            return euler_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)

        result = refinement_study(solver, self._reference, CONV_STEPS)
        assert np.all(result["rates"] > 0.5)
        assert np.all(result["rates"] < 2.0)

    def test_rk4_refinement(self):
        def solver(h):
            return rk4_solve(stiff_rhs, T_SPAN_SHORT, Y0, h=h)

        result = refinement_study(solver, self._reference, CONV_STEPS)
        assert np.all(result["rates"] > 3.0)
