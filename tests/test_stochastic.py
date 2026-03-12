"""Tests for the stochastic processes module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nms.stochastic import (
    brownian_motion,
    geometric_brownian_motion,
    poisson_process,
    compound_poisson_process,
    MarkovChain,
)


class TestBrownianMotion:
    def test_shape(self):
        t, W = brownian_motion(1.0, 100, n_paths=5, seed=0)
        assert t.shape == (101,)
        assert W.shape == (5, 101)

    def test_starts_at_zero(self):
        _, W = brownian_motion(1.0, 50, n_paths=3, seed=0)
        assert np.all(W[:, 0] == 0.0)

    def test_variance_scaling(self):
        T = 2.0
        _, W = brownian_motion(T, 10_000, n_paths=5_000, seed=42)
        terminal_var = np.var(W[:, -1])
        assert_allclose(terminal_var, T, atol=0.15)

    def test_multidimensional(self):
        _, W = brownian_motion(1.0, 50, n_paths=2, dim=3, seed=0)
        assert W.shape == (2, 51, 3)


class TestGeometricBrownianMotion:
    def test_positive(self):
        _, S = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, n_paths=100, seed=0)
        assert np.all(S > 0)

    def test_shape(self):
        _, S = geometric_brownian_motion(100, 0.05, 0.2, 1.0, 252, n_paths=10, seed=0)
        assert S.shape == (10, 253)

    def test_expected_terminal(self):
        """E[S(T)] = S0 * exp(mu * T) under real measure."""
        n_paths = 50_000
        _, S = geometric_brownian_motion(100, 0.08, 0.3, 1.0, 1, n_paths=n_paths, seed=7)
        mean_terminal = np.mean(S[:, -1])
        expected = 100 * np.exp(0.08)
        assert_allclose(mean_terminal, expected, rtol=0.02)


class TestPoissonProcess:
    def test_single_path(self):
        arrivals = poisson_process(5.0, 10.0, n_paths=1, seed=0)
        assert len(arrivals) == 1
        assert np.all(arrivals[0] >= 0)
        assert np.all(arrivals[0] <= 10.0)

    def test_sorted_arrivals(self):
        arrivals = poisson_process(10.0, 5.0, n_paths=3, seed=42)
        for path in arrivals:
            assert np.all(np.diff(path) > 0)

    def test_mean_count(self):
        rate, T = 3.0, 10.0
        arrivals = poisson_process(rate, T, n_paths=10_000, seed=0)
        counts = [len(a) for a in arrivals]
        assert_allclose(np.mean(counts), rate * T, atol=1.0)

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            poisson_process(-1.0, 1.0)


class TestCompoundPoisson:
    def test_returns_pairs(self):
        paths = compound_poisson_process(2.0, 5.0, n_paths=3, seed=0)
        assert len(paths) == 3
        for times, cumulative in paths:
            assert len(times) == len(cumulative)


class TestMarkovChain:
    @pytest.fixture()
    def two_state_chain(self):
        P = [[0.7, 0.3],
             [0.4, 0.6]]
        return MarkovChain(P, state_labels=["A", "B"])

    def test_simulate_length(self, two_state_chain):
        states = two_state_chain.simulate(100, seed=0)
        assert len(states) == 101

    def test_simulate_valid_states(self, two_state_chain):
        states = two_state_chain.simulate(200, seed=0)
        assert np.all((states >= 0) & (states < 2))

    def test_stationary_distribution(self, two_state_chain):
        pi = two_state_chain.stationary_distribution()
        # pi should satisfy pi @ P = pi
        assert_allclose(pi @ two_state_chain.P, pi, atol=1e-10)
        assert_allclose(pi.sum(), 1.0)

    def test_n_step_matrix(self, two_state_chain):
        P2 = two_state_chain.n_step_matrix(2)
        expected = two_state_chain.P @ two_state_chain.P
        assert_allclose(P2, expected)

    def test_invalid_matrix(self):
        with pytest.raises(ValueError):
            MarkovChain([[0.5, 0.3], [0.4, 0.6]])

    def test_absorbing(self):
        P = [[1.0, 0.0], [0.5, 0.5]]
        mc = MarkovChain(P)
        assert mc.is_absorbing()

    def test_hitting_time(self, two_state_chain):
        h = two_state_chain.expected_hitting_time(0)
        assert h[0] == 0.0
        assert h[1] > 0.0
