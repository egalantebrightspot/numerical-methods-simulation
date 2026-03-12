"""Tests for the Monte Carlo module."""

import numpy as np
import pytest

from nms.monte_carlo import random_walk_1d, random_walk_2d, european_option_mc, asian_option_mc


class TestRandomWalk1D:
    def test_shape(self):
        paths = random_walk_1d(100, n_walks=5, seed=0)
        assert paths.shape == (5, 101)

    def test_starts_at_origin(self):
        paths = random_walk_1d(50, n_walks=3, seed=0)
        assert np.all(paths[:, 0] == 0.0)

    def test_custom_start(self):
        paths = random_walk_1d(10, start=5.0, seed=0)
        assert paths[0, 0] == 5.0

    def test_step_size(self):
        paths = random_walk_1d(100, step_size=2.0, seed=0)
        diffs = np.abs(np.diff(paths, axis=1))
        assert np.allclose(diffs, 2.0)

    def test_reproducibility(self):
        a = random_walk_1d(20, seed=42)
        b = random_walk_1d(20, seed=42)
        assert np.array_equal(a, b)


class TestRandomWalk2D:
    def test_shape(self):
        paths = random_walk_2d(50, n_walks=3, seed=0)
        assert paths.shape == (3, 51, 2)

    def test_starts_at_origin(self):
        paths = random_walk_2d(10, seed=0)
        assert np.allclose(paths[:, 0, :], 0.0)

    def test_fixed_step_length(self):
        paths = random_walk_2d(100, step_size=1.0, seed=0)
        dx = np.diff(paths[0, :, 0])
        dy = np.diff(paths[0, :, 1])
        step_lengths = np.sqrt(dx**2 + dy**2)
        assert np.allclose(step_lengths, 1.0, atol=1e-12)


class TestEuropeanOption:
    def test_call_put_parity(self):
        """Put-call parity: C - P = S0 - K*exp(-rT)."""
        params = dict(s0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=500_000, seed=42)
        call = european_option_mc(**params, option_type="call")
        put = european_option_mc(**params, option_type="put")
        parity = call.price - put.price
        expected = params["s0"] - params["K"] * np.exp(-params["r"] * params["T"])
        assert abs(parity - expected) < 0.5

    def test_deep_itm_call(self):
        result = european_option_mc(
            s0=200, K=100, r=0.05, sigma=0.1, T=0.25, n_paths=100_000, seed=0,
        )
        intrinsic = 200 - 100 * np.exp(-0.05 * 0.25)
        assert result.price > intrinsic * 0.95

    def test_invalid_option_type(self):
        with pytest.raises(ValueError):
            european_option_mc(100, 100, 0.05, 0.2, 1.0, option_type="straddle")


class TestAsianOption:
    def test_asian_less_than_european(self):
        """An Asian call should cost less than the corresponding European call."""
        params = dict(s0=100, K=100, r=0.05, sigma=0.3, T=1.0, n_paths=200_000, seed=7)
        euro = european_option_mc(**params)
        asian = asian_option_mc(**params, n_steps=252)
        assert asian.price < euro.price + euro.std_error * 3

    def test_confidence_interval(self):
        result = asian_option_mc(
            s0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=50_000, seed=0,
        )
        assert result.ci_lower < result.price < result.ci_upper
