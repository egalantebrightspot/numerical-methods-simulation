"""
Monte Carlo option pricing.

European and Asian option valuation via simulated paths of geometric
Brownian motion (GBM).  Includes variance reduction through antithetic
variates and confidence-interval estimation.

Under risk-neutral pricing the discounted expected payoff gives the
fair option value:
    C = e^{-rT} * E[max(S_T - K, 0)]   (European call)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PricingResult:
    """Container for a Monte Carlo pricing estimate."""

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int


def _simulate_gbm_terminal(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool,
) -> np.ndarray:
    """Sample terminal values S(T) under risk-neutral GBM."""
    n = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(n)
    if antithetic:
        z = np.concatenate([z, -z])
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return s0 * np.exp(drift + diffusion)


def _simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    antithetic: bool,
) -> np.ndarray:
    """Simulate full GBM paths, shape (n_paths, n_steps + 1)."""
    dt = T / n_steps
    n = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((n, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)

    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.zeros((z.shape[0], n_steps + 1))
    log_paths[:, 0] = np.log(s0)
    np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
    log_paths[:, 1:] += np.log(s0)
    return np.exp(log_paths)


def _build_result(payoffs: np.ndarray, r: float, T: float) -> PricingResult:
    discounted = np.exp(-r * T) * payoffs
    mean = float(np.mean(discounted))
    se = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
    return PricingResult(
        price=mean,
        std_error=se,
        ci_lower=mean - 1.96 * se,
        ci_upper=mean + 1.96 * se,
        n_paths=len(discounted),
    )


def european_option_mc(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    *,
    option_type: str = "call",
    antithetic: bool = True,
    seed: int | None = None,
) -> PricingResult:
    """Price a European option via Monte Carlo.

    Parameters
    ----------
    s0 : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized, continuous compounding).
    sigma : float
        Volatility (annualized).
    T : float
        Time to expiry in years.
    n_paths : int
        Number of simulated paths.
    option_type : {"call", "put"}
    antithetic : bool
        Use antithetic variates for variance reduction.
    seed : int or None
        Random seed.
    """
    rng = np.random.default_rng(seed)
    st = _simulate_gbm_terminal(s0, r, sigma, T, n_paths, rng, antithetic)

    if option_type == "call":
        payoffs = np.maximum(st - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - st, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return _build_result(payoffs, r, T)


def asian_option_mc(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 100_000,
    *,
    option_type: str = "call",
    antithetic: bool = True,
    seed: int | None = None,
) -> PricingResult:
    """Price an arithmetic Asian option via Monte Carlo.

    The payoff is based on the arithmetic average of the price path:
        max(A - K, 0) for a call, where A = (1/N) * sum(S_i).

    Parameters
    ----------
    s0, K, r, sigma, T, n_paths, option_type, antithetic, seed
        Same as :func:`european_option_mc`.
    n_steps : int
        Number of time steps in each path (e.g. 252 trading days).
    """
    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, rng, antithetic)
    avg_price = paths[:, 1:].mean(axis=1)

    if option_type == "call":
        payoffs = np.maximum(avg_price - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - avg_price, 0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return _build_result(payoffs, r, T)
