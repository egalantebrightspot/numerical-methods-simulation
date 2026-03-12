# Numerical Methods & Simulation Engine

A modular, mathematically grounded library for solving differential equations, running Monte Carlo simulations, modeling stochastic processes, and performing numerical optimization. This project highlights the quantitative foundations behind scientific computing, applied research, and decision systems.

The engine is designed to be transparent, extensible, and production‑ready, with a focus on numerical stability, convergence behavior, and reproducibility.

---

## Overview

This repository provides a unified framework for core numerical methods used in applied mathematics, quantitative finance, optimization, and simulation modeling. It includes:

- differential equation solvers  
- Monte Carlo simulation tools  
- stochastic process generators  
- numerical optimization algorithms  
- convergence and stability analysis utilities  

Each module is implemented with clarity and mathematical rigor, making the library suitable for research, education, and real‑world modeling workflows.

---

## Features

### Differential equation solvers  
Implementations of foundational ODE solvers, including Euler, Runge–Kutta (RK4), and adaptive step methods. These tools support modeling of dynamic systems, physical processes, and control problems.

### Monte Carlo simulation  
Tools for random walk generation, option‑style payoff simulation, variance reduction, and confidence interval estimation. Useful for risk modeling, forecasting, and probabilistic analysis.

### Stochastic processes  
Generators for Brownian motion, geometric Brownian motion, Poisson processes, and Markov chains. These processes form the backbone of quantitative finance, queueing theory, and probabilistic modeling.

### Numerical optimization  
Algorithms such as gradient descent, Newton’s method, and line search routines. These methods support convex and non‑convex optimization tasks across scientific and engineering domains.

### Convergence and stability analysis  
Utilities for evaluating numerical error, step size sensitivity, and solver stability. These tools help diagnose and compare algorithmic performance.

---

## Validation

Every module is tested against known analytical solutions, statistical properties, or theoretical bounds. Run the full suite with:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

### ODE solvers — accuracy and convergence

| Solver | Test problem | Metric | Result |
|--------|-------------|--------|--------|
| Euler | y' = −y, y(0) = 1 | max error at t = 1, h = 0.001 | < 2 × 10⁻³ |
| RK4 | y' = −y, y(0) = 1 | max error at t = 2, h = 0.1 | < 10⁻⁶ |
| RK4 | Harmonic oscillator y'' + y = 0 | period return error | < 10⁻⁵ |
| Adaptive (RKF45) | y' = −50y (mildly stiff) | error at t = 0.5 | < 10⁻⁴ |

### ODE solvers — stability and convergence order

Stiff test equation y' = −25y solved with step sizes h ∈ {0.1, 0.05, 0.01}:

| Criterion | Expected | Verified |
|-----------|----------|----------|
| Euler unstable at h = 0.1 | \|1 + hλ\| = 1.5 > 1 | divergent solution |
| Euler stable at h = 0.01 | \|1 + hλ\| = 0.75 < 1 | error < 0.01 |
| RK4 stable at all tested h | \|R(hλ)\| < 1 for h ≤ 0.1 | bounded error |
| Euler convergence rate | O(h) — slope ≈ 1 | measured 0.9–1.2 |
| RK4 convergence rate | O(h⁴) — slope ≈ 4 | measured 3.8–4.2 |

### Monte Carlo simulation

| Test | Reference | Tolerance |
|------|-----------|-----------|
| European put-call parity | C − P = S₀ − Ke⁻ʳᵀ | < 0.50 (500k paths) |
| Deep ITM call | intrinsic value floor | > 95% of intrinsic |
| Asian call < European call | Jensen's inequality | confirmed |
| Confidence interval coverage | 95% CI contains price | confirmed |

### Stochastic processes

| Process | Property | Method |
|---------|----------|--------|
| Brownian motion | Var[W(T)] = T | 5 000 paths, atol 0.15 |
| Geometric Brownian motion | E[S(T)] = S₀eᵘᵀ | 50 000 paths, rtol 2% |
| Poisson process | E[N(T)] = λT | 10 000 paths, atol 1.0 |
| Markov chain | πP = π (stationary dist.) | exact to 10⁻¹⁰ |

### Numerical optimization

| Algorithm | Test function | Convergence | Accuracy |
|-----------|--------------|-------------|----------|
| Gradient descent | ½\|x\|² | ✓ | atol 10⁻⁶ |
| Gradient descent + line search | ½\|x\|² | ✓ | atol 10⁻⁶ |
| Momentum SGD | ½\|x\|² | ✓ | atol 10⁻⁵ |
| Newton (exact Hessian) | ½\|x\|² | 1 iteration | atol 10⁻¹⁰ |
| Newton (FD Hessian) | Rosenbrock | ✓ | atol 10⁻⁴ |
| Newton root-finding | x² + y² = 1, x = y | ✓ | atol 10⁻¹⁰ |

### Test map

```
tests/
├── test_ode.py                    # Euler, RK4, adaptive solver accuracy
├── test_convergence_stability.py  # Stability limits, convergence order
├── test_monte_carlo.py            # Option pricing, random walks
├── test_stochastic.py             # Brownian, Poisson, Markov chains
└── test_optimization.py           # Gradient descent, Newton methods
```