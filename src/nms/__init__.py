"""
Numerical Methods & Simulation Engine.

A modular library for solving differential equations, running Monte Carlo
simulations, modeling stochastic processes, and performing numerical
optimization.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nms")
except PackageNotFoundError:
    __version__ = "0.1.0"
