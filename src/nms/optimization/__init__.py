"""Numerical optimization: gradient descent, Newton's method, and line search."""

from .gradient_descent import gradient_descent, gradient_descent_momentum
from .newton import newton_optimize, newton_root

__all__ = [
    "gradient_descent",
    "gradient_descent_momentum",
    "newton_optimize",
    "newton_root",
]
