"""
Discrete-time Markov chain simulator.

A Markov chain is defined by a transition matrix P where P[i, j] is the
probability of moving from state i to state j.  The chain satisfies the
Markov property: the future state depends only on the current state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


class MarkovChain:
    """Discrete-time, finite-state Markov chain.

    Parameters
    ----------
    transition_matrix : array_like, shape (n, n)
        Row-stochastic matrix (each row sums to 1).
    state_labels : list[str] or None
        Human-readable labels for the states.
    """

    def __init__(
        self,
        transition_matrix: ArrayLike,
        state_labels: list[str] | None = None,
    ) -> None:
        P = np.asarray(transition_matrix, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("transition_matrix must be square")
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Each row of the transition matrix must sum to 1")
        if np.any(P < 0):
            raise ValueError("Transition probabilities must be non-negative")

        self.P = P
        self.n_states = P.shape[0]
        self.state_labels = state_labels or [str(i) for i in range(self.n_states)]

    def simulate(
        self,
        n_steps: int,
        start: int = 0,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Run the chain for *n_steps* starting from state *start*.

        Returns
        -------
        states : np.ndarray, shape (n_steps + 1,)
            Sequence of visited state indices.
        """
        rng = np.random.default_rng(seed)
        states = np.empty(n_steps + 1, dtype=int)
        states[0] = start

        for i in range(n_steps):
            states[i + 1] = rng.choice(self.n_states, p=self.P[states[i]])

        return states

    def stationary_distribution(self) -> np.ndarray:
        """Compute the stationary distribution pi such that pi @ P = pi.

        Uses the left-eigenvector corresponding to eigenvalue 1.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()
        return pi

    def is_absorbing(self) -> bool:
        """Return True if the chain has at least one absorbing state."""
        return bool(np.any(np.diag(self.P) == 1.0))

    def n_step_matrix(self, n: int) -> np.ndarray:
        """Return P^n (the *n*-step transition matrix)."""
        return np.linalg.matrix_power(self.P, n)

    def expected_hitting_time(self, target: int) -> np.ndarray:
        """Mean first-passage time to *target* from every other state.

        Solves the linear system  h_i = 1 + sum_{j != target} P[i,j] * h_j
        for all i != target.

        Returns
        -------
        h : np.ndarray, shape (n_states,)
            h[target] == 0; h[i] is the expected number of steps to reach
            *target* from state *i*.
        """
        n = self.n_states
        mask = np.ones(n, dtype=bool)
        mask[target] = False
        idx = np.where(mask)[0]

        A = np.eye(len(idx)) - self.P[np.ix_(idx, idx)]
        b = np.ones(len(idx))
        h_sub = np.linalg.solve(A, b)

        h = np.zeros(n)
        h[idx] = h_sub
        return h
