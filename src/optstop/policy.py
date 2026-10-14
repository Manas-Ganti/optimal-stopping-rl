"""The one interface every exercise rule implements, learned or classical.

A policy answers a single question: at exercise date k, at moneyness x = S/K, exercise now?
Both LSM and the learned agent implement it, so `boundary.py` and `evaluate/` treat them
identically. Convention: callers only ask about in-the-money states as candidates for
exercise; exercising a worthless option is never optimal and the evaluators forbid it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from optstop.contracts import Contract


class Policy(Protocol):
    contract: Contract

    def exercise(self, x: np.ndarray, k: int) -> np.ndarray:
        """Boolean array: exercise at date index k for each moneyness in x."""
        ...


class HoldToExpiry:
    """European behaviour: never exercise before expiry. The no-early-exercise reference."""

    def __init__(self, contract: Contract):
        self.contract = contract

    def exercise(self, x: np.ndarray, k: int) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float), dtype=bool)


class ThresholdPolicy:
    """Exercise a put when x <= b[k] (a call when x >= b[k]). Used to test boundary extraction
    on a policy whose boundary is known by construction."""

    def __init__(self, contract: Contract, boundary: np.ndarray):
        self.contract = contract
        self.b = np.asarray(boundary, dtype=float)

    def exercise(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x <= self.b[k] if self.contract.kind == "put" else x >= self.b[k]
