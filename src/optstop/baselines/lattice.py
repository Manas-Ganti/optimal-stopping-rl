"""Exact reference for the Bermudan problem: a Cox-Ross-Rubinstein binomial lattice in which
exercise is allowed only at the contract's exercise dates.

Not a competitor and not a gate threshold. It answers "what is the exact price and boundary
for *this* exercise schedule under GBM", so LSM and the agent can both be compared to the
truth rather than only to each other. Converges at O(1/n) in lattice steps; assumes GBM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optstop.contracts import Contract


@dataclass
class LatticeResult:
    price: float
    boundary_x: np.ndarray  # S/K boundary at each exercise date k = 0..steps (nan if none)


def bermudan_lattice(contract: Contract, spot: float, vol: float,
                     substeps: int = 200, american: bool = False) -> LatticeResult:
    """Price by backward induction on a CRR tree with `substeps` tree steps per exercise date.

    If `american`, exercise is allowed at every tree node (approximating continuous exercise).
    The put boundary at date k is the largest node price at which exercise is optimal, i.e. the
    edge of the exercise region resolved to one tree-node spacing.
    """
    n = contract.steps * substeps
    h = contract.expiry / n
    u = np.exp(vol * np.sqrt(h))
    d = 1.0 / u
    p = (np.exp(contract.rate * h) - d) / (u - d)
    disc = np.exp(-contract.rate * h)
    k_strike = contract.strike

    j = np.arange(n + 1)
    s = spot * u ** (n - 2 * j)  # level n nodes, descending in price
    v = contract.payoff(s)
    boundary = np.full(contract.steps + 1, np.nan)
    boundary[-1] = 1.0
    for level in range(n - 1, -1, -1):
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        s = spot * u ** (level - 2 * np.arange(level + 1))
        if american or level % substeps == 0:
            ex_val = contract.payoff(s)
            ex = (ex_val >= v) & (ex_val > 0)
            v = np.where(ex, ex_val, v)
            if level % substeps == 0 and ex.any():
                kk = level // substeps
                sel = s[ex]
                boundary[kk] = (sel.max() if contract.kind == "put" else sel.min()) / k_strike
    return LatticeResult(price=float(v[0]), boundary_x=boundary)
