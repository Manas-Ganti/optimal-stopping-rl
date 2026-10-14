"""Longstaff-Schwartz (2001) least-squares Monte Carlo.

Backward induction over the exercise dates. At each date k the continuation value of the
in-the-money paths is estimated by regressing the realised discounted future cash flow on a
small basis of the current moneyness; a path is exercised when immediate payoff >= fitted
continuation. The realised cash flow (not the fitted value) is then propagated backwards.

Structurally this is fitted Q-iteration with a *linear* approximator that is *separate for
every date* and fitted only on in-the-money states: Q_k(x, hold) ~ beta_k . phi(x). The agent in
`agent/fitted_q.py` solves the same Bellman equation with one approximator shared across time.
Agreement between the two is agreement on the same fixed point.

Assumes: the basis can represent the continuation value over the in-the-money region; enough
in-the-money paths at every date for the regression (fails near t=0 when all paths start at the
same spot, which is why date 0 then uses the sample mean instead; with dispersed initial
spots date 0 is regressed like any other); the fitted rule is only valid on
the region of state space the fitting paths visited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import laguerre

from optstop.contracts import Contract


def basis(x: np.ndarray, kind: str, degree: int) -> np.ndarray:
    """Design matrix. 'laguerre': constant + weighted Laguerre L_0..L_{degree-1} as in LS2001
    (exp(-x/2) L_n(x)); 'polynomial': 1, x, ..., x^degree."""
    x = np.asarray(x, dtype=float)
    if kind == "polynomial":
        return np.vander(x, degree + 1, increasing=True)
    if kind == "laguerre":
        w = np.exp(-0.5 * x)
        cols = [np.ones_like(x)]
        for n in range(degree):
            c = np.zeros(n + 1)
            c[n] = 1.0
            cols.append(w * laguerre.lagval(x, c))
        return np.stack(cols, axis=1)
    raise ValueError(f"unknown basis {kind!r}")


@dataclass
class LSMPolicy:
    """Exercise rule fitted by LSM. Implements `optstop.policy.Policy`."""

    contract: Contract
    basis_kind: str
    degree: int
    coefs: list[np.ndarray | None] = field(default_factory=list)  # per date k; None => no fit
    cont0: float = np.nan  # date-0 continuation value (valid only at the fitted spot)
    in_sample_price: float = np.nan

    def continuation(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if k == 0 and self.coefs[0] is None:
            return np.full_like(x, self.cont0)
        beta = self.coefs[k]
        if beta is None:
            return np.full_like(x, np.inf)  # no fit => never exercise
        return basis(x, self.basis_kind, self.degree) @ beta

    def exercise(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pay = self.contract.payoff_x(x)
        if k == self.contract.steps:
            return pay > 0
        return (pay > 0) & (pay >= self.continuation(x, k))


def fit_lsm(contract: Contract, paths: np.ndarray, basis_kind: str = "laguerre",
            degree: int = 3, min_itm: int = 50) -> LSMPolicy:
    """Fit the LSM exercise rule on `paths` (n_paths, steps + 1) in currency units.

    Regression is in moneyness x = S/K and per-unit-strike payoffs, which keeps the Laguerre
    weights exp(-x/2) well scaled for any strike.
    """
    x = paths / contract.strike
    steps = contract.steps
    disc = contract.step_discount
    cash = contract.payoff_x(x[:, steps])  # value at date k, discounted to date k
    coefs: list[np.ndarray | None] = [None] * (steps + 1)
    has_cross_section = np.ptp(x[:, 0]) > 0
    last = 0 if has_cross_section else 1
    for k in range(steps - 1, last - 1, -1):
        cash = cash * disc
        pay = contract.payoff_x(x[:, k])
        itm = pay > 0
        if itm.sum() < min_itm:
            continue
        A = basis(x[itm, k], basis_kind, degree)
        beta, *_ = np.linalg.lstsq(A, cash[itm], rcond=None)
        coefs[k] = beta
        ex = np.zeros_like(itm)
        ex[itm] = pay[itm] >= A @ beta
        cash = np.where(ex, pay, cash)
    if has_cross_section:  # date 0 handled by the regression above; no single price
        return LSMPolicy(contract=contract, basis_kind=basis_kind, degree=degree, coefs=coefs)
    cash = cash * disc
    cont0 = float(cash.mean())
    pay0 = float(contract.payoff_x(x[0, 0]))
    price = max(pay0, cont0) if pay0 > 0 else cont0
    return LSMPolicy(contract=contract, basis_kind=basis_kind, degree=degree, coefs=coefs,
                     cont0=cont0, in_sample_price=price * contract.strike)
