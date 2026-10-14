"""Contracts: payoff functions and the exercise schedule.

Conventions (used everywhere, easy to get subtly wrong):
- Exercise dates are t_k = k * dt for k = 0..steps, dt = expiry / steps. k = steps is expiry.
- The holder may exercise at any t_k including t_0 (American-style, discretised: a Bermudan
  option with steps + 1 exercise dates). At t_steps an in-the-money option is always exercised.
- Values are discounted to t_0 with exp(-rate * t_k). The one-step discount factor is
  exp(-rate * dt).
- The RL state is (x, tau) with x = S / K and tau = expiry - t_k. Payoffs in state units are
  payoff / K, so the problem is scale-free in K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from optstop.config import Config


@dataclass(frozen=True)
class Contract:
    kind: Literal["put", "call"]
    strike: float
    expiry: float
    steps: int
    rate: float

    @property
    def dt(self) -> float:
        return self.expiry / self.steps

    @property
    def step_discount(self) -> float:
        return float(np.exp(-self.rate * self.dt))

    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.expiry, self.steps + 1)

    def taus(self) -> np.ndarray:
        """Time to expiry at each exercise date k = 0..steps."""
        return self.expiry - self.times()

    def payoff(self, spot: np.ndarray | float) -> np.ndarray:
        """Immediate exercise value in currency units."""
        s = np.asarray(spot, dtype=float)
        if self.kind == "put":
            return np.maximum(self.strike - s, 0.0)
        return np.maximum(s - self.strike, 0.0)

    def payoff_x(self, x: np.ndarray | float) -> np.ndarray:
        """Immediate exercise value per unit strike, as a function of moneyness x = S/K."""
        x = np.asarray(x, dtype=float)
        if self.kind == "put":
            return np.maximum(1.0 - x, 0.0)
        return np.maximum(x - 1.0, 0.0)

    def with_(self, **kw) -> "Contract":
        d = dict(kind=self.kind, strike=self.strike, expiry=self.expiry, steps=self.steps,
                 rate=self.rate)
        d.update(kw)
        return Contract(**d)


def contract_from_config(cfg: Config) -> Contract:
    c = cfg.contract
    return Contract(kind=c.kind, strike=c.strike, expiry=c.expiry, steps=c.steps, rate=c.rate)
