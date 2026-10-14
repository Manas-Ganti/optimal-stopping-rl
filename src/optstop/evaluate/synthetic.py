"""Policy value on simulated paths, with confidence intervals.

Evaluation paths must be independent of whatever paths the policy was fitted on; otherwise the
estimate is optimistically biased (the policy has peeked at the futures it is scored on).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optstop.contracts import Contract
from optstop.policy import Policy
from optstop.stats import Estimate, mc_estimate


@dataclass
class Rollout:
    values: np.ndarray       # discounted payoff per path, currency units, discounted to t_0
    stop_index: np.ndarray   # exercise date index per path; steps + 1 => expired worthless


def rollout(policy: Policy, paths: np.ndarray, contract: Contract | None = None) -> Rollout:
    """Apply `policy` to every path. Exercise only ever happens in the money; at expiry an
    in-the-money option is always exercised."""
    c = contract or policy.contract
    n = paths.shape[0]
    x = paths / c.strike
    alive = np.ones(n, dtype=bool)
    values = np.zeros(n)
    stop = np.full(n, c.steps + 1)
    times = c.times()
    for k in range(c.steps + 1):
        pay = c.payoff_x(x[:, k])
        if k == c.steps:
            ex = alive & (pay > 0)
        else:
            ex = alive & (pay > 0) & policy.exercise(x[:, k], k)
        values[ex] = np.exp(-c.rate * times[k]) * pay[ex] * c.strike
        stop[ex] = k
        alive &= ~ex
    return Rollout(values=values, stop_index=stop)


def policy_value(policy: Policy, paths: np.ndarray, antithetic: bool = True,
                 ci_level: float = 0.95, contract: Contract | None = None) -> Estimate:
    return mc_estimate(rollout(policy, paths, contract).values, antithetic, ci_level)


def paired_difference(policy_a: Policy, policy_b: Policy, paths: np.ndarray,
                      antithetic: bool = True, ci_level: float = 0.95) -> Estimate:
    """V_a - V_b on common random numbers: far tighter than differencing two separate CIs."""
    va = rollout(policy_a, paths).values
    vb = rollout(policy_b, paths).values
    return mc_estimate(va - vb, antithetic, ci_level)
