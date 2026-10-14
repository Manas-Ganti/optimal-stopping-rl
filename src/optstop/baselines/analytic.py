"""Closed forms. Used for the validation gates, never as competitors.

- Black-Scholes European put/call: assumes GBM, constant r and sigma, no dividends.
- Merton (1976) jump-diffusion European put/call: series over the number of jumps; assumes
  log-normal jump sizes and that jump risk is not priced beyond the compensator.
- Perpetual American put (McKean 1965 / Merton 1973): the only American put with a closed
  form. Its boundary is the T -> infinity limit of the finite-expiry boundary, so any finite
  boundary must lie above it.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm


def bs_price(spot, strike, rate, vol, expiry, kind: str = "put"):
    spot = np.asarray(spot, dtype=float)
    sqrt_t = np.sqrt(expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    disc = strike * np.exp(-rate * expiry)
    if kind == "put":
        return disc * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return spot * norm.cdf(d1) - disc * norm.cdf(d2)


def merton_price(spot, strike, rate, vol, expiry, intensity, jump_mean, jump_std,
                 kind: str = "put", n_terms: int = 80):
    """Merton jump-diffusion European price. `vol` is the diffusive vol only."""
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    lam_p = intensity * (1.0 + kappa)
    total = 0.0
    for n in range(n_terms):
        weight = np.exp(-lam_p * expiry + n * np.log(lam_p * expiry + 1e-300) - gammaln(n + 1))
        vol_n = np.sqrt(vol**2 + n * jump_std**2 / expiry)
        rate_n = rate - intensity * kappa + n * np.log1p(kappa) / expiry
        total = total + weight * bs_price(spot, strike, rate_n, vol_n, expiry, kind)
    return total


def perpetual_put_boundary(strike: float, rate: float, vol: float) -> float:
    """Optimal exercise level b* = K * g / (1 + g), g = 2r / sigma^2. Requires rate > 0."""
    g = 2.0 * rate / vol**2
    return strike * g / (1.0 + g)


def perpetual_put_price(spot, strike: float, rate: float, vol: float):
    b = perpetual_put_boundary(strike, rate, vol)
    g = 2.0 * rate / vol**2
    spot = np.asarray(spot, dtype=float)
    return np.where(spot <= b, strike - spot, (strike - b) * (spot / b) ** (-g))
