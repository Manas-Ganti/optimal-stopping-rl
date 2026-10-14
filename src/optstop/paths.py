"""Path samplers: the environment's dynamics. Nothing is learned here.

All samplers are risk-neutral (drift = rate, jump-compensated where relevant), seeded via an
explicit `np.random.Generator`, vectorised, and return an array of shape (n_paths, steps + 1)
whose column 0 is the spot. With `antithetic=True` the second half of the rows are the
antithetic partners of the first half (rows i and i + n/2); use `stats.mc_estimate(...,
antithetic=True)` for honest standard errors.

- GBM is sampled *exactly* (log-normal increments), so there is no time-discretisation bias.
- Merton jump-diffusion is sampled exactly at the grid (Poisson count, Gaussian log-jumps).
- Heston is sampled with a full-truncation Euler scheme on a finer sub-grid; this *does* have
  discretisation bias, which is acceptable because Heston is used only as an out-of-model
  stress test, never as a reference price.
"""

from __future__ import annotations

import numpy as np


def _normals(rng: np.random.Generator, n_paths: int, steps: int, antithetic: bool) -> np.ndarray:
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")
        z = rng.standard_normal((n_paths // 2, steps))
        return np.concatenate([z, -z], axis=0)
    return rng.standard_normal((n_paths, steps))


def _assemble(spot, log_increments: np.ndarray) -> np.ndarray:
    n = log_increments.shape[0]
    log_s = np.concatenate([np.zeros((n, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return np.reshape(np.asarray(spot, dtype=float), (-1, 1)) * np.exp(log_s)


def dispersed_spots(strike: float, low: float, high: float, n_paths: int,
                    rng: np.random.Generator, antithetic: bool = True) -> np.ndarray:
    """Initial spots S0 = K * U[low, high], one per path; antithetic partners share a spot.

    Used to fit exercise rules whose boundary must be read off over the whole (x, tau) plane:
    with a single S0, early dates are only visited near x = S0/K and the boundary there is an
    extrapolation (see GATES.md changes log, 2026-10-14).
    """
    if antithetic:
        half = strike * rng.uniform(low, high, n_paths // 2)
        return np.concatenate([half, half])
    return strike * rng.uniform(low, high, n_paths)


def gbm_paths(spot, rate: float, vol: float, expiry: float, steps: int, n_paths: int,
              rng: np.random.Generator, antithetic: bool = True) -> np.ndarray:
    """Risk-neutral geometric Brownian motion, sampled exactly on the exercise grid.

    `spot` is a scalar or one initial spot per path.

    Assumes constant rate and volatility and continuous, jump-free log-normal returns. Fails as
    a description of real prices exactly where real prices are interesting: fat tails, jumps,
    volatility clustering.
    """
    dt = expiry / steps
    z = _normals(rng, n_paths, steps, antithetic)
    return _assemble(spot, (rate - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z)


def merton_paths(spot, rate: float, vol: float, expiry: float, steps: int, n_paths: int,
                 rng: np.random.Generator, intensity: float, jump_mean: float, jump_std: float,
                 antithetic: bool = True) -> np.ndarray:
    """Merton (1976) jump-diffusion: GBM plus compound-Poisson log-normal jumps.

    `vol` is the diffusive volatility only. Drift is compensated so that e^{-rt} S_t is a
    martingale: mu = rate - intensity * (exp(jump_mean + jump_std^2 / 2) - 1).
    Antithetic pairing is applied to the diffusive normals only; jumps are independent.
    """
    dt = expiry / steps
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    z = _normals(rng, n_paths, steps, antithetic)
    n_jumps = rng.poisson(intensity * dt, size=(n_paths, steps))
    jumps = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * rng.standard_normal((n_paths, steps))
    drift = (rate - intensity * kappa - 0.5 * vol**2) * dt
    return _assemble(spot, drift + vol * np.sqrt(dt) * z + jumps)


def heston_paths(spot: float, rate: float, v0: float, expiry: float, steps: int, n_paths: int,
                 rng: np.random.Generator, kappa: float, theta: float, xi: float, rho: float,
                 substeps: int = 8) -> np.ndarray:
    """Heston (1993) stochastic volatility, full-truncation Euler with `substeps` per grid step.

    Has O(dt) discretisation bias and is not antithetic. Use for stress tests only.
    """
    dt = expiry / (steps * substeps)
    log_s = np.zeros(n_paths)
    v = np.full(n_paths, v0)
    out = np.empty((n_paths, steps + 1))
    out[:, 0] = spot
    for k in range(steps):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)
            log_s += (rate - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        out[:, k + 1] = spot * np.exp(log_s)
    return out


def merton_diffusive_vol(total_vol: float, intensity: float, jump_mean: float,
                         jump_std: float) -> float:
    """Diffusive vol such that total annualised log-return variance equals total_vol^2.

    Var per year = vol_d^2 + intensity * (jump_mean^2 + jump_std^2). Raises if the jumps alone
    already exceed the target variance.
    """
    jump_var = intensity * (jump_mean**2 + jump_std**2)
    resid = total_vol**2 - jump_var
    if resid <= 0:
        raise ValueError(f"jump variance {jump_var:.4f} exceeds target {total_vol**2:.4f}")
    return float(np.sqrt(resid))


def calibrate_gbm(prices: np.ndarray, trading_days_per_year: int = 252) -> tuple[float, float]:
    """Annualised (drift, volatility) of a daily price series, from log returns.

    Drift is the real-world drift mu with E[dS/S] = mu dt (i.e. log-mean + sigma^2/2). It is
    estimated very poorly from any realistic sample (standard error ~ sigma / sqrt(years)), which
    is one reason pricing uses the risk-neutral drift rather than this number.
    """
    r = np.diff(np.log(np.asarray(prices, dtype=float)))
    r = r[np.isfinite(r)]
    sigma = float(r.std(ddof=1) * np.sqrt(trading_days_per_year))
    mu = float(r.mean() * trading_days_per_year + 0.5 * sigma**2)
    return mu, sigma
