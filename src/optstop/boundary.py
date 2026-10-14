"""Exercise-boundary extraction from any policy, plus the figures.

The boundary at date k is read off the policy by querying it on a fine moneyness grid:
  put:  b(k) = max{ x in grid : policy exercises at (x, k) }   (exercise region is x <= b)
  call: b(k) = min{ x in grid : policy exercises at (x, k) }   (exercise region is x >= b)
nan if the policy never exercises at that date. Because this only uses `policy.exercise`, it
treats LSM, the learned agent and hand-built threshold rules identically.

`holes` counts grid points on the exercise side of b where the policy nevertheless holds: a
well-formed stopping rule has a connected exercise region, so holes > 0 flags approximation
artefacts. Holes are counted only inside the fitted data's support (if given);
outside it every regression extrapolates. Resolution is the grid spacing; anything finer is not claimed.

The headline figure is drawn by `figures.plot_boundaries` from `results/metrics.json`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optstop.policy import Policy


@dataclass
class Boundary:
    tau: np.ndarray   # time to expiry at each date k = 0..steps
    b: np.ndarray     # boundary in S/K units, nan where no exercise
    holes: np.ndarray

    def to_dict(self) -> dict:
        return {"tau": _clean(self.tau), "b": _clean(self.b), "holes": self.holes.tolist()}


def _clean(a: np.ndarray) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def path_support(paths: np.ndarray, strike: float, q: float = 0.005) -> np.ndarray:
    """Per-date [low, high] S/K quantiles of the paths a rule was fitted on: shape (steps+1, 2).
    Outside this band a fitted rule is extrapolating and its 'boundary' means nothing."""
    x = paths / strike
    return np.stack([np.quantile(x, q, axis=0), np.quantile(x, 1 - q, axis=0)], axis=1)


def extract_boundary(policy: Policy, x_grid: np.ndarray, dates: range | None = None,
                     support: np.ndarray | None = None) -> Boundary:
    """`support` (from `path_support`) restricts the *holes* count to states the fit saw."""
    c = policy.contract
    dates = dates if dates is not None else range(c.steps + 1)
    b = np.full(c.steps + 1, np.nan)
    holes = np.zeros(c.steps + 1, dtype=int)
    x_grid = np.sort(np.asarray(x_grid, dtype=float))
    for k in dates:
        ex = np.asarray(policy.exercise(x_grid, k), dtype=bool)
        if not ex.any():
            continue
        seen = np.ones_like(ex) if support is None else \
            (x_grid >= support[k, 0]) & (x_grid <= support[k, 1])
        if c.kind == "put":
            i = np.flatnonzero(ex).max()
            b[k] = x_grid[i]
            side = (np.arange(len(x_grid)) <= i) & (x_grid < 1.0)
        else:
            i = np.flatnonzero(ex).min()
            b[k] = x_grid[i]
            side = (np.arange(len(x_grid)) >= i) & (x_grid > 1.0)
        holes[k] = int((~ex & side & seen).sum())
    return Boundary(tau=c.taus(), b=b, holes=holes)


def compare(b1: Boundary, b2: Boundary, dates: np.ndarray) -> dict:
    """Agreement statistics over the given dates where both boundaries exist."""
    d = b1.b[dates] - b2.b[dates]
    ok = np.isfinite(d)
    if not ok.any():
        return {"n_dates": 0, "mean_abs": None, "max_abs": None}
    return {"n_dates": int(ok.sum()), "n_missing": int((~ok).sum()),
            "mean_abs": float(np.abs(d[ok]).mean()), "max_abs": float(np.abs(d[ok]).max()),
            "mean_signed": float(d[ok].mean())}
