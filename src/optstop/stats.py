"""Monte Carlo summary statistics. A point estimate without an interval is not a result."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Estimate:
    mean: float
    se: float
    ci_low: float
    ci_high: float
    n: int  # number of *independent* samples behind the estimate

    def to_dict(self) -> dict:
        return asdict(self)


def mc_estimate(values: np.ndarray, antithetic: bool = False, ci_level: float = 0.95) -> Estimate:
    """Mean, standard error and normal CI of iid samples.

    If `antithetic`, `values` is laid out as [draws, antithetic partners] (as produced by
    `paths.gbm_paths(..., antithetic=True)`); pair averages are the iid units, which is the only
    honest way to compute the SE of an antithetic estimator.
    """
    v = np.asarray(values, dtype=float)
    if antithetic:
        half = v.shape[0] // 2
        v = 0.5 * (v[:half] + v[half: 2 * half])
    n = v.shape[0]
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(0.5 + ci_level / 2)
    return Estimate(mean, se, mean - z * se, mean + z * se, n)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000,
                      ci_level: float = 0.95) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - ci_level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
