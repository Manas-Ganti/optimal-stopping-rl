"""Summarise real-data outcomes as a distribution, never as one P&L number.

One historical path is a sample of size one. Every statistic here is over many underlyings
and start dates, with uncertainty from a cluster bootstrap over calendar *years* of the start
date (windows that start in the same year share market shocks across tickers, so resampling
individual windows would overstate the effective sample size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from optstop.gates import require_gates


def cluster_bootstrap(df: pd.DataFrame, stat, rng: np.random.Generator, cluster: str = "year",
                      n_boot: int = 5000, ci_level: float = 0.95) -> tuple[float, float, float]:
    groups = [g for _, g in df.groupby(cluster)]
    point = float(stat(df))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = stat(pd.concat([groups[j] for j in pick]))
    a = (1 - ci_level) / 2
    return point, float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a))


def summarise(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    require_gates("A", "B", "C", "D", stage="real-data evaluation (Gate E)")
    d = df.assign(diff=df.v_fqi - df.v_lsm)
    capture = lambda col: (lambda g: g[col].sum() / g.v_foresight.sum())  # noqa: E731
    out = {
        "n_windows": int(len(d)),
        "n_tickers": int(d.ticker.nunique()),
        "n_years": int(d.year.nunique()),
        "mean_value_per_100_strike": {
            p: _ci(cluster_bootstrap(d, lambda g, p=p: 100 * g[f"v_{p}"].mean(), rng))
            for p in ("fqi", "lsm", "hold", "foresight")},
        "paired_diff_fqi_minus_lsm_per_100_strike": _ci(
            cluster_bootstrap(d, lambda g: 100 * g["diff"].mean(), rng)),
        "capture_of_foresight": {
            p: _ci(cluster_bootstrap(d, capture(f"v_{p}"), rng)) for p in ("fqi", "lsm", "hold")},
        "fraction_windows": {
            "fqi_better": float((d["diff"] > 1e-12).mean()),
            "lsm_better": float((d["diff"] < -1e-12).mean()),
            "tie": float((d["diff"].abs() <= 1e-12).mean()),
            "same_stop_date": float((d.stop_fqi == d.stop_lsm).mean()),
        },
        "per_ticker": [],
    }
    for t, g in d.groupby("ticker"):
        out["per_ticker"].append({
            "ticker": t, "n": int(len(g)),
            "mean_diff_per_100_strike": float(100 * g["diff"].mean()),
            "mean_vol_hat": float(g.vol_hat.mean()),
            "capture_fqi": float(g.v_fqi.sum() / max(g.v_foresight.sum(), 1e-12)),
            "capture_lsm": float(g.v_lsm.sum() / max(g.v_foresight.sum(), 1e-12)),
        })
    return out


def _ci(t: tuple[float, float, float]) -> dict:
    return {"mean": t[0], "ci_low": t[1], "ci_high": t[2]}
