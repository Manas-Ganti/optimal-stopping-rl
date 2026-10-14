"""Out-of-sample evaluation of synthetic-trained policies on real price paths.

Protocol (fixed before any real data was looked at, see GATES.md Gate E):
- Contract per window: an at-the-money American put written on day i (K = S_i), expiring
  one year (252 trading days) later, exercisable on `steps` dates spaced ~252/steps days apart.
- Volatility is calibrated on the `calibration_days` closes *up to and including* day i: no
  look-ahead. The policy used is the one trained on GBM at the nearest vol in `vol_grid`.
- Windows are non-overlapping (start days 252 trading days apart) so each is an independent
  option lifetime for a given ticker. They are NOT independent across tickers (shared market
  shocks), which `distribution.py` handles by bootstrapping over calendar-year clusters.
- Score: realised payoff discounted at the flat `rate`, per unit strike. Real paths are drawn
  under the real-world measure, so a realised mean is not a price; only *paired* comparisons
  between policies on the same path, and capture of the perfect-foresight payoff, are read.

Never trains on real paths, never uses option market prices.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from optstop.agent.fitted_q import train_fitted_q
from optstop.baselines.lsm import fit_lsm
from optstop.config import RESULTS_DIR, Config
from optstop.contracts import Contract
from optstop.data.fetch import fetch_close
from optstop.evaluate.synthetic import rollout
from optstop.gates import require_gates, rng_for
from optstop.paths import calibrate_gbm, dispersed_spots, gbm_paths
from optstop.policy import HoldToExpiry

BANK_PATH = RESULTS_DIR / "policies" / "real_bank.pkl"


def real_contract(cfg: Config, strike: float = 1.0) -> Contract:
    return Contract(kind="put", strike=strike, expiry=1.0, steps=cfg.contract.steps,
                    rate=cfg.real.rate)


def build_policy_bank(cfg: Config, verbose: bool = True) -> dict:
    """One (FQI, LSM) pair per vol in the grid, trained on GBM only, same recipe as Gate D."""
    if BANK_PATH.exists():
        bank = pickle.loads(BANK_PATH.read_bytes())
        if sorted(bank) == sorted(cfg.real.vol_grid):
            return bank
    c = real_contract(cfg, strike=1.0)
    bt = cfg.boundary_training
    bank = {}
    for vol in cfg.real.vol_grid:
        rng = rng_for(cfg.seed, f"real_bank_{vol}")
        def fit_paths(n):
            s0 = dispersed_spots(1.0, bt.spot_low, bt.spot_high, n, rng)
            return gbm_paths(s0, c.rate, vol, c.expiry, c.steps, n, rng)
        lsm = fit_lsm(c, fit_paths(cfg.lsm.n_paths), cfg.lsm.basis, cfg.lsm.degree)
        fqi = train_fitted_q(c, fit_paths(cfg.fitted_q.n_paths), cfg.fitted_q, seed=cfg.seed)
        bank[vol] = (fqi, lsm)
        if verbose:
            print(f"  trained policy pair for vol={vol:.2f}")
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    BANK_PATH.write_bytes(pickle.dumps(bank))
    return bank


def evaluate_real(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    require_gates("A", "B", "C", "D", stage="real-data evaluation (Gate E)")  # Gate E
    rc = cfg.real
    bank = build_policy_bank(cfg, verbose)
    grid = np.array(sorted(bank))
    T_days = rc.trading_days_per_year
    steps = cfg.contract.steps
    offsets = np.rint(np.arange(steps + 1) * T_days / steps).astype(int)
    rows = []
    for ticker in rc.tickers:
        try:
            s = fetch_close(ticker, rc.start, rc.end, rc.cache_dir)
        except Exception as e:
            print(f"  skipping {ticker}: {e!r}")
            continue
        prices = s.to_numpy()
        dates = s.index
        for i in range(rc.calibration_days, len(prices) - T_days, T_days):
            _, vol_hat = calibrate_gbm(prices[i - rc.calibration_days: i + 1], T_days)
            vol = float(grid[np.argmin(np.abs(grid - vol_hat))])
            fqi, lsm = bank[vol]
            K = float(prices[i])
            c = real_contract(cfg, strike=K)
            path = prices[i + offsets][None, :]
            r_fqi = rollout(fqi, path, c)
            r_lsm = rollout(lsm, path, c)
            r_eur = rollout(HoldToExpiry(c), path, c)
            disc_pay = np.exp(-c.rate * c.times()) * c.payoff_x(path[0] / K)
            rows.append({
                "ticker": ticker, "start": dates[i].date().isoformat(),
                "year": dates[i].year, "vol_hat": vol_hat, "vol_policy": vol,
                "v_fqi": r_fqi.values[0] / K, "v_lsm": r_lsm.values[0] / K,
                "v_hold": r_eur.values[0] / K, "v_foresight": float(disc_pay.max()),
                "stop_fqi": int(r_fqi.stop_index[0]), "stop_lsm": int(r_lsm.stop_index[0]),
                "final_x": float(path[0, -1] / K),
            })
        if verbose:
            print(f"  {ticker}: {sum(r['ticker'] == ticker for r in rows)} windows")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "real_windows.csv", index=False)
    return df
