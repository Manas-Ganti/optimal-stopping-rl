"""The simulator samples what it claims to (a fast version of Gate A)."""

import numpy as np
import pytest

from optstop.baselines.analytic import bs_price, merton_price
from optstop.paths import (calibrate_gbm, dispersed_spots, gbm_paths, heston_paths,
                           merton_diffusive_vol, merton_paths)
from optstop.stats import mc_estimate


@pytest.mark.parametrize("spot,vol,T", [(36, 0.2, 1), (40, 0.4, 2), (44, 0.2, 1)])
def test_european_mc_matches_black_scholes(spot, vol, T, rng):
    P = gbm_paths(spot, 0.06, vol, T, 50 * T, 100_000, rng)
    est = mc_estimate(np.exp(-0.06 * T) * np.maximum(40 - P[:, -1], 0), antithetic=True)
    assert abs(est.mean - bs_price(spot, 40, 0.06, vol, T)) <= 3.5 * est.se


def test_merton_mc_matches_series(rng):
    P = merton_paths(40, 0.06, 0.15, 1.0, 50, 100_000, rng, 1.0, -0.1, 0.15)
    est = mc_estimate(np.exp(-0.06) * np.maximum(40 - P[:, -1], 0), antithetic=True)
    assert abs(est.mean - merton_price(40, 40, 0.06, 0.15, 1.0, 1.0, -0.1, 0.15)) <= 3.5 * est.se


def test_heston_discounted_spot_is_martingale(rng):
    P = heston_paths(40, 0.06, 0.04, 1.0, 50, 50_000, rng, 2.0, 0.04, 0.5, -0.7, substeps=4)
    est = mc_estimate(np.exp(-0.06) * P[:, -1])
    assert abs(est.mean - 40) <= 3.5 * est.se


def test_antithetic_layout_and_dispersed_spots(rng):
    s0 = dispersed_spots(40, 0.7, 1.3, 10, rng)
    assert np.all(s0[:5] == s0[5:]) and np.all((s0 >= 28) & (s0 <= 52))
    P = gbm_paths(s0, 0.06, 0.2, 1.0, 5, 10, rng)
    np.testing.assert_allclose(P[:, 0], s0)
    lr = np.diff(np.log(P), axis=1)
    np.testing.assert_allclose(lr[:5] + lr[5:], 2 * (0.06 - 0.02) * 0.2, atol=1e-12)


def test_matched_variance():
    vd = merton_diffusive_vol(0.2, 1.0, -0.1, 0.15)
    assert vd**2 + 1.0 * (0.01 + 0.0225) == pytest.approx(0.04)


def test_calibration_recovers_vol(rng):
    P = gbm_paths(100.0, 0.05, 0.3, 40.0, 40 * 252, 2, rng, antithetic=False)
    _, sigma = calibrate_gbm(P[0])
    assert sigma == pytest.approx(0.3, rel=0.03)
