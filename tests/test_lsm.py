"""LSM reproduces published American put prices (a fast version of Gate B)."""

import numpy as np
import pytest

from optstop.baselines.benchmarks import LS2001_TABLE1
from optstop.baselines.lsm import basis, fit_lsm
from optstop.contracts import Contract
from optstop.evaluate.synthetic import policy_value
from optstop.paths import gbm_paths

CASES = [row for row in LS2001_TABLE1 if row[2] == 1 and row[1] == 0.2]


@pytest.mark.parametrize("spot,vol,T,fd,eur,lsm_paper", CASES)
def test_lsm_matches_ls2001(spot, vol, T, fd, eur, lsm_paper, rng):
    c = Contract("put", 40.0, float(T), 50 * T, 0.06)
    pol = fit_lsm(c, gbm_paths(spot, 0.06, vol, T, c.steps, 100_000, rng))
    est = policy_value(pol, gbm_paths(spot, 0.06, vol, T, c.steps, 100_000, rng))
    assert abs(est.mean - fd) <= 0.04
    assert abs(pol.in_sample_price - fd) <= 0.04


def test_lsm_never_beats_european_by_negative_amount(rng, put):
    from optstop.baselines.analytic import bs_price

    pol = fit_lsm(put, gbm_paths(40, 0.06, 0.2, 1, 50, 50_000, rng))
    est = policy_value(pol, gbm_paths(40, 0.06, 0.2, 1, 50, 50_000, rng))
    assert est.mean > bs_price(40, 40, 0.06, 0.2, 1.0)  # early-exercise premium is positive


def test_basis_shapes():
    x = np.linspace(0.5, 1.5, 7)
    assert basis(x, "laguerre", 3).shape == (7, 4)
    assert basis(x, "polynomial", 3).shape == (7, 4)
    np.testing.assert_allclose(basis(x, "laguerre", 1)[:, 1], np.exp(-x / 2))
