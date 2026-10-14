"""Closed forms against hand-worked and published values."""

import numpy as np
import pytest

from optstop.baselines.analytic import (bs_price, merton_price, perpetual_put_boundary,
                                        perpetual_put_price)
from optstop.baselines.benchmarks import LS2001_RATE, LS2001_STRIKE, LS2001_TABLE1
from optstop.baselines.lattice import bermudan_lattice
from optstop.contracts import Contract


@pytest.mark.parametrize("spot,vol,T,fd,eur,lsm", LS2001_TABLE1)
def test_black_scholes_matches_ls2001_european_column(spot, vol, T, fd, eur, lsm):
    assert bs_price(spot, LS2001_STRIKE, LS2001_RATE, vol, T, "put") == pytest.approx(eur, abs=6e-4)


def test_put_call_parity():
    S, K, r, v, T = 37.0, 40.0, 0.05, 0.3, 0.75
    lhs = bs_price(S, K, r, v, T, "call") - bs_price(S, K, r, v, T, "put")
    assert lhs == pytest.approx(S - K * np.exp(-r * T), abs=1e-12)


def test_perpetual_put_hand_worked():
    # r = 0.06, sigma = 0.2 => g = 2r/sigma^2 = 3, b* = K g/(1+g) = 30,
    # P(40) = (K - b*) (S/b*)^(-g) = 10 * (4/3)^(-3) = 4.21875
    assert perpetual_put_boundary(40.0, 0.06, 0.2) == pytest.approx(30.0)
    assert float(perpetual_put_price(40.0, 40.0, 0.06, 0.2)) == pytest.approx(4.21875)
    assert float(perpetual_put_price(25.0, 40.0, 0.06, 0.2)) == pytest.approx(15.0)  # exercise region


def test_merton_reduces_to_black_scholes_without_jumps():
    a = merton_price(38.0, 40.0, 0.06, 0.25, 1.0, intensity=0.0, jump_mean=0.0, jump_std=0.1)
    assert a == pytest.approx(bs_price(38.0, 40.0, 0.06, 0.25, 1.0), abs=1e-10)


def test_lattice_call_is_european_and_put_boundary_above_perpetual():
    c = Contract("call", 40.0, 1.0, 50, 0.06)
    lat = bermudan_lattice(c, 40.0, 0.2, substeps=40)
    assert lat.price == pytest.approx(bs_price(40.0, 40.0, 0.06, 0.2, 1.0, "call"), abs=0.01)
    assert np.all(np.isnan(lat.boundary_x[:-1]))  # never exercised before expiry
    p = bermudan_lattice(c.with_(kind="put"), 40.0, 0.2, substeps=40)
    b = p.boundary_x[1:-1]
    assert np.all(b[np.isfinite(b)] >= perpetual_put_boundary(1.0, 0.06, 0.2))
