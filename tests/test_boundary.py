"""Boundary extraction is correct on policies whose boundary is known by construction."""

import numpy as np

from optstop.boundary import compare, extract_boundary
from optstop.contracts import Contract
from optstop.policy import HoldToExpiry, ThresholdPolicy

GRID = np.linspace(0.4, 1.6, 1201)  # spacing 0.001


def test_threshold_put_recovered(put):
    true_b = np.linspace(0.80, 1.0, put.steps + 1)
    b = extract_boundary(ThresholdPolicy(put, true_b), GRID)
    np.testing.assert_allclose(b.b, true_b, atol=0.001 + 1e-12)
    assert b.holes.sum() == 0
    assert compare(b, b, np.arange(put.steps))["max_abs"] == 0.0


def test_threshold_call_recovered():
    call = Contract("call", 40.0, 1.0, 10, 0.06)
    true_b = np.linspace(1.4, 1.0, 11)
    b = extract_boundary(ThresholdPolicy(call, true_b), GRID)
    np.testing.assert_allclose(b.b, true_b, atol=0.001 + 1e-12)


def test_never_exercise_gives_nan(put):
    b = extract_boundary(HoldToExpiry(put), GRID)
    assert np.all(np.isnan(b.b))


def test_holes_detected_and_support_respected(put):
    class Holey:
        contract = put

        def exercise(self, x, k):
            x = np.asarray(x)
            return (x <= 0.9) & ~((x > 0.5) & (x < 0.6))

    b = extract_boundary(Holey(), GRID)
    assert b.b[5] == np.float64(GRID[GRID <= 0.9].max())
    assert b.holes[5] > 90
    support = np.tile([0.7, 1.3], (put.steps + 1, 1))
    assert extract_boundary(Holey(), GRID, support=support).holes[5] == 0
