"""Gate C as a test: the agent, trained with the project settings on a no-dividend American
call, must (essentially) never exercise early. Uses exactly the Gate C criterion from config.

This test is expected to FAIL until the agent is fixed: see GATES.md (2026-10-14).
"""

import numpy as np

from optstop.agent.fitted_q import train_fitted_q
from optstop.baselines.analytic import bs_price
from optstop.baselines.lattice import bermudan_lattice
from optstop.contracts import contract_from_config
from optstop.evaluate.synthetic import rollout
from optstop.gates import rng_for
from optstop.paths import gbm_paths
from optstop.pipeline import _fit_paths
from optstop.stats import mc_estimate


def test_exact_reference_never_exercises_call(cfg):
    c = contract_from_config(cfg).with_(kind="call")
    lat = bermudan_lattice(c, 40.0, 0.2, substeps=40)
    assert np.all(np.isnan(lat.boundary_x[:-1]))


def test_agent_never_exercises_call_early(cfg):
    c = contract_from_config(cfg).with_(kind="call")
    vol, spot, gc = cfg.market.vol, cfg.market.spot, cfg.gates.c
    q = train_fitted_q(c, _fit_paths(cfg, c, vol, cfg.fitted_q.n_paths,
                                     rng_for(cfg.seed, "test_call_train")), cfg.fitted_q)
    ev = gbm_paths(spot, c.rate, vol, c.expiry, c.steps, gc.n_paths,
                   rng_for(cfg.seed, "test_call_eval"))
    ro = rollout(q, ev)
    early = float(np.mean(ro.stop_index < c.steps))
    est = mc_estimate(ro.values, antithetic=True)
    assert early <= gc.max_early_exercise_rate, f"early-exercise rate {early:.4%}"
    assert est.mean >= bs_price(spot, c.strike, c.rate, vol, c.expiry, "call") - gc.z_max * est.se
