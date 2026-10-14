"""The four stages behind the CLI, in the order the project must be built and believed:

  baseline -> Gates A (simulator) and B (LSM)
  train    -> Gate C (agent never exercises a no-dividend call early); put policies saved
  boundary -> Gate D (learned boundary agrees with LSM); headline figure
  evaluate -> synthetic values with CIs, misspecification study, and (Gate E) real data

Each stage writes its numbers to results/metrics.json and its gate to results/gates.json.
"""

from __future__ import annotations

import pickle
import time

import numpy as np

from optstop.agent.fitted_q import train_fitted_q
from optstop.baselines.analytic import (bs_price, merton_price, perpetual_put_boundary)
from optstop.baselines.benchmarks import (LS2001_DATES_PER_YEAR, LS2001_RATE, LS2001_STRIKE,
                                          LS2001_TABLE1)
from optstop.baselines.lattice import bermudan_lattice
from optstop.baselines.lsm import fit_lsm
from optstop.boundary import Boundary, compare, extract_boundary, path_support
from optstop.config import FIGURES_DIR, RESULTS_DIR, Config
from optstop.contracts import Contract, contract_from_config
from optstop.evaluate.synthetic import paired_difference, policy_value, rollout
from optstop.gates import (METRICS_PATH, read_metrics, record_gate, require_gates, rng_for,
                           update_metrics)
from optstop.paths import (dispersed_spots, gbm_paths, heston_paths, merton_diffusive_vol,
                           merton_paths)
from optstop.policy import HoldToExpiry
from optstop.stats import mc_estimate

POLICY_DIR = RESULTS_DIR / "policies"


def _fit_paths(cfg: Config, c: Contract, vol: float, n: int, rng: np.random.Generator):
    """Training/fitting paths for the boundary experiment: dispersed initial spots."""
    bt = cfg.boundary_training
    s0 = dispersed_spots(c.strike, bt.spot_low, bt.spot_high, n, rng)
    return gbm_paths(s0, c.rate, vol, c.expiry, c.steps, n, rng)


# ---------------------------------------------------------------------------------- baseline
def run_baseline(cfg: Config) -> dict:
    ga = cfg.gates.a
    z_max = ga.z_max

    # Gate A1: GBM European put vs Black-Scholes, all 20 LS2001 parameter sets.
    a1 = []
    for spot, vol, T, *_ in LS2001_TABLE1:
        steps = LS2001_DATES_PER_YEAR * T
        rng = rng_for(cfg.seed, f"gateA_gbm_{spot}_{vol}_{T}")
        P = gbm_paths(spot, LS2001_RATE, vol, T, steps, ga.n_paths, rng)
        disc = np.exp(-LS2001_RATE * T) * np.maximum(LS2001_STRIKE - P[:, -1], 0.0)
        est = mc_estimate(disc, antithetic=True)
        bs = float(bs_price(spot, LS2001_STRIKE, LS2001_RATE, vol, T, "put"))
        z = (est.mean - bs) / est.se
        a1.append({"spot": spot, "vol": vol, "expiry": T, "mc": est.mean, "se": est.se,
                   "bs": bs, "z": z, "pass": abs(z) <= z_max})

    # Gate A2: Merton jump-diffusion vs Merton series.
    a2 = []
    for case in ga.merton_cases:
        rng = rng_for(cfg.seed, f"gateA_merton_{case.spot}_{case.vol}_{case.expiry}")
        steps = max(1, int(round(LS2001_DATES_PER_YEAR * case.expiry)))
        P = merton_paths(case.spot, LS2001_RATE, case.vol, case.expiry, steps, ga.n_paths, rng,
                         case.intensity, case.jump_mean, case.jump_std)
        disc = np.exp(-LS2001_RATE * case.expiry) * np.maximum(LS2001_STRIKE - P[:, -1], 0.0)
        est = mc_estimate(disc, antithetic=True)
        ref = float(merton_price(case.spot, LS2001_STRIKE, LS2001_RATE, case.vol, case.expiry,
                                 case.intensity, case.jump_mean, case.jump_std, "put"))
        z = (est.mean - ref) / est.se
        a2.append({**case.model_dump(), "mc": est.mean, "se": est.se, "merton": ref, "z": z,
                   "pass": abs(z) <= z_max})

    # Gate A3: Heston discounted spot is a martingale.
    h = cfg.misspecification.heston
    vol = cfg.market.vol
    rng = rng_for(cfg.seed, "gateA_heston")
    c = contract_from_config(cfg)
    P = heston_paths(cfg.market.spot, c.rate, h.theta_scale * vol**2, c.expiry, c.steps,
                     ga.n_paths, rng, h.kappa, h.theta_scale * vol**2, h.xi, h.rho)
    est = mc_estimate(np.exp(-c.rate * c.expiry) * P[:, -1])
    z = (est.mean - cfg.market.spot) / est.se
    a3 = {"mean_discounted_ST": est.mean, "se": est.se, "spot": cfg.market.spot, "z": z,
          "pass": abs(z) <= z_max}

    a_pass = all(r["pass"] for r in a1) and all(r["pass"] for r in a2) and a3["pass"]
    record_gate("A", a_pass, {"z_max": z_max, "n_paths": ga.n_paths,
                              "max_abs_z_gbm": max(abs(r["z"]) for r in a1),
                              "max_abs_z_merton": max(abs(r["z"]) for r in a2),
                              "z_heston_martingale": z}, cfg.seed)

    # Gate B: LSM vs LS2001 finite-difference American put values.
    gb = cfg.gates.b
    b_rows = []
    for spot, vol, T, fd, eur, lsm_paper in LS2001_TABLE1:
        steps = LS2001_DATES_PER_YEAR * T
        cT = Contract("put", LS2001_STRIKE, float(T), steps, LS2001_RATE)
        fit = gbm_paths(spot, LS2001_RATE, vol, T, steps, cfg.lsm.n_paths,
                        rng_for(cfg.seed, f"gateB_fit_{spot}_{vol}_{T}"))
        ev = gbm_paths(spot, LS2001_RATE, vol, T, steps, cfg.lsm.n_paths,
                       rng_for(cfg.seed, f"gateB_eval_{spot}_{vol}_{T}"))
        pol = fit_lsm(cT, fit, cfg.lsm.basis, cfg.lsm.degree)
        oos = policy_value(pol, ev)
        lat = bermudan_lattice(cT, spot, vol, substeps=100)
        b_rows.append({"spot": spot, "vol": vol, "expiry": T, "fd_published": fd,
                       "lsm_published": lsm_paper, "european_published": eur,
                       "lsm_in_sample": pol.in_sample_price, "lsm_oos": oos.mean, "lsm_oos_se": oos.se,
                       "bermudan_lattice": lat.price, "abs_err_vs_fd": abs(oos.mean - fd),
                       "pass": abs(oos.mean - fd) <= gb.abs_tol})
    mean_abs = float(np.mean([r["abs_err_vs_fd"] for r in b_rows]))
    b_pass = all(r["pass"] for r in b_rows) and mean_abs <= gb.mean_abs_tol
    record_gate("B", b_pass, {"abs_tol": gb.abs_tol, "mean_abs_tol": gb.mean_abs_tol,
                              "mean_abs_err": mean_abs,
                              "max_abs_err": max(r["abs_err_vs_fd"] for r in b_rows),
                              "max_abs_err_vs_lattice": max(abs(r["lsm_oos"] - r["bermudan_lattice"])
                                                            for r in b_rows)}, cfg.seed)
    out = {"gate_a": {"gbm_vs_black_scholes": a1, "merton_vs_series": a2, "heston_martingale": a3},
           "gate_b": {"cases": b_rows, "mean_abs_err": mean_abs}}
    update_metrics("baseline", out, cfg.seed)
    return out


# ------------------------------------------------------------------------------------- train
def run_train(cfg: Config) -> dict:
    require_gates("A", "B", stage="train")
    c = contract_from_config(cfg)
    vol, spot = cfg.market.vol, cfg.market.spot

    # Gate C: same agent, same settings, on a no-dividend American call.
    gc = cfg.gates.c
    call = c.with_(kind="call")
    t0 = time.time()
    q_call = train_fitted_q(call, _fit_paths(cfg, call, vol, cfg.fitted_q.n_paths,
                                             rng_for(cfg.seed, "train_call")), cfg.fitted_q, cfg.seed)
    ev = gbm_paths(spot, c.rate, vol, c.expiry, c.steps, gc.n_paths, rng_for(cfg.seed, "gateC_eval"))
    ro = rollout(q_call, ev)
    early = float(np.mean(ro.stop_index < c.steps))
    est = mc_estimate(ro.values, antithetic=True, ci_level=cfg.evaluation.ci_level)
    bs_call = float(bs_price(spot, c.strike, c.rate, vol, c.expiry, "call"))
    z = (est.mean - bs_call) / est.se
    xg = _x_grid(cfg)
    call_b = extract_boundary(q_call, xg, range(c.steps))
    c_pass = early <= gc.max_early_exercise_rate and z >= -gc.z_max
    record_gate("C", c_pass, {"early_exercise_rate": early, "max_early_exercise_rate":
                              gc.max_early_exercise_rate, "value": est.mean, "se": est.se,
                              "bs_call": bs_call, "z": z,
                              "dates_with_any_exercise_on_grid": int(np.isfinite(call_b.b).sum())},
                cfg.seed)
    t_call = time.time() - t0

    # The put experiment: LSM and FQI fitted on the same dispersed-start distribution.
    t0 = time.time()
    lsm_paths = _fit_paths(cfg, c, vol, cfg.lsm.n_paths, rng_for(cfg.seed, "fit_lsm_put"))
    lsm = fit_lsm(c, lsm_paths, cfg.lsm.basis, cfg.lsm.degree)
    t_lsm = time.time() - t0
    t0 = time.time()
    fqi_paths = _fit_paths(cfg, c, vol, cfg.fitted_q.n_paths, rng_for(cfg.seed, "train_fqi_put"))
    fqi = train_fitted_q(c, fqi_paths, cfg.fitted_q, cfg.seed)
    t_fqi = time.time() - t0
    POLICY_DIR.mkdir(parents=True, exist_ok=True)
    (POLICY_DIR / "put_policies.pkl").write_bytes(pickle.dumps({
        "lsm": lsm, "fitted_q": fqi, "support": path_support(lsm_paths, c.strike)}))
    out = {"gate_c": {"early_exercise_rate": early, "value": est.to_dict(), "bs_call": bs_call,
                      "z": z, "train_seconds": t_call},
           "put": {"fitted_q_regressor": cfg.fitted_q.regressor, "fitted_q_paths": cfg.fitted_q.n_paths,
                   "fitted_q_iterations": len(fqi.history),
                   "fitted_q_final_max_dq": fqi.history[-1], "fitted_q_seconds": t_fqi,
                   "lsm_paths": cfg.lsm.n_paths, "lsm_seconds": t_lsm}}
    update_metrics("train", out, cfg.seed)
    return out


def _x_grid(cfg: Config) -> np.ndarray:
    b = cfg.boundary
    return np.linspace(b.x_grid_min, b.x_grid_max, b.x_grid_points)


def _load_put_policies() -> dict:
    return pickle.loads((POLICY_DIR / "put_policies.pkl").read_bytes())


# ---------------------------------------------------------------------------------- boundary
def run_boundary(cfg: Config) -> dict:
    require_gates("A", "B", "C", stage="boundary")
    c = contract_from_config(cfg)
    vol, spot = cfg.market.vol, cfg.market.spot
    pol = _load_put_policies()
    xg = _x_grid(cfg)
    bl = extract_boundary(pol["lsm"], xg, support=pol["support"])
    bq = extract_boundary(pol["fitted_q"], xg, support=pol["support"])
    lat = bermudan_lattice(c, spot, vol, substeps=200)
    blat = Boundary(c.taus(), lat.boundary_x, np.zeros(c.steps + 1, dtype=int))
    dates = np.arange(0, c.steps)  # strictly before expiry
    agree = compare(bq, bl, dates)

    ev = gbm_paths(spot, c.rate, vol, c.expiry, c.steps, cfg.evaluation.n_paths,
                   rng_for(cfg.seed, "gateD_eval"))
    v_q = policy_value(pol["fitted_q"], ev, ci_level=cfg.evaluation.ci_level)
    v_l = policy_value(pol["lsm"], ev, ci_level=cfg.evaluation.ci_level)
    diff = paired_difference(pol["fitted_q"], pol["lsm"], ev, ci_level=cfg.evaluation.ci_level)
    rel = abs(v_q.mean - v_l.mean) / v_l.mean
    gd = cfg.gates.d
    d_pass = (agree["mean_abs"] is not None and agree["mean_abs"] <= gd.mean_abs_boundary_tol
              and agree["max_abs"] <= gd.max_abs_boundary_tol and rel <= gd.rel_value_tol)
    record_gate("D", d_pass, {"boundary": agree, "rel_value_diff": rel,
                              "thresholds": gd.model_dump()}, cfg.seed)
    out = {
        "contract": {"strike": c.strike, "expiry": c.expiry, "steps": c.steps, "rate": c.rate,
                     "kind": c.kind},
        "vol": vol, "spot": spot,
        "perpetual_b": perpetual_put_boundary(1.0, c.rate, vol),
        "curves": {"lsm": bl.to_dict(), "fitted_q": bq.to_dict(), "lattice": blat.to_dict()},
        "agreement": {"fitted_q_vs_lsm": agree, "lsm_vs_lattice": compare(bl, blat, dates),
                      "fitted_q_vs_lattice": compare(bq, blat, dates)},
        "holes_in_support": {"lsm": int(bl.holes[:-1].sum()), "fitted_q": int(bq.holes[:-1].sum())},
        "values_at_spot": {"fitted_q": v_q.to_dict(), "lsm": v_l.to_dict(),
                           "paired_diff_fitted_q_minus_lsm": diff.to_dict(),
                           "bermudan_lattice_exact": lat.price,
                           "european_bs": float(bs_price(spot, c.strike, c.rate, vol, c.expiry))},
    }
    update_metrics("boundary", out, cfg.seed)
    from optstop.figures import plot_boundaries
    fig = plot_boundaries(METRICS_PATH, FIGURES_DIR / "boundary.png")
    print(f"  figure: {fig}")
    return out


# ---------------------------------------------------------------------------------- evaluate
def run_synthetic(cfg: Config) -> dict:
    """Values under the training dynamics at several spots, against the exact lattice."""
    require_gates("A", "B", "C", "D", stage="synthetic evaluation")
    c = contract_from_config(cfg)
    vol = cfg.market.vol
    pol = _load_put_policies()
    rows = []
    for spot in (36.0, 40.0, 44.0):
        ev = gbm_paths(spot, c.rate, vol, c.expiry, c.steps, cfg.evaluation.n_paths,
                       rng_for(cfg.seed, f"synthetic_eval_{spot}"))
        rows.append({
            "spot": spot,
            "fitted_q": policy_value(pol["fitted_q"], ev, ci_level=cfg.evaluation.ci_level).to_dict(),
            "lsm": policy_value(pol["lsm"], ev, ci_level=cfg.evaluation.ci_level).to_dict(),
            "paired_diff": paired_difference(pol["fitted_q"], pol["lsm"], ev,
                                             ci_level=cfg.evaluation.ci_level).to_dict(),
            "bermudan_lattice_exact": bermudan_lattice(c, spot, vol, substeps=200).price,
            "european_bs": float(bs_price(spot, c.strike, c.rate, vol, c.expiry)),
        })
    return {"gbm_by_spot": rows}


def run_misspecification(cfg: Config) -> dict:
    """Train on GBM, evaluate on GBM / Merton / Heston with total variance matched.

    Reference in each regime: LSM *refitted on that regime's own paths* (fixed spot), i.e. a
    classical method that knows the true dynamics. Regret = V_refit - V_policy on common paths.
    The finding is the regret of the learned policy relative to the regret of GBM-fitted LSM.
    """
    require_gates("A", "B", "C", "D", stage="misspecification study")
    c = contract_from_config(cfg)
    vol, spot = cfg.market.vol, cfg.market.spot
    ms = cfg.misspecification
    n = ms.n_paths
    pol = _load_put_policies()
    mj = ms.merton
    vol_d = merton_diffusive_vol(vol, mj.intensity, mj.jump_mean, mj.jump_std)
    h = ms.heston
    v_bar = h.theta_scale * vol**2

    def sampler(regime: str, purpose: str):
        rng = rng_for(cfg.seed, f"misspec_{regime}_{purpose}")
        if regime == "gbm":
            return gbm_paths(spot, c.rate, vol, c.expiry, c.steps, n, rng), True
        if regime == "merton":
            return merton_paths(spot, c.rate, vol_d, c.expiry, c.steps, n, rng, mj.intensity,
                                mj.jump_mean, mj.jump_std), True
        return heston_paths(spot, c.rate, v_bar, c.expiry, c.steps, n, rng, h.kappa, v_bar,
                            h.xi, h.rho), False

    labels = {"gbm": "GBM (training dynamics)",
              "merton": f"Merton jumps (λ={mj.intensity:g}, μJ={mj.jump_mean:g}, σJ={mj.jump_std:g})",
              "heston": f"Heston (κ={h.kappa:g}, ξ={h.xi:g}, ρ={h.rho:g})"}
    regimes = {}
    for regime in ("gbm", "merton", "heston"):
        fit, _ = sampler(regime, "refit")
        ev, anti = sampler(regime, "eval")
        refit = fit_lsm(c, fit, cfg.lsm.basis, cfg.lsm.degree)
        policies = {"fitted_q": pol["fitted_q"], "lsm": pol["lsm"], "hold": HoldToExpiry(c),
                    "lsm_refit": refit}
        vals = {k: rollout(p, ev).values for k, p in policies.items()}
        ci = cfg.evaluation.ci_level
        regimes[regime] = {
            "label": labels[regime],
            "value": {k: mc_estimate(v, anti, ci).to_dict() for k, v in vals.items()},
            "regret_vs_refit": {k: mc_estimate(vals["lsm_refit"] - vals[k], anti, ci).to_dict()
                                for k in ("fitted_q", "lsm", "hold")},
            "fitted_q_minus_lsm": mc_estimate(vals["fitted_q"] - vals["lsm"], anti, ci).to_dict(),
        }
    return {"total_vol": vol, "merton_diffusive_vol": vol_d, "heston_long_run_vol": float(np.sqrt(v_bar)),
            "n_paths": n, "regimes": regimes}


def run_evaluate(cfg: Config, real: bool = True) -> dict:
    out = {"synthetic": run_synthetic(cfg), "misspecification": run_misspecification(cfg)}
    update_metrics("evaluate", out, cfg.seed)
    from optstop.figures import plot_degradation, plot_real
    print(f"  figure: {plot_degradation(METRICS_PATH, FIGURES_DIR / 'degradation.png')}")
    if real:
        from optstop.evaluate.distribution import summarise
        from optstop.evaluate.real import evaluate_real
        df = evaluate_real(cfg)
        out["real"] = summarise(df, rng_for(cfg.seed, "real_bootstrap"))
        out["real"]["protocol"] = {"rate": cfg.real.rate, "start": cfg.real.start,
                                   "end": cfg.real.end, "calibration_days": cfg.real.calibration_days,
                                   "vol_grid": cfg.real.vol_grid}
        update_metrics("evaluate", out, cfg.seed)
        print(f"  figure: {plot_real(METRICS_PATH, FIGURES_DIR / 'real_distribution.png')}")
    return out


def summary() -> dict:
    return read_metrics()
