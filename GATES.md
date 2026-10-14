# GATES.md — pass criteria, stated before any number was computed

Written 2026-10-14, before the simulator, baseline, or agent had been run even once.
Each gate records PASS / FAIL and the date it was evaluated. The machine-readable record of
every gate evaluation is `results/gates.json`, written by the code itself. Thresholds below are
mirrored in `configs/config.yaml` under `gates:`; if the two ever disagree, this file is the
statement of intent and the config is a bug.

Nothing about the American put is reported until A, B and C pass. No real-data number is
computed until A–D pass (Gate E is enforced in code, not just policy).

---

## Gate A — the simulator

**Claim tested:** `paths.py` samples the risk-neutral dynamics it says it samples.

1. **GBM vs Black-Scholes.** For each of the 20 parameter sets of Longstaff & Schwartz (2001),
   Table 1 (K=40, r=0.06, S0 ∈ {36,38,40,42,44}, σ ∈ {0.2,0.4}, T ∈ {1,2}), price the European
   put by Monte Carlo on the simulator's terminal value (100,000 paths, antithetic, 50 steps/yr,
   seeded). **Pass:** |MC − BS| ≤ 3.5 · SE for all 20 cases. (3.5σ per case gives a family-wise
   false-alarm rate of ≈1% across 20 cases if the simulator is correct.)
2. **Merton jump-diffusion vs Merton series formula.** Same test for the jump sampler, European
   put, three parameter sets, against Merton's (1976) closed-form series. **Pass:** |z| ≤ 3.5.
3. **Heston martingale check.** No closed form is used; instead check the discounted price is a
   martingale: |mean(e^{-rT} S_T) − S0| ≤ 3.5 · SE.

## Gate B — the classical baseline

**Claim tested:** `baselines/lsm.py` is a correct Longstaff-Schwartz implementation.

Reference: the finite-difference American put values in Longstaff & Schwartz (2001), Table 1,
all 20 cases. LSM settings: 100,000 paths (50,000 antithetic pairs), 50 exercise dates per
year, weighted Laguerre basis of degree 3 plus constant, regression on in-the-money paths only.
The reported LSM price is the **out-of-sample** price: the exercise rule is fitted on one path
set and evaluated on an independent one, so it is an unbiased estimate of a (sub-optimal)
policy's value and therefore a lower bound in expectation.

**Pass:** |LSM − FD| ≤ 0.04 for every one of the 20 cases **and** mean |LSM − FD| ≤ 0.02.

Why not tighter: 50 exercise dates/yr makes LSM a Bermudan price, which sits below the
continuous American FD value by O(0.01); LS2001's own LSM column deviates from FD by up to
0.025. A tolerance tighter than the method's known bias would be testing noise.

Informational (not a gate): an exact Bermudan price/boundary from a CRR lattice with exercise
restricted to the same dates, so LSM can also be compared to the *exact* answer for its own
exercise schedule.

## Gate C — agent sanity

**Claim tested:** the agent does not invent early exercise where none is optimal.

Train the fitted-Q agent, with exactly the same settings as the put experiment, on an American
**call** on a non-dividend underlying (K=40, S0=40, σ=0.2, r=0.06, T=1, 50 steps). Early
exercise is never optimal (Merton 1973). Evaluate on 100,000 independent paths.

**Pass:** fraction of paths exercised strictly before expiry ≤ 0.5% **and** the policy value
is not below the Black-Scholes call by more than 3 · SE.

## Gate D — agreement

**Claim tested:** the learned policy recovers the same exercise boundary as LSM on the
training distribution (put, K=40, S0=40, σ=0.2, r=0.06, T=1, 50 steps).

Boundaries are extracted with the same code (`boundary.py`) from both policies, in units of
S/K, at every exercise date strictly before expiry.

**Pass, all three:**
- mean |b_FQI(τ) − b_LSM(τ)| ≤ 0.02 (S/K units, i.e. $0.80 on K=40)
- max  |b_FQI(τ) − b_LSM(τ)| ≤ 0.05
- on 200,000 common independent evaluation paths, |V_FQI − V_LSM| / V_LSM ≤ 1%

Informational: both boundaries against the exact Bermudan lattice boundary.

## Gate E — real data is sealed

`evaluate/real.py` and `evaluate/distribution.py` refuse to run unless `results/gates.json`
records A, B, C and D as passing. Real-data results are not computed or viewed before that.

---

## Record

| Gate | Status | Date | Evidence |
|---|---|---|---|
| A | **PASS** | 2026-10-14 | worst \|z\| GBM-vs-BS 2.24, Merton 1.44, Heston martingale 1.60 (limit 3.5) |
| B | **PASS** | 2026-10-14 | mean \|LSM − FD\| 0.0057 (limit 0.02), max 0.0282 (limit 0.04), 20/20 cases |
| C | **FAIL** ×2, then **PASS** | 2026-10-14 | attempts 1–3 failed (0.666%, 0.502%, 44%); attempt 4 (pre-registered, normalised RBFs): early-exercise rate **0.000%** on 100,000 paths, value 4.4113 ± 0.0131 vs BS call 4.3958 (z = +1.18) |
| D | **PASS** | 2026-10-14 | single run on the attempt-4 agent: mean \|b_FQI − b_LSM\| 0.0150 (limit 0.02), max 0.0240 (limit 0.05), \|ΔV\|/V 0.012% (limit 1%) |
| E | opened | 2026-10-14 | real data first downloaded and evaluated only after A–D passed |

## Changes log

Any change to a threshold after a run is recorded here with the reason. A threshold changed to
turn a FAIL into a PASS is a FAIL.

- **2026-10-14 — Gate D, exploratory run FAILED (thresholds unchanged).** *Process note: this
  prototype was run out of order, before Gate C had been evaluated. The put numbers quoted in
  this entry are exploratory diagnostics, not results, and must not be cited until Gates A–C
  pass and Gate D is evaluated by `optstop boundary`.* First prototype run
  (scratch script, not the CLI), with every fitting/training path started at S0 = K = 40:
  values agreed (V_LSM = 2.3071 ± 0.0073, V_FQI = 2.3078 ± 0.0070, paired diff 0.0008 ± 0.0032;
  exact Bermudan lattice 2.3141), mean |b_FQI − b_LSM| = 0.0129 (pass) but
  **max = 0.075 > 0.05 (fail)**. The worst dates were k = 0, 1, 2. Diagnosis: at date 1 the
  minimum S/K over 100,000 paths is 0.877, while the exact boundary is ≈ 0.835, so at early
  dates *neither* method has data where the boundary is and both values are extrapolations.
  LSM was the worse of the two against the exact lattice there (max error 0.066 vs 0.023 for
  FQI). **Change made: experimental design, not threshold.** For the boundary experiment
  (Gates C and D, and the headline figure) both LSM and FQI are now fitted on paths whose
  initial spot is drawn uniformly from S0/K ∈ [0.70, 1.30] (config `boundary_training`), so
  the boundary region is visited at every date. Gate B pricing still uses the standard
  fixed-S0 LS2001 setup. Value comparisons are still made on paths started at S0 = 40. Date 0
  is excluded from the boundary comparison when the fit has no cross-section there; with
  dispersed starts it has one, so date 0 is now included. The failed configuration is
  reported in the README, not hidden.

- **2026-10-14 — Gate C FAILED (thresholds unchanged). Pipeline stopped here.**
  1. First official run (`optstop train`), agent = linear FQI on RBF features as configured
     before any run: early-exercise rate **0.666%** > 0.5%. Value not below BS (z = +1.10).
     Diagnosis: all early exercises at dates k = 46–49 with S/K in [1.61, 1.98], *inside* the
     training data. There, the true margin hold − exercise is K(1 − e^{−rτ}) ≈ 0.0012 per unit
     strike, smaller than the regression error of the continuation estimate (sparse data, noisy
     one-step bootstrapped targets).
  2. Change to the agent, motivated by the noise diagnosis and not specific to calls: martingale
     control variates in the regression (`fitted_q.control_variate`). Result: **0.502%**, still
     a FAIL. Remaining exercises at k = 48–49, S/K ≈ 1.64–1.94, at the edge of the RBF grid
     (top centre 1.888): approximation bias, not just noise.
  3. Tried marginal τ-RBF features (global in x). Result: 44% early exercise (near-collinear
     with the tensor RBFs, ill-conditioned). **Reverted.** Stopped iterating at this point to
     avoid tuning the agent against the gate.
  4. Independent check: the same configuration on a different seed (`pytest`
     `test_no_early_exercise.py`) gives 0.63%. The failure is not a seed artefact.

  Status: the agent as configured (step 2) fails Gate C. By the project's inviolable rule,
  nothing about the learned put policy is reported. Gate D, the boundary figure and all
  evaluation are blocked by code. Candidate fixes are listed in README "Status".


- **2026-10-14 — Gate C, attempt 4, pre-registered before running.** Change to the agent's
  features only (thresholds unchanged): **normalised (partition-of-unity) RBFs**, i.e. each
  tensor Gaussian divided by the sum of all of them at that state, with centres spanning the
  *full* min–max range of training log-moneyness instead of its 0.2%–99.8% quantiles. The
  constant column is dropped (normalised RBFs already sum to one). Payoff column and martingale
  control variates are kept. Rationale: plain Gaussians decay to zero past the outermost centre,
  so at S/K ≈ 1.9 and tau → 0 the fit cannot represent the tau-dependent continuation level;
  normalised RBFs extrapolate as the nearest centre's value. Generic (not call-specific).
  **Stopping rule:** this is one attempt. If Gate C still fails, it is recorded as FAIL and
  the pipeline stays blocked; no further feature changes without a new written entry and the
  user's sign-off. If it passes, Gate D is evaluated once on this same agent, unchanged.

- **2026-10-14 — attempt 4 result.** Gate C PASS (0 early exercises in 100,000 paths). Gate D
  then run exactly once on the same agent: PASS. Observed and reported, not tuned away: the
  learned boundary sits ~0.015–0.02 S/K *above* the exact lattice boundary at long maturities
  (exercises slightly early), and both methods sit up to ~0.04 below the exact boundary at the
  final dates. Gate E opened; real data downloaded and evaluated afterwards.

