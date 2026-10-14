# CLAUDE.md

## Project: Learned Optimal Stopping for American Options

An RL agent decides, at each step, whether to exercise an American option or hold it. The
classical answer is known: Longstaff-Schwartz computes the exercise boundary by backward
induction, and for some cases a closed form exists. **The deliverable is not a better policy.
It is a learned policy verified against a known-correct answer, and then an honest measurement
of what happens when the world stops matching the assumptions it was trained on.**

The structure that makes this project worth building: there is a regime where the right answer
is provably known, and a regime where it is not. Prove the agent is correct in the first before
believing anything it says about the second.

---

## HOW TO WORK ON THIS PROJECT (read first)

- **Scaffold the whole codebase in this session**, runnable end to end.
- Build in this order: simulator -> classical baselines -> gates written down -> RL agent ->
  boundary comparison -> real-data evaluation -> writeup. **The baselines come before the
  agent.** If the agent is built first, the baseline quietly gets tuned until the agent wins.
- **Never report a number that was not produced by a real run.** No illustrative figures in
  the README.
- **Write GATES.md before running anything.** State the pass criteria, then run.

---

## THE INVIOLABLE RULE

**No learned policy is believed until it has reproduced an analytically known answer in a
regime where one exists.**

This project has three such regimes, and all three are cheap:

1. A European option priced by Monte Carlo on your simulator must match the Black-Scholes
   closed form within Monte Carlo error. This validates the **simulator**.
2. Your Longstaff-Schwartz implementation must reproduce published benchmark American put
   prices within tolerance. This validates the **baseline**.
3. An American call on a non-dividend-paying underlying should never be exercised early. Run
   the agent on one. If it exercises, the **agent** is broken, and you know it before touching
   the interesting case.

Nothing about the put problem gets reported until all three pass.

---

## The decision problem

The agent holds one American put: the right to sell the underlying at strike `K` at any time
before expiry `T`.

At each step it observes the state and chooses **exercise** or **hold**.

- Exercise: receive `max(K - S, 0)` and the episode ends.
- Hold: the episode continues to the next step.

The hard quantity is the **continuation value**, which is unobservable and depends on where
the price might go and how much time remains. The point at which continuation value falls
below immediate payoff traces the **exercise boundary**, and recovering that curve is the
whole task.

**State:** underlying price (normalised by strike, `S/K`), time to expiry. Keep it 2D for the
first version. A 2D state means the learned boundary can be plotted directly, which is the
project's headline figure.

**Reward:** the discounted payoff on exercise, zero otherwise. Mechanically computable from
the path. No judge, no shaping, no auxiliary terms.

---

## Modules

### 1. `paths.py` — the environment's dynamics
- Geometric Brownian motion path generator, seeded and vectorised
- Later: Merton jump-diffusion and Heston, used **only** for evaluation, never for training
  in the main experiment
- Calibration helper: estimate drift and volatility from a real price series

Nothing is learned here. This is a sampler.

### 2. `contracts.py`
- Payoff functions for put and call
- Strike, expiry, step count, discount rate as config

### 3. `baselines/`
**`lsm.py` — Longstaff-Schwartz least-squares Monte Carlo.** Regression-based backward
induction. Basis functions configurable (polynomial, Laguerre). This is the thing the agent
is compared against, and it is also, structurally, fitted Q-iteration with a linear function
approximator, so the two methods are solving the same Bellman equation by different routes.
Say that in the README; it reframes the comparison from "my model beat theirs" into "two
methods agree on the same fixed point," which is a stronger and more honest claim.

**`analytic.py`** — Black-Scholes European price, and the perpetual American put closed-form
boundary. Used for the validation gates, not as competitors.

### 4. `env.py` — the RL environment
- Gymnasium interface over `paths.py` and `contracts.py`
- One episode is one option lifetime
- Binary action space

### 5. `agent/`
- DQN over the 2D state, or fitted Q-iteration. **Start with fitted Q-iteration**: it is closer
  to LSM, easier to debug, and if it already matches the boundary you may not need DQN at all.
- Add DQN as the second method only if the first works.

### 6. `boundary.py`
- Extract the exercise boundary from **any** policy, learned or classical, as a function of
  time to expiry
- This is the comparison surface and the figure generator. Write it early; it is how you debug
  everything else.

### 7. `evaluate/`
- **`synthetic.py`** — policy value across many simulated paths, with confidence intervals
- **`real.py`** — run the learned policy and LSM on actual historical price paths
- **`distribution.py`** — evaluate across many underlyings and many start dates, because one
  historical path is a sample of size one

---

## Real data

Daily OHLC from a free source (`yfinance`, Stooq). Used for exactly three things:

1. **Calibrating** the simulator's drift and volatility
2. **Out-of-sample evaluation** of a policy trained on synthetic paths
3. **Legibility** in the writeup, where a real price series with real dates reads better than
   an abstraction

**Never train on real paths.** One stock over twenty years gives roughly a hundred
non-overlapping option lifetimes. Overlapping windows share almost all their days, so they are
the same episode counted repeatedly. A hundred effectively independent episodes produces
memorisation, not a policy.

**Never use real option prices.** Market prices already embed the early-exercise premium and
matching them requires an implied volatility surface. Use real underlying prices with a
contract you define yourself.

---

## Pre-stated gates

Write into `GATES.md` **before running anything**, with pass or fail and the date recorded.

- **Gate A (simulator)** — Monte Carlo European put price matches Black-Scholes within Monte
  Carlo error at a stated tolerance.
- **Gate B (baseline)** — LSM reproduces published benchmark American put prices within
  tolerance.
- **Gate C (agent sanity)** — the agent trained on an American call with no dividend learns
  never to exercise early.
- **Gate D (agreement)** — the learned exercise boundary agrees with LSM's within a stated
  tolerance on the training distribution.
- **Gate E** — no real-data result is computed or looked at until A through D pass.

Gate E matters. Looking at real-data performance early is how you end up unconsciously
selecting the configuration that happened to do well on history.

---

## The findings to target

1. **Boundary agreement.** The learned policy recovers the LSM boundary. This is the
   validation result and it is the one that makes everything else credible.
2. **Degradation under misspecification, measured relatively.** Train on GBM, evaluate on
   jump-diffusion and on real paths. Report **how much the learned policy degrades relative to
   how much LSM degrades.** If the agent wins on its training distribution and loses on real
   data, it overfit the generator. That is the real finding and it is worth more than a win.
3. **The size-one caveat, handled.** Report a distribution of outcomes across many underlyings
   and start dates, never a single historical P&L number.

A negative result here is a result. "Learned stopping matches the classical boundary on the
training distribution and degrades faster than it under fat tails" is a complete, useful,
honest conclusion.

---

## Repository Structure

```
optimal-stopping-rl/
|- README.md                   # what it is, REAL results, the boundary figure, limitations
|- GATES.md                    # gates stated before any numbers
|- pyproject.toml
|- configs/config.yaml         # strike, expiry, steps, rate, vol, seeds, tolerances
|- src/optstop/
|  |- paths.py                 # GBM, jump-diffusion, Heston, calibration
|  |- contracts.py             # payoffs
|  |- baselines/
|  |  |- lsm.py                # Longstaff-Schwartz
|  |  |- analytic.py           # Black-Scholes, perpetual American put
|  |- env.py                   # Gymnasium environment
|  |- agent/
|  |  |- fitted_q.py           # build this first
|  |  |- dqn.py                # only if fitted Q works
|  |- boundary.py              # boundary extraction + figures
|  |- evaluate/
|  |  |- synthetic.py
|  |  |- real.py
|  |  |- distribution.py
|  |- data/fetch.py            # yfinance / Stooq
|  |- cli.py                   # `optstop baseline`, `train`, `boundary`, `evaluate`
|- tests/
|  |- test_paths.py            # European MC price vs Black-Scholes
|  |- test_lsm.py              # vs published benchmark prices
|  |- test_analytic.py         # closed forms vs hand-worked values
|  |- test_boundary.py         # extraction is correct on a known policy
|  |- test_no_early_exercise.py# call policy never exercises early
|- results/                    # metrics.json + figures from real runs
|- figures/
```

---

## Conventions

- **Seeded and reproducible.** Seed in config, recorded in `results/metrics.json`.
- Config-driven. No hardcoded strikes, tolerances, or basis functions.
- Every figure generated by code from `results/metrics.json`. Never drawn by hand, never with
  placeholder data.
- Monte Carlo numbers always reported with confidence intervals. A point estimate from a
  stochastic simulation is not a result.
- Docstrings state what each method assumes and when it fails.
- Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `gymnasium`, `scikit-learn`,
  `pydantic`, `pytest`, and `yfinance` for the real-data path. Add `torch` only if DQN is
  actually built. Ask before adding more.

---

## Definition of Done

1. `optstop baseline` produces LSM prices and boundaries, with Gates A and B passing.
2. `optstop train` produces a learned policy, with Gate C passing.
3. `optstop boundary` emits the headline figure: learned and LSM exercise boundaries on the
   same axes, price against time to expiry.
4. `optstop evaluate` gives synthetic performance with confidence intervals, and real-data
   performance as a distribution across underlyings and start dates.
5. `results/metrics.json` carries the real numbers with seed and date.
6. README leads with the boundary figure and the agreement result, then the degradation
   finding, then a limitations section naming the real problems.
7. `pytest` passes, including every validation test.

---

## Scope and cut line

Target three weeks part time. If it slips:

- **Cut first:** DQN (fitted Q-iteration alone is sufficient), Heston (jump-diffusion alone
  demonstrates misspecification), the multi-underlying distribution study.
- **Never cut:** any gate, the boundary figure, the confidence intervals.

Stages 1 to 3 standing alone are a complete project: a verified simulator, a verified
classical baseline, and a learned policy proven to recover a known-correct answer. That is
finished work even without the degradation study.

---

## Shared agent knowledge base

This machine runs an MCP knowledge base ("agent call centre") that other agents write to. The
tools are:

`mcp__callcentre__search`, `mcp__callcentre__get`, `mcp__callcentre__submit`,
`mcp__callcentre__report_outcome`, `mcp__callcentre__open_issue`, `mcp__callcentre__claim_issue`,
`mcp__callcentre__log_attempt`, `mcp__callcentre__resolve_issue`

Use them:

- **`search` first, before spending time on any error** — Monte Carlo variance problems,
  LSM regression instability, Gymnasium API changes, DQN non-convergence, data-fetch failures.
- **`get`** to pull the full write-up of a matching entry.
- **`submit`** when you solve something non-obvious that cost real time, particularly LSM basis
  function choices and anything about discounting or step conventions, which are easy to get
  subtly wrong.
- **`report_outcome`** after trying a suggested fix, whether or not it worked.
- **`open_issue`** for a blocker you cannot resolve, with what you tried.
- **`claim_issue`** before working an open issue so two agents do not duplicate effort.
- **`log_attempt`** while working a claimed issue, so the trail survives if you stop.
- **`resolve_issue`** when it is fixed, with the resolution.

---

## Why this project exists (keep this framing in the README)

Most reinforcement learning results in finance cannot be checked, because nobody knows the
right answer. Optimal stopping is different: the classical solution is computable and in
limiting cases exact. That makes it the rare setting where you can prove a learned policy is
correct before asking it anything hard.

This project does that, and then measures the thing that actually matters: how fast the
verified policy stops being correct once the world stops matching the model it was trained on.