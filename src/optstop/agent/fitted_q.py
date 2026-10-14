"""Fitted Q-iteration (Ernst, Geurts & Wehenkel 2005) for the stopping problem.

Only the *hold* action needs a learned value: Q(s, exercise) = payoff(s) is known exactly from
the reward. So the agent learns Q_hold(x, tau), the continuation value, by iterating

    Q_{i+1}(s_k) <- fit[ gamma * V_i(s_{k+1}) ],   V_i(s) = max(payoff(s), Q_i(s)),

with gamma = exp(-r dt) and V(s) = payoff(s) at expiry, on transitions from simulated paths.
The policy is greedy: exercise when in the money and payoff >= Q_hold.

How this differs from LSM (and why the comparison means something):
- ONE approximator over the whole (x, tau) plane, shared across time, instead of a separate
  regression per date. Time to expiry is an input, not an index.
- Fitted on all visited states, not only in-the-money ones.
- Bootstraps from its own value estimate V_i(s') (Q-learning style) instead of propagating the
  realised path cash flow (LSM style). This is the classic RL ingredient, and the classic
  source of upward bias: max over a noisy estimate is biased upward.

Regressors:
- 'rbf' (default): linear FQI on normalised (partition-of-unity) Gaussian radial basis
  functions over (log x, sqrt tau), plus the immediate payoff as a feature. Centres span the
  full training range; normalisation makes the fit extrapolate flat past the outer centres
  instead of decaying to zero (the cause of the original Gate C failure, see GATES.md). The design matrix never changes between
  iterations, so the regularised Gram matrix is factorised once and every iteration is a single
  triangular solve. Fast, deterministic, smooth.
  Optionally (default on) the regression includes martingale control variates (see
  `_cv_features`), which cut the noise in the bootstrapped targets without biasing Q.
- 'extra_trees': the original FQI regressor. Non-parametric, piecewise-constant, slow; produces
  a visibly noisier boundary.

Assumes: the training paths cover the region of state space where the boundary lives; outside
the RBF support the continuation estimate decays to the constant term and is meaningless. Fails
when the boundary sits where training paths rarely go (e.g. far out-of-the-money spots).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from optstop.config import FittedQCfg
from optstop.contracts import Contract

_CHUNK = 100_000


@dataclass
class RBFFeatures:
    u_centers: np.ndarray   # centres in u = log x
    w_centers: np.ndarray   # centres in w = sqrt(tau)
    u_width: float
    w_width: float
    kind: str               # contract kind, for the payoff feature
    normalized: bool = True # partition-of-unity RBFs (extrapolate flat, not to zero)

    @classmethod
    def from_data(cls, x: np.ndarray, expiry: float, n_x: int, n_tau: int, width: float,
                  kind: str, normalized: bool = True) -> "RBFFeatures":
        u = np.log(x)
        lo, hi = (u.min(), u.max()) if normalized else np.quantile(u, [0.002, 0.998])
        uc = np.linspace(lo, hi, n_x)
        wc = np.linspace(0.0, np.sqrt(expiry), n_tau)
        return cls(uc, wc, width * (uc[1] - uc[0]), width * (wc[1] - wc[0]), kind, normalized)

    def __call__(self, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        tau = np.broadcast_to(np.asarray(tau, dtype=float), x.shape).ravel()
        u = np.log(x)[:, None]
        w = np.sqrt(np.maximum(tau, 0.0))[:, None]
        gu = np.exp(-0.5 * ((u - self.u_centers[None, :]) / self.u_width) ** 2)
        gw = np.exp(-0.5 * ((w - self.w_centers[None, :]) / self.w_width) ** 2)
        rbf = (gu[:, :, None] * gw[:, None, :]).reshape(len(x), -1)
        pay = np.maximum(1.0 - x, 0.0) if self.kind == "put" else np.maximum(x - 1.0, 0.0)
        if self.normalized:
            # Partition of unity: rows sum to one, so the constant is already in the span.
            # Past the outermost centre the features tend to the nearest centre's one-hot
            # instead of all decaying to zero, so the fit extrapolates flat.
            rbf = rbf / np.maximum(rbf.sum(axis=1, keepdims=True), 1e-300)
            return np.column_stack([pay, rbf])
        return np.column_stack([np.ones(len(x)), pay, rbf])

    @property
    def n_features(self) -> int:
        nu, nw = len(self.u_centers), len(self.w_centers)
        return nu * nw + (1 if self.normalized else 2)


@dataclass
class FittedQPolicy:
    """Greedy policy w.r.t. a learned continuation value. Implements `optstop.policy.Policy`."""

    contract: Contract
    regressor: str
    features: RBFFeatures | None = None
    beta: np.ndarray | None = None
    model: object | None = None
    history: list[float] = field(default_factory=list)  # max |Q_{i+1} - Q_i| per iteration

    def q_hold(self, x: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.regressor == "rbf":
            return (self.features(x, tau) @ self.beta).reshape(x.shape)
        tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), x.shape)
        X = np.column_stack([np.log(x.ravel()), np.sqrt(tau_arr.ravel())])
        return self.model.predict(X).reshape(x.shape)

    def exercise(self, x: np.ndarray, k: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        pay = self.contract.payoff_x(x)
        if k == self.contract.steps:
            return pay > 0
        tau = self.contract.taus()[k]
        return (pay > 0) & (pay >= self.q_hold(x, tau))


def _cv_features(s_x: np.ndarray, n_x: np.ndarray, gamma: float, feats: RBFFeatures,
                 n_centers: int) -> np.ndarray:
    """Martingale control variates Z * psi_j(x), Z = gamma * x' - x.

    Under any risk-neutral simulator the discounted underlying is a martingale, so
    E[Z | s] = 0 and E[Z psi_j(x) | s] = 0 for any function psi_j of the current state. Adding
    these columns to the regression leaves the conditional-mean (Q) coefficients unbiased while
    removing the part of the target's variance explained by the one-step price move, with a
    state-dependent (delta-like) coefficient through coarse RBFs psi_j in log-moneyness.
    Assumes the training paths are risk-neutral; it would bias Q on real-world-measure paths.
    """
    centers = np.linspace(feats.u_centers[0], feats.u_centers[-1], n_centers)
    width = centers[1] - centers[0]
    psi = np.exp(-0.5 * ((np.log(s_x)[:, None] - centers[None, :]) / width) ** 2)
    z = gamma * n_x - s_x
    return (psi * z[:, None]).astype(np.float32)


def _transitions(contract: Contract, paths: np.ndarray):
    x = paths / contract.strike
    n, m = x.shape
    taus = contract.taus()
    s_x = x[:, :-1].ravel()
    s_tau = np.tile(taus[:-1], n)
    n_x = x[:, 1:].ravel()
    n_tau = np.tile(taus[1:], n)
    terminal = np.tile(np.arange(1, m) == contract.steps, n)
    return s_x, s_tau, n_x, n_tau, terminal


def train_fitted_q(contract: Contract, paths: np.ndarray, cfg: FittedQCfg,
                   seed: int = 0, verbose: bool = False) -> FittedQPolicy:
    """Learn Q_hold from simulated paths (n_paths, steps + 1) in currency units."""
    gamma = contract.step_discount
    s_x, s_tau, n_x, n_tau, terminal = _transitions(contract, paths)
    pay_next = contract.payoff_x(n_x)
    n_iter = cfg.n_iterations or contract.steps
    history: list[float] = []

    if cfg.regressor == "rbf":
        # Every non-terminal next state s_{k+1} is also a current state of the same path, so
        # Q at next states is read off Q evaluated on current states: one design matrix only.
        n_paths, steps = paths.shape[0], contract.steps
        feats = RBFFeatures.from_data(paths / contract.strike, contract.expiry, cfg.rbf_centers_x,
                                      cfg.rbf_centers_tau, cfg.rbf_width, contract.kind,
                                      cfg.rbf_normalized)
        n_q = feats.n_features
        cv = _cv_features(s_x, n_x, gamma, feats, cfg.cv_centers_x) if cfg.control_variate \
            else None
        n_cv = 0 if cv is None else cv.shape[1]
        Phi = np.empty((len(s_x), n_q + n_cv), dtype=np.float32)
        G = np.zeros((Phi.shape[1], Phi.shape[1]))
        for a in range(0, len(s_x), _CHUNK):
            block = feats(s_x[a:a + _CHUNK], s_tau[a:a + _CHUNK])
            if cv is not None:
                block = np.hstack([block, cv[a:a + _CHUNK]])
            Phi[a:a + _CHUNK] = block
            G += block.T @ block
        G[np.diag_indices_from(G)] += cfg.ridge * len(Phi)
        chol = cho_factor(G)
        beta = np.zeros(Phi.shape[1])
        pay_next_2d = pay_next.reshape(n_paths, steps)
        q_next = np.zeros((n_paths, steps))
        for i in range(n_iter):
            v_next = np.maximum(pay_next_2d, q_next)
            v_next[:, -1] = pay_next_2d[:, -1]  # expiry: payoff only
            y = (gamma * v_next).ravel().astype(np.float32)
            beta = cho_solve(chol, (Phi.T @ y).astype(float))
            # Control-variate columns have zero conditional mean: they soak up target noise
            # during the fit and are dropped from the Q estimate.
            q_cur = (Phi[:, :n_q] @ beta[:n_q].astype(np.float32)).reshape(n_paths, steps)
            q_cur = q_cur.astype(float)
            q_new = np.zeros_like(q_next)
            q_new[:, :-1] = q_cur[:, 1:]
            history.append(float(np.max(np.abs(q_new - q_next))))
            q_next = q_new
            if verbose:
                print(f"  FQI iter {i + 1:3d}/{n_iter}  max|dQ| = {history[-1]:.2e}")
        return FittedQPolicy(contract, "rbf", features=feats, beta=beta[:n_q], history=history)

    if cfg.regressor == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        X = np.column_stack([np.log(s_x), np.sqrt(s_tau)])
        Xn = np.column_stack([np.log(n_x), np.sqrt(n_tau)])
        q_next = np.zeros(len(n_x))
        model = None
        for i in range(n_iter):
            v_next = np.where(terminal, pay_next, np.maximum(pay_next, q_next))
            model = ExtraTreesRegressor(n_estimators=cfg.extra_trees.n_estimators,
                                        min_samples_leaf=cfg.extra_trees.min_samples_leaf,
                                        random_state=seed + i, n_jobs=-1)
            model.fit(X, gamma * v_next)
            q_new = model.predict(Xn)
            history.append(float(np.max(np.abs(q_new - q_next))))
            q_next = q_new
            if verbose:
                print(f"  FQI iter {i + 1:3d}/{n_iter}  max|dQ| = {history[-1]:.2e}")
        return FittedQPolicy(contract, "extra_trees", model=model, history=history)

    raise ValueError(f"unknown regressor {cfg.regressor!r}")
