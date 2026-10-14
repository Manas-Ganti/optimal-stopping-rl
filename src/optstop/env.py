"""Gymnasium environment: one episode is one option lifetime.

Observation: [x, tau] with x = S/K and tau = time to expiry (years), float32.
Actions: 0 = hold, 1 = exercise.
Reward: on exercise, the payoff discounted to t_0, i.e. exp(-r t_k) * max(K - S, 0) / K
(per unit strike); zero otherwise. With undiscounted episode return this equals the discounted
payoff of the stopping rule, so the optimal policy maximises exactly the option value. At expiry
the episode ends and an in-the-money option pays out regardless of the action.

The environment is a thin interface over `paths.py` + `contracts.py`. The fitted-Q agent learns
offline from batches of simulated paths (equivalent to transitions logged under an always-hold
behaviour policy, which visits every state of every path). The env exists so any Gymnasium
agent (e.g. DQN) can be plugged in and so learned policies can be rolled out step by step.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from optstop.contracts import Contract
from optstop.paths import gbm_paths

PathSampler = Callable[[np.random.Generator], np.ndarray]  # returns one path, shape (steps+1,)


class OptionStoppingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, contract: Contract, spot: float, vol: float,
                 sampler: PathSampler | None = None):
        super().__init__()
        self.contract = contract
        self.spot = spot
        self.vol = vol
        self._sampler = sampler or (lambda rng: gbm_paths(
            spot, contract.rate, vol, contract.expiry, contract.steps, 1, rng,
            antithetic=False)[0])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, contract.expiry], dtype=np.float32), dtype=np.float32)
        self._path: np.ndarray | None = None
        self._k = 0

    def _obs(self) -> np.ndarray:
        x = self._path[self._k] / self.contract.strike
        return np.array([x, self.contract.expiry - self._k * self.contract.dt], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._path = self._sampler(self.np_random) if not options or "path" not in options \
            else np.asarray(options["path"], dtype=float)
        self._k = 0
        return self._obs(), {"k": 0}

    def step(self, action: int):
        c = self.contract
        x = self._path[self._k] / c.strike
        pay = float(c.payoff_x(x))
        disc = float(np.exp(-c.rate * self._k * c.dt))
        if (action == 1 and pay > 0) or self._k == c.steps:
            return self._obs(), disc * pay, True, False, {"k": self._k, "exercised": pay > 0}
        if action == 1:  # exercising worthless: ends the episode with nothing
            return self._obs(), 0.0, True, False, {"k": self._k, "exercised": False}
        self._k += 1
        return self._obs(), 0.0, False, False, {"k": self._k}
