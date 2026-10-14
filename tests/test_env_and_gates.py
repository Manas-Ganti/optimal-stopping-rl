"""The Gymnasium env agrees with the vectorised evaluator; Gate E refuses without passes."""

import numpy as np
import pytest

from optstop import gates
from optstop.env import OptionStoppingEnv
from optstop.evaluate.synthetic import rollout
from optstop.paths import gbm_paths
from optstop.policy import ThresholdPolicy


def test_env_rollout_matches_vectorised(put, rng):
    pol = ThresholdPolicy(put, np.full(put.steps + 1, 0.9))
    P = gbm_paths(40, 0.06, 0.3, 1.0, 50, 20, rng)
    vec = rollout(pol, P).values / put.strike
    env = OptionStoppingEnv(put, 40.0, 0.3)
    for i in range(len(P)):
        obs, _ = env.reset(options={"path": P[i]})
        total, done, k = 0.0, False, 0
        while not done:
            a = int(pol.exercise(np.array([obs[0]], dtype=float), k)[0] and obs[0] < 1)
            obs, r, done, _, info = env.step(a)
            total += r
            k += 1
        assert total == pytest.approx(vec[i], abs=1e-6)


def test_env_api(put):
    env = OptionStoppingEnv(put, 40.0, 0.2)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs) and obs[1] == pytest.approx(1.0)


def test_require_gates_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "GATES_PATH", tmp_path / "gates.json")
    with pytest.raises(gates.GateNotPassed):
        gates.require_gates("A", "B", "C", "D")
    gates.record_gate("A", True, {}, seed=0)
    gates.record_gate("B", False, {}, seed=0)
    with pytest.raises(gates.GateNotPassed, match="B"):
        gates.require_gates("A", "B")
    gates.require_gates("A")
