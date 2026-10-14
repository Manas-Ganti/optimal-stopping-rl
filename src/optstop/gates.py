"""Gate bookkeeping. `results/gates.json` is the code's own record of every gate evaluation.

Gate E is enforced here: real-data evaluation calls `require_gates("A", "B", "C", "D")` and
refuses to run unless each has a recorded PASS.
"""

from __future__ import annotations

import datetime as dt
import json
import zlib
from pathlib import Path

import numpy as np

from optstop.config import RESULTS_DIR

GATES_PATH = RESULTS_DIR / "gates.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"


class GateNotPassed(RuntimeError):
    pass


def today() -> str:
    return dt.date.today().isoformat()


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    """Independent, reproducible stream per purpose (fit paths never reuse evaluation paths)."""
    return np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(purpose.encode())]))


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def record_gate(name: str, passed: bool, details: dict, seed: int) -> None:
    g = _read(GATES_PATH)
    g[name] = {"passed": bool(passed), "date": today(), "seed": seed, **details}
    _write(GATES_PATH, g)
    status = "PASS" if passed else "FAIL"
    print(f"[Gate {name}] {status}")


def update_metrics(section: str, data: dict, seed: int) -> None:
    m = _read(METRICS_PATH)
    m[section] = {"date": today(), "seed": seed, **data}
    m.setdefault("gates", {})
    m["gates"] = _read(GATES_PATH)
    _write(METRICS_PATH, m)


def read_metrics() -> dict:
    return _read(METRICS_PATH)


def require_gates(*names: str, stage: str = "this stage") -> None:
    g = _read(GATES_PATH)
    missing = [n for n in names if not g.get(n, {}).get("passed")]
    if missing:
        raise GateNotPassed(
            f"refusing to run {stage}: gate(s) {', '.join(missing)} not passed "
            f"(see results/gates.json and GATES.md).")
