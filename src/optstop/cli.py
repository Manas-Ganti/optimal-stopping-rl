"""optstop baseline | train | boundary | evaluate | all

Stages must run in order; each refuses to start until the gates before it have passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from optstop.config import load_config
from optstop.gates import GateNotPassed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="optstop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="Gates A (simulator) and B (LSM vs published prices)")
    sub.add_parser("train", help="Gate C (no early call exercise); train put policies")
    sub.add_parser("boundary", help="Gate D (boundary agreement); headline figure")
    ev = sub.add_parser("evaluate", help="synthetic + misspecification + real data (Gate E)")
    ev.add_argument("--no-real", action="store_true", help="skip the real-data evaluation")
    allp = sub.add_parser("all", help="every stage in order, stopping at the first failed gate")
    allp.add_argument("--no-real", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    from optstop import pipeline as p

    stages = {"baseline": lambda: p.run_baseline(cfg), "train": lambda: p.run_train(cfg),
              "boundary": lambda: p.run_boundary(cfg),
              "evaluate": lambda: p.run_evaluate(cfg, real=not getattr(args, "no_real", False))}
    order = list(stages) if args.cmd == "all" else [args.cmd]
    try:
        for name in order:
            t0 = time.time()
            print(f"== optstop {name}")
            stages[name]()
            print(f"   done in {time.time() - t0:.1f}s")
    except GateNotPassed as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        return 2
    gates = p.summary().get("gates", {})
    print("gates:", json.dumps({k: v["passed"] for k, v in gates.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
