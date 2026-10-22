"""Score local reader reward counterexamples, without inference or training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reader_rl_audit import SCORERS, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.output)
    print(json.dumps({"records": summary["records"], "new_model_calls": 0,
        "training_reward_ready": summary["training_reward_ready"], "scorers": {s: {
            "unacceptable_full_reward": len(summary["scorers"][s]["unacceptable_full_reward"]),
            "adversarial_gate_passed": summary["scorers"][s]["adversarial_gate_passed"]}
            for s in SCORERS}}))


if __name__ == "__main__":
    main()
