"""Compare Claude-in-Envoy runs with the existing Qwen episode files on the same QASPER sets.

Reads episodes.jsonl files written by scripts/train_qasper_grpo.py (eval mode) and
scripts/run_claude_qasper_eval.py, which share one scorer (`qasper-answer-evidence-v1`),
and reports protocol metrics plus paired bootstrap intervals against a reference system.

Automatic reward is a lexical/span-overlap proxy. It is reported for reproducibility;
the project's earlier findings show it mis-ranks semantically correct answers, so a
blinded semantic review should decide any ranking claim.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["question_id"]: r for r in rows}


def had_error(row: dict) -> bool:
    return any("Traceback (most recent call last)" in (t.get("observation") or "") for t in row["trajectory"])


def metrics(rows: dict[str, dict]) -> dict:
    values = list(rows.values())
    answerable = [r for r in values if r["expected_answerability"] == "sufficient"]
    unanswerable = [r for r in values if r["expected_answerability"] == "insufficient"]
    out = {
        "n": len(values),
        "mean_reward": st.mean(r["reward"] for r in values),
        "answerable_reward": st.mean(r["reward"] for r in answerable) if answerable else None,
        "unanswerable_reward": st.mean(r["reward"] for r in unanswerable) if unanswerable else None,
        "valid_submissions": sum(bool(r.get("valid")) for r in values),
        "abstained_on_unanswerable": sum(bool(r.get("abstained")) for r in unanswerable),
        "n_unanswerable": len(unanswerable),
        "false_refusals": sum(bool(r.get("abstained")) for r in answerable),
        "n_answerable": len(answerable),
        "mean_steps": st.mean(len(r["trajectory"]) for r in values),
        "episodes_with_python_error": sum(had_error(r) for r in values),
        "hit_step_limit": sum(r.get("finish") == "step_limit" for r in values),
        "mean_seconds": st.mean(r["duration_seconds"] for r in values),
    }
    if any("cost_usd" in r for r in values):
        out["total_cost_usd"] = sum(r.get("cost_usd", 0.0) for r in values)
        out["cost_per_question_usd"] = out["total_cost_usd"] / len(values)
        out["mean_output_tokens"] = st.mean(r.get("output_tokens", 0) for r in values)
    return out


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], seed: int = 42, draws: int = 10_000):
    ids = sorted(set(a) & set(b))
    diffs = [b[i]["reward"] - a[i]["reward"] for i in ids]
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return {"n_paired": len(ids), "mean_diff": st.mean(diffs),
            "ci95": [means[int(0.025 * draws)], means[int(0.975 * draws)]],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system", action="append", required=True, metavar="NAME=PATH",
                   help="Repeatable. The first system is the paired-comparison reference.")
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    systems = {}
    for item in args.system:
        name, _, path = item.partition("=")
        systems[name] = load(Path(path))
    reference = next(iter(systems))

    report = {"reference": reference, "systems": {n: metrics(r) for n, r in systems.items()},
              "paired_vs_reference": {n: paired_bootstrap(systems[reference], r)
                                      for n, r in systems.items() if n != reference}}
    for name, m in report["systems"].items():
        cost = f" | ${m['cost_per_question_usd']:.3f}/q" if "cost_per_question_usd" in m else ""
        print(f"{name:22} reward {m['mean_reward']:.3f} (ans {m['answerable_reward']:.3f} / unans "
              f"{m['unanswerable_reward']:.3f}) | valid {m['valid_submissions']}/{m['n']} | "
              f"abstain {m['abstained_on_unanswerable']}/{m['n_unanswerable']} | false-refuse "
              f"{m['false_refusals']}/{m['n_answerable']} | steps {m['mean_steps']:.1f} | "
              f"py-err {m['episodes_with_python_error']} | {m['mean_seconds']:.0f}s{cost}")
    for name, d in report["paired_vs_reference"].items():
        print(f"  {name} vs {reference}: {d['mean_diff']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, "
              f"{d['ci95'][1]:+.3f}]  W/L/T {d['wins']}/{d['losses']}/{d['ties']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
