"""Compare frozen QASPER evaluations produced by train_qasper_grpo.py."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random


ERROR_MARKERS = ("Traceback", "SyntaxError", "timed out", "ERROR:")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    return rows


def has_execution_error(row: dict) -> bool:
    return any(
        any(marker in str(step.get("observation", "")) for marker in ERROR_MARKERS)
        for step in row.get("trajectory", [])
    )


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    reasons = Counter(row.get("reason") for row in rows)
    execution_errors = sum(has_execution_error(row) for row in rows)
    steps = [len(row.get("trajectory", [])) for row in rows]
    durations = [float(row.get("duration_seconds", 0.0)) for row in rows]
    return {
        "questions": count,
        "mean_reward": sum(float(row.get("reward", 0.0)) for row in rows) / count,
        "submitted": sum(row.get("finish") == "submitted" for row in rows),
        "protocol_valid": sum(bool(row.get("valid")) for row in rows),
        "supported_answer_proxy": sum(
            row.get("reason") == "scored" and float(row.get("reward", 0.0)) > 0
            for row in rows
        ),
        "correct_abstentions": reasons["correct_abstention"],
        "false_refusals": reasons["false_refusal"],
        "execution_error_episodes": execution_errors,
        "mean_steps": sum(steps) / count,
        "max_step_episodes": sum(row.get("finish") == "step_limit" for row in rows),
        "mean_duration_seconds": sum(durations) / count,
        "reasons": dict(reasons),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired(
    baseline: list[dict],
    candidate: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    base = {row["question_id"]: row for row in baseline}
    cand = {row["question_id"]: row for row in candidate}
    if list(base) != list(cand):
        raise ValueError("Evaluation arms do not contain identical ordered question IDs")
    deltas = [float(cand[q]["reward"]) - float(base[q]["reward"]) for q in base]
    rng = random.Random(seed)
    bootstrap = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    ]
    return {
        "mean_reward_delta": sum(deltas) / len(deltas),
        "reward_wins": sum(delta > 1e-12 for delta in deltas),
        "reward_losses": sum(delta < -1e-12 for delta in deltas),
        "reward_ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_95_percentile_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
    }


def arm_order(path: Path) -> tuple[int, int | str]:
    if path.name == "sft":
        return (0, 0)
    if path.name.startswith("rl-") and path.name[3:].isdigit():
        return (1, int(path.name[3:]))
    return (2, path.name)


def analyze(root: Path, *, bootstrap_samples: int = 100_000, seed: int = 42) -> dict:
    arms = sorted(
        (
            path
            for path in root.iterdir()
            if (path / "episodes.jsonl").is_file() and (path / "summary.json").is_file()
        ),
        key=arm_order,
    )
    if not arms or arms[0].name != "sft":
        raise ValueError("Expected an sft baseline arm")
    rows = {arm.name: load_rows(arm / "episodes.jsonl") for arm in arms}
    baseline_rows = rows["sft"]
    baseline = summarize(baseline_rows)
    systems = {"sft": baseline}
    comparisons = {}
    for label, candidate_rows in rows.items():
        if label == "sft":
            continue
        summary = summarize(candidate_rows)
        systems[label] = summary
        comparison = paired(
            baseline_rows,
            candidate_rows,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        comparison.update(
            supported_answer_proxy_delta=(
                summary["supported_answer_proxy"] - baseline["supported_answer_proxy"]
            ),
            false_refusal_delta=summary["false_refusals"] - baseline["false_refusals"],
            protocol_valid_delta=summary["protocol_valid"] - baseline["protocol_valid"],
            execution_error_episode_delta=(
                summary["execution_error_episodes"] - baseline["execution_error_episodes"]
            ),
        )
        comparison["automatic_promotion_proxy"] = {
            "supported_answers_plus_4": comparison["supported_answer_proxy_delta"] >= 4,
            "false_refusals_at_most_sft_plus_1": comparison["false_refusal_delta"] <= 1,
            "execution_errors_no_regression": (
                comparison["execution_error_episode_delta"] <= 0
            ),
            "note": (
                "Nonzero annotation-overlap reward is only a supported-answer proxy; "
                "semantic promotion still requires blinded review."
            ),
        }
        comparisons[label] = comparison
    return {
        "schema_version": "qasper-grpo-paired-eval-v1",
        "systems": systems,
        "comparisons_to_sft": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(
        args.root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
