"""Score identity-blind QASPER semantic judgments and report a paired interval."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


VERDICT_SCORE = {"fail": 0, "partial": 1, "pass": 2}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--baseline-label", default="sft")
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.judgments.read_text().splitlines()
        if line.strip()
    ]
    key = json.loads(args.blind_key.read_text())
    identities = {row["blind_id"]: row for row in key["rows"]}
    expected_systems = {args.baseline_label, args.candidate_label}
    counts: dict[str, Counter] = defaultdict(Counter)
    paired: dict[str, dict[str, int]] = defaultdict(dict)

    for record in records:
        for judgment in record["verdicts"]:
            blind_id = judgment["blind_id"]
            if blind_id not in identities:
                raise ValueError(f"Unknown blind ID: {blind_id}")
            verdict = judgment["verdict"]
            if verdict not in VERDICT_SCORE:
                raise ValueError(f"Invalid verdict: {verdict}")
            identity = identities[blind_id]
            system = identity["system"]
            counts[system][verdict] += 1
            paired[identity["question_id"]][system] = VERDICT_SCORE[verdict]

    if set(counts) != expected_systems:
        raise ValueError(f"Expected systems {expected_systems}, found {set(counts)}")
    if any(set(scores) != expected_systems for scores in paired.values()):
        raise ValueError("Each question must have one judgment for each system")

    summaries = {}
    for system in sorted(expected_systems):
        total = sum(counts[system].values())
        summaries[system] = {
            "questions": total,
            "pass": counts[system]["pass"],
            "partial": counts[system]["partial"],
            "fail": counts[system]["fail"],
            "mean_score_0_to_2": sum(
                VERDICT_SCORE[label] * counts[system][label]
                for label in VERDICT_SCORE
            )
            / total,
        }

    deltas = [
        scores[args.candidate_label] - scores[args.baseline_label]
        for scores in paired.values()
    ]
    rng = random.Random(args.seed)
    bootstrap = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(args.bootstrap_samples)
    ]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    result = {
        "schema_version": "qasper-grpo-blind-assistant-score-v1",
        "systems": summaries,
        "paired": {
            "baseline": args.baseline_label,
            "candidate": args.candidate_label,
            "candidate_mean_delta_0_to_2": sum(deltas) / len(deltas),
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": sum(delta == 0 for delta in deltas),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
            "bootstrap_95_percentile_ci": [
                percentile(bootstrap, 0.025),
                percentile(bootstrap, 0.975),
            ],
        },
        "semantic_gates": {
            "candidate_mean_exceeds_sft": (
                summaries[args.candidate_label]["mean_score_0_to_2"]
                > summaries[args.baseline_label]["mean_score_0_to_2"]
            ),
            "at_least_two_more_wins_than_losses": wins - losses >= 2,
            "at_least_28_passes": summaries[args.candidate_label]["pass"] >= 28,
        },
        "review": {
            "disclosure": (
                "Codex assistant source-based review with system identities hidden until "
                "judgments were recorded and checksummed; not independent human review."
            ),
            "judgment_file": str(args.judgments),
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
