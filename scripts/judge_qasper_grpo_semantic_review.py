"""Use Claude Sonnet to judge a blinded QASPER multi-system review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv


MODEL = "claude-sonnet-5"
VERDICT_SCORE = {"fail": 0, "partial": 1, "pass": 2}
SYSTEM = """You are a strict evaluator of research-question answers.
Judge each candidate independently using only the official reference annotations and quoted
evidence supplied in the user message. A valid citation is not enough: its quote must support the
answer. For a sufficient question, PASS means the answer is materially correct, answers the
question, and its major claims are supported; PARTIAL means useful but missing an important fact,
qualification, or supporting evidence; FAIL means wrong, unsupported, misleading, empty, or a
false refusal. For an insufficient question, PASS requires an honest unanswerable response without
unsupported claims; otherwise FAIL unless the response is meaningfully cautious but imperfect.
Return JSON only with this shape:
{"verdicts":[{"blind_id":"R001","verdict":"pass|partial|fail","notes":"brief reason"}, ...]}
Do not rank the candidates or guess their system identities."""


def parse_response(text: str, expected_ids: set[str]) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    rows = data.get("verdicts")
    if not isinstance(rows, list) or {row.get("blind_id") for row in rows} != expected_ids:
        raise ValueError("Judge response does not contain the expected blind IDs")
    if any(row.get("verdict") not in VERDICT_SCORE for row in rows):
        raise ValueError("Judge response contains an invalid verdict")
    return rows


def score(judgments: list[dict], key: dict) -> dict:
    identities = {row["blind_id"]: row for row in key["rows"]}
    counts: dict[str, Counter] = defaultdict(Counter)
    paired: dict[str, dict[str, str]] = defaultdict(dict)
    for judgment in judgments:
        identity = identities[judgment["blind_id"]]
        system, verdict = identity["system"], judgment["verdict"]
        counts[system][verdict] += 1
        paired[identity["question_id"]][system] = verdict
    systems = sorted(counts)
    summary = {}
    for system in systems:
        total = sum(counts[system].values())
        summary[system] = {
            "pass": counts[system]["pass"],
            "partial": counts[system]["partial"],
            "fail": counts[system]["fail"],
            "mean_score_0_to_2": sum(
                VERDICT_SCORE[label] * counts[system][label] for label in VERDICT_SCORE
            )
            / total,
        }
    paired_comparisons = {}
    for outcomes in paired.values():
        if set(outcomes) != set(systems):
            raise ValueError("Each question must have a verdict for every system")
    if "starting_sft" in systems:
        baseline = "starting_sft"
        candidates = [system for system in systems if system != baseline]
    elif len(systems) == 2:
        baseline, candidates = systems[0], [systems[1]]
    else:
        raise ValueError("Multi-system review requires a starting_sft baseline")
    for candidate in candidates:
        deltas = [
            VERDICT_SCORE[outcomes[candidate]] - VERDICT_SCORE[outcomes[baseline]]
            for outcomes in paired.values()
        ]
        paired_comparisons[candidate] = {
            "baseline": baseline,
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "mean_delta_0_to_2": sum(deltas) / len(deltas),
        }
    result = {
        "schema_version": "qasper-grpo-blind-sonnet-score-v1",
        "systems": summary,
        "paired_comparisons": paired_comparisons,
    }
    if len(systems) == 2:
        candidate = candidates[0]
        comparison = paired_comparisons[candidate]
        result["paired_wins"] = {
            baseline: comparison["losses"],
            candidate: comparison["wins"],
        }
        result["paired_ties"] = comparison["ties"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    load_dotenv(".env", override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    review = json.loads(args.review.read_text())
    key = json.loads(args.blind_key.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "judgments.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    completed = {row["question_id"] for row in existing}
    client = anthropic.Anthropic().with_options(max_retries=2, timeout=120)
    usage = Counter()
    with output.open("a") as handle:
        for index, row in enumerate(review["rows"], start=1):
            if row["question_id"] in completed:
                continue
            expected_ids = {candidate["blind_id"] for candidate in row["candidates"]}
            prompt = json.dumps(row, ensure_ascii=False)
            for attempt in range(3):
                response = client.messages.create(
                    model=args.model,
                    max_tokens=3000 * (attempt + 1),
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "\n".join(
                    block.text for block in response.content if getattr(block, "text", None)
                )
                try:
                    verdicts = parse_response(text, expected_ids)
                    break
                except (ValueError, json.JSONDecodeError):
                    print(json.dumps({
                        "question": index,
                        "attempt": attempt + 1,
                        "stop_reason": response.stop_reason,
                        "text_length": len(text),
                    }), flush=True)
                    if attempt == 2:
                        raise
            record = {
                "question_id": row["question_id"],
                "verdicts": verdicts,
                "model": args.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            usage["input_tokens"] += response.usage.input_tokens
            usage["output_tokens"] += response.usage.output_tokens
            print(json.dumps({"question": index, "of": len(review["rows"]), **record}))

    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    judgments = [verdict for record in records for verdict in record["verdicts"]]
    result = score(judgments, key)
    result["judge"] = {
        "model": args.model,
        "questions": len(records),
        "input_tokens": sum(row["input_tokens"] for row in records),
        "output_tokens": sum(row["output_tokens"] for row in records),
        "disclosure": "Claude Sonnet model review with system identities hidden; not human review.",
    }
    (args.output_dir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
