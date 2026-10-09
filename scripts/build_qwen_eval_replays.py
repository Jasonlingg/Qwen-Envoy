"""Bundle two real, paired Qwen evaluation cases for the local viewer.

The source episodes are ignored local artifacts. The small replay bundles are
committable examples; observations are excerpted, while actions and outcomes
remain copied from the recorded runs. This script never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.eval.qasper_reward import parse_strict

ARMS = ("starting_sft", "epoch_1", "epoch_2")
CASES = {
    "qwen_answerable_tradeoff": {
        "question_id": "qasper_validation_0d34c0812f1e69ea33f76ca8c24c23b0415ebc8d",
        "label": "Recorded Qwen: one answerable question, three checkpoints",
        "selection": "Epoch 1 repairs a starting-checkpoint false refusal; epoch 2 refuses again.",
    },
    "qwen_abstention_regression": {
        "question_id": "qasper_validation_596aede2b311deb8cb0a82d2e7de314ef6e83e4e",
        "label": "Recorded Qwen: an unanswerable question, three checkpoints",
        "selection": (
            "The starting checkpoint abstains; both continuation checkpoints "
            "answer without support."
        ),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(path: Path, question_id: str) -> tuple[dict, str]:
    matches = [
        line for line in path.read_text().splitlines()
        if line.strip() and json.loads(line).get("question_id") == question_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one episode for {question_id} in {path}")
    return json.loads(matches[0]), hashlib.sha256(matches[0].encode()).hexdigest()


def load_verdicts(review_dir: Path, question_id: str) -> dict[str, dict]:
    key = json.loads((review_dir / "blind-key.json").read_text())
    identities = {row["blind_id"]: row for row in key["rows"]}
    records = [
        json.loads(line) for line in (review_dir / "judgments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matches = [row for row in records if row["question_id"] == question_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one blind review for {question_id}")
    verdicts = {
        identities[row["blind_id"]]["system"]: row
        for row in matches[0]["verdicts"]
    }
    if set(verdicts) != set(ARMS):
        raise ValueError(f"Incomplete review for {question_id}")
    return verdicts


def build_case(eval_root: Path, review_dir: Path, name: str, case: dict) -> dict:
    question_id = case["question_id"]
    episodes = {}
    hashes = {}
    for arm in ARMS:
        episodes[arm], hashes[arm] = load_episode(
            eval_root / arm / "episodes.jsonl", question_id
        )
    questions = {row["question"] for row in episodes.values()}
    answerability = {row["expected_answerability"] for row in episodes.values()}
    if len(questions) != 1 or len(answerability) != 1:
        raise ValueError(f"Arms disagree on {question_id}")
    verdicts = load_verdicts(review_dir, question_id)
    events = []
    max_steps = max(len(episode["trajectory"]) for episode in episodes.values())
    for step_index in range(max_steps):
        for arm in ARMS:
            episode = episodes[arm]
            if step_index >= len(episode["trajectory"]):
                continue
            step = episode["trajectory"][step_index]
            final = step_index == len(episode["trajectory"]) - 1
            parsed = parse_strict(step["action"]) if final else None
            if final and (not parsed or not episode["valid"] or episode["finish"] != "submitted"):
                raise ValueError(f"Invalid recorded submission: {arm} {question_id}")
            events.append({
                "policy": arm,
                "question_id": question_id,
                "question": episode["question"],
                "step": step_index + 1,
                "action": step["action"],
                "observation": step.get("observation", "")[:400],
                "done": final,
                "total_reward": episode["reward"] if final else 0.0,
                "predicted_answer": parsed[0] if parsed else "",
                "predicted_citations": parsed[1] if parsed else [],
                "semantic_verdict": verdicts[arm]["verdict"] if final else None,
                "semantic_notes": verdicts[arm]["notes"] if final else None,
                "scored": True,
            })
    return {
        "meta": {
            "question_id": question_id,
            "question": next(iter(questions)),
            "expected_answerability": next(iter(answerability)),
            "label": case["label"],
            "selection": case["selection"],
            "source": "2026-09-25 controlled QASPER evaluation; recorded Qwen3-8B actions",
            "replay_not_live": True,
            "observations_excerpted_to_chars": 400,
            "semantic_review": "Identity-blind Claude Sonnet 5; not independent human review",
            "episode_sha256": hashes,
            "judgments_sha256": sha256(review_dir / "judgments.jsonl"),
            "name": name,
        },
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/replays"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        bundle = build_case(args.eval_root, args.review_dir, name, case)
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
        print(f"{path}: {len(bundle['events'])} recorded actions")


if __name__ == "__main__":
    main()
