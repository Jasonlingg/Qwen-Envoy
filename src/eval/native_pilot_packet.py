"""Blinded human-review inputs; never invent scores or convert provenance to quality."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def tool_receipts(items):
    receipts = []
    for item in items:
        if item["type"] not in {"webSearch", "mcpToolCall", "commandExecution"}:
            continue
        encoded = json.dumps(item, sort_keys=True, ensure_ascii=False).encode()
        receipts.append({key: item.get(key) for key in (
            "id", "type", "server", "tool", "arguments", "action", "query", "status",
            "error", "durationMs", "exitCode") if key in item})
        receipts[-1].update(item_sha256=hashlib.sha256(encoded).hexdigest(),
                            source_urls=sorted(set(re.findall(
                                r'https?://[^\s"<>\\]+', encoded.decode()))))
    return receipts


def package(source: Path, destination: Path, manifest: dict, questions: dict, seed: int):
    """Raw local traces stay outside the blind packet and retain their hashes."""
    destination.mkdir(parents=True, exist_ok=False)
    rng, mapping, summaries, worksheet = random.Random(seed), {}, [], []
    for question in questions["questions"]:
        qid = question["id"]
        arms = ["A", "B"]
        rng.shuffle(arms)
        review_question = {"id": qid, "common_instruction": questions["common_instruction"],
                           "question": question["question"], "input": question["input"]}
        save(destination / "review" / qid / "question.json", review_question)
        pair = {"question_id": qid, "answers": [], "pairwise_preference": None,
                "preference_reason": None, "blinding_cues": [], "reviewer_kind": None,
                "reviewer_identity": None, "initial_review_seconds": None}
        for arm in arms:
            path = source / f"{qid}-{arm}"
            result = (json.loads((path / "result.json").read_text())
                      if (path / "result.json").is_file() else {
                          "question_id": qid, "arm": arm, "status": "unattempted",
                          "model_turns_attempted": 0, "answer": "", "items": []})
            anonymous = f"answer-{rng.getrandbits(64):016x}"
            mapping[anonymous] = {"question_id": qid, "arm": arm,
                                  "local_raw_directory": str(path.resolve())}
            answer = result.get("answer", "")
            candidate = result.get("unclassified_assistant_messages", [])
            if not answer:
                answer = ("[No classified final answer was recorded.]\n" +
                          ("\nUnclassified assistant messages:\n\n" + "\n\n".join(candidate)
                           if candidate else ""))
            answer_path = destination / "review" / qid / f"{anonymous}.md"
            answer_path.write_text(answer)
            pair["answers"].append({"anonymous_answer_id": anonymous,
                "review_timestamp": None, "four_dimension_scores_or_null": None,
                "must_have_checks": [], "initial_verdict": "review_unknown",
                "major_error_answer_flag": None, "material_claim_errors": [],
                "claim_source_checks": [], "reason": None, "human_correction_seconds_or_null": None,
                "correction_censored": None, "corrected_artifact_path_or_null": None,
                "review_limitations": []})
            receipts = tool_receipts(result.get("items", []))
            exported = {key: result.get(key) for key in (
                "question_id", "arm", "status", "model_turns_attempted", "elapsed_seconds",
                "research_calls_observed", "research_target_overshoot", "answer_present",
                "answer_words_including_sources", "started_at_utc", "finished_at_utc",
                "watchdog_fired", "research_interrupt_requested", "token_usage", "error",
                "prompt_sha256")}
            exported.update(status=result["status"], human_review="not_performed",
                            actual_spend_usd=None, host_model_requests=None,
                            tool_receipts=receipts,
                            raw_artifact_sha256={name: digest(path / name) for name in (
                                "events.jsonl", "result.json", "answer.md", "settings.json",
                                "mcp-inventory.json", "thread.json", "boundary-checks.json")})
            save(destination / "operator" / f"{qid}-{arm}.json", exported)
            summaries.append({key: value for key, value in exported.items()
                              if key not in {"tool_receipts", "raw_artifact_sha256"}})
        worksheet.append(pair)
    save(destination / "operator" / "arm-map.json", {"seed": seed, "mapping": mapping})
    save(destination / "review" / "worksheet.json", {"human_review_status": "not_performed",
                                                    "pairs": worksheet})
    save(destination / "collection-report.json", {
        "schema_version": "native-ml-pilot-collection-v1", "scheduled_answers": 10,
        "scheduled_pairs": 5,
        "outcomes": sorted(summaries, key=lambda x: (x["question_id"], x["arm"])),
        "generation_turns_attempted": sum(row.get("model_turns_attempted") or 0
                                         for row in summaries),
        "human_quality_winner": "unknown_pending_human_source_review",
        "actual_spend_usd": None, "extra_qwen_or_provider_endpoint_calls": 0, "gpu_starts": 0,
        "execution_manifest": manifest, "raw_local_collection": str(source.resolve()),
        "limitations": ["Raw source return bodies and full native traces remain in the local "
                        "collection directory; receipts bind hashes rather than semantic truth.",
                        "Human quality, preference, correction and time-to-useful remain unknown.",
                        "Host request counts, complete remote activity and dollars are unknown.",
                        "Per-answer words include sources; they are not a token-spend cap."]})
