"""Atomic, identity-bound checkpoints for complete exact V1 replay questions."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _validate(result: dict, source: dict) -> None:
    if (result.get("question_id") != source["question_id"]
            or result.get("question") != source["question"]
            or result.get("v1_exact_match") is not True
            or not isinstance(result.get("initial_observation"), str)
            or not result["initial_observation"]
            or not isinstance(result.get("actions"), list)
            or len(result["actions"]) != len(source["trajectory"])):
        raise ValueError("Checkpoint does not contain a complete exact V1 question")
    for actual, saved in zip(result["actions"], source["trajectory"]):
        fields = {"step": "step", "action": "action", "recorded_observation": "observation",
                  "v1_observation": "observation", "v1_reward": "reward",
                  "recorded_reward": "reward", "v1_done": "done", "recorded_done": "done"}
        if (not isinstance(actual, dict) or actual.get("v1_exact_match") is not True
                or actual.get("normalization") != "none"
                or any(actual.get(key) != saved[target] for key, target in fields.items())
                or type(actual.get("v1_executed")) is not bool
                or (actual["v1_executed"] and not isinstance(
                    actual.get("v1_tool_observation"), str))
                or (not actual["v1_executed"] and actual.get("v1_tool_observation") is not None)):
            raise ValueError("Checkpoint action differs from the complete recorded V1 action")


def checkpointed_baselines(rows: list[dict], modules: list, execute, *,
                           checkpoint: Path, identity: dict, workers: int,
                           resume: bool) -> tuple[list[dict], int]:
    """Reuse only validated complete rows; persist each completed exact question."""
    sources = {row["question_id"]: row for row in rows}
    token = _hash(identity)
    entries = {}
    if checkpoint.exists():
        if not resume:
            raise ValueError(f"Checkpoint already exists; use --resume: {checkpoint}")
        saved = json.loads(checkpoint.read_text())
        if saved.get("schema") != "harness-v1-checkpoint-v1" or saved.get("identity") != token:
            raise ValueError("Checkpoint run/corpus/image/infrastructure identities differ")
        entries = saved.get("questions")
        if not isinstance(entries, dict) or set(entries) - sources.keys():
            raise ValueError("Checkpoint contains unknown or invalid questions")
        for qid, item in entries.items():
            if (not isinstance(item, dict) or item.get("source_hash") != _hash(sources[qid])
                    or item.get("result_hash") != _hash(item.get("result"))):
                raise ValueError("Checkpoint per-question/action binding differs")
            _validate(item["result"], sources[qid])
    reused = len(entries)
    results = {qid: item["result"] for qid, item in entries.items()}
    pending = [(row, module) for row, module in zip(rows, modules)
               if row["question_id"] not in entries]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, pair): pair[0] for pair in pending}
        for future in as_completed(futures):
            row, result = futures[future], future.result()
            results[row["question_id"]] = result
            if result.get("v1_exact_match") is not True:
                continue
            _validate(result, row)
            entries[row["question_id"]] = {"source_hash": _hash(row), "result": result,
                                            "result_hash": _hash(result)}
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            temporary.write_text(json.dumps({"schema": "harness-v1-checkpoint-v1",
                                            "identity": token, "questions": entries}) + "\n")
            temporary.replace(checkpoint)
    return [results[row["question_id"]] for row in rows], reused
