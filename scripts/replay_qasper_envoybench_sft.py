"""Replay reviewed QASPER demonstrations under the exact EnvoyBench protocol.

This is a local, CPU-only data migration. It does not generate actions or alter
answers: every saved action is executed against the frozen paper corpus in the
same Docker sandbox, prompt, and verifier configuration as EnvoyBench. A row is
excluded if its tool-output bytes change (apart from the step-budget footer),
if the verifier intervenes, or if it does not finish with its saved submission.

The source-visible semantic review remains model-assisted, not human review.
Replaying an episode cannot establish that its answer is correct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, initial_observation
from src.eval.artifacts import content_hash
from src.eval.hashing import sha256
from src.eval.sft_quality import known_paper_id, trajectory_hash, trajectory_issues
from src.policies.code_execution import QASPER_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "out/research/qasper-synthetic-reliability-v1/accepted"
SOURCE_CANDIDATES = SOURCE_DIR / "candidates.jsonl"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
SOURCE_TRAIN = SOURCE_DIR / "train.jsonl"
SOURCE_VAL = SOURCE_DIR / "val.jsonl"
SOURCE_BENCHMARK = ROOT / "out/research/qasper-sonnet-expansion-20260919/benchmark.json"
SOURCE_CORPUS = ROOT / "out/research/qasper-rl-v1/train-v2/corpus"
HOLDOUT_LOCK = ROOT / "data/research/qasper-v6-holdout-lock.json"
DEFAULT_OUTPUT = ROOT / "out/research/qasper-v6-runtime-replayed-20261013"
PINNED_CANDIDATES_SHA256 = "87630b733b83d13689bd67b5d0920241a2e30412d0834ff00f4936eadce7af42"
RUNTIME_PROTOCOL = "envoybench-v1"
MAX_STEPS = 15


class ReplayRejectedError(ValueError):
    """A saved example cannot become a runtime-aligned training example."""


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _observation_payload(observation: str, *, step: int, budget: int) -> str:
    """Remove only the environment's budget footer, retaining exact tool bytes."""
    remaining = budget - step
    if remaining <= 1:
        # No accepted source demonstration reaches the final-step hint. Avoid
        # guessing its submission-format variant in a provenance migration.
        raise ReplayRejectedError("unexpected_step_footer")
    if remaining <= 5:
        footer = (
            f"\n\n[Step {step}/{budget} — {remaining} steps left. "
            "Run sufficiency check: can you answer now? If yes → SUBMIT immediately.]"
        )
    else:
        footer = f"\n\n[Step {step}/{budget}/{budget}]"
    if not observation.endswith(footer):
        raise ReplayRejectedError("unexpected_step_footer")
    return observation[:-len(footer)]


def _split_ids(path: Path, by_question: dict[str, str]) -> list[str]:
    ids = []
    for row in _load_jsonl(path):
        messages = row.get("messages", [])
        if len(messages) < 2 or messages[1].get("role") != "user":
            raise ValueError(f"Malformed source split: {path}")
        question = messages[1]["content"].removeprefix("Question: ").strip()
        if question not in by_question:
            raise ValueError(f"Unknown question in source split: {question[:100]}")
        ids.append(by_question[question])
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate questions in source split: {path}")
    return ids


def _load_sources() -> tuple[list[dict], dict[str, dict], dict[str, str], dict]:
    if sha256(SOURCE_CANDIDATES) != PINNED_CANDIDATES_SHA256:
        raise ValueError("The accepted 137-row source file no longer matches its published hash")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    benchmark = json.loads(SOURCE_BENCHMARK.read_text())
    holdout = json.loads(HOLDOUT_LOCK.read_text())
    source = source_manifest["source"]
    for key, path in (
        ("benchmark_sha256", SOURCE_BENCHMARK),
        ("train_sha256", ROOT / source["train"]),
        ("validation_sha256", ROOT / source["validation"]),
        ("semantic_review_sha256", ROOT / source["semantic_review"]),
    ):
        if sha256(path) != source[key]:
            raise ValueError(f"Source manifest binding changed: {key}")
    if benchmark.get("source_split") != "train":
        raise ValueError("Only original QASPER train examples may become SFT data")
    if content_hash(SOURCE_CORPUS) != benchmark["corpus_hash"]:
        raise ValueError("Frozen corpus differs from the source benchmark")
    rows = _load_jsonl(SOURCE_CANDIDATES)
    if len(rows) != source_manifest["accepted"] or len(rows) != 137:
        raise ValueError("Accepted source count changed")
    questions = {q["id"]: q for q in benchmark["questions"]}
    if len(questions) != len(benchmark["questions"]):
        raise ValueError("Duplicate benchmark question IDs")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate accepted question IDs")
    by_text = {q["question"]: qid for qid, q in questions.items()}
    if len(by_text) != len(questions):
        raise ValueError("Benchmark question text is not unique")
    train_ids = _split_ids(SOURCE_TRAIN, by_text)
    val_ids = _split_ids(SOURCE_VAL, by_text)
    candidate_ids = {r["question_id"] for r in rows}
    if set(train_ids) & set(val_ids) or set(train_ids + val_ids) != candidate_ids:
        raise ValueError("Accepted candidates and paper-disjoint source splits disagree")
    assignment = {qid: "train" for qid in train_ids}
    assignment.update({qid: "val" for qid in val_ids})
    train_papers = {known_paper_id(questions[qid]["question"]) for qid in train_ids}
    val_papers = {known_paper_id(questions[qid]["question"]) for qid in val_ids}
    held_papers = set(holdout["reserved_doc_ids"])
    if train_papers & val_papers or (train_papers | val_papers) & held_papers:
        raise ValueError("Training, validation, and frozen holdout papers overlap")
    return rows, questions, assignment, source_manifest


def _validate_source_row(row: dict, question: dict) -> None:
    if row.get("student_protocol") != "qasper-span-v1":
        raise ReplayRejectedError("not_qasper_span_protocol")
    if row.get("question") != question["question"]:
        raise ReplayRejectedError("question_mismatch")
    if row.get("expected_answerability") != question["expected_answerability"]:
        raise ReplayRejectedError("answerability_mismatch")
    if row.get("status") != "completed":
        raise ReplayRejectedError("source_not_completed")
    if row.get("review", {}).get("trajectory_sha256") != trajectory_hash(row):
        raise ReplayRejectedError("stale_source_review")
    if not all(row.get("review", {}).get(k) is True for k in
               ("replay_verified", "evidence_checked", "stopping_checked")):
        raise ReplayRejectedError("incomplete_source_review")
    if not row.get("qasper_score", {}).get("valid"):
        raise ReplayRejectedError("invalid_source_span_score")
    if trajectory_issues(row):
        raise ReplayRejectedError("source_trajectory_issues")
    steps = row["trajectory"]
    if not steps[-1]["action"].startswith("SUBMIT:") or not steps[-1].get("done"):
        raise ReplayRejectedError("source_incomplete_submission")
    if any(s.get("raw_action", s["action"]) != s["action"] for s in steps):
        raise ReplayRejectedError("source_action_modified")


def _conversation(row: dict) -> dict:
    messages = [
        {"role": "system", "content": QASPER_SYSTEM_PROMPT},
        {"role": "user", "content": row["initial_observation"]},
    ]
    for step in row["trajectory"]:
        messages.append({"role": "assistant", "content": step["action"]})
        if not step["done"]:
            messages.append({"role": "user", "content": step["observation"]})
    return {"messages": messages}


def replay_row(row: dict, question: dict, corpus: Corpus, corpus_path: Path) -> dict:
    """Return one exact-runtime candidate, or raise ReplayRejectedError."""
    _validate_source_row(row, question)
    original_hash = trajectory_hash(row)
    env = DocumentExplorationEnv(
        corpus=corpus, questions=[question], max_steps=MAX_STEPS,
        use_docker=True, corpus_path=str(corpus_path), require_evidence=True,
        include_preamble=True, evidence_verifier=True, verifier_feedback_budget=1,
        escalate_after_verifier_failure=True,
    )
    started = time.monotonic()
    try:
        initial = env.reset(0)
        if initial != initial_observation(
            question["question"], require_evidence=True, include_preamble=True,
        ):
            raise ReplayRejectedError("runtime_initial_observation_mismatch")
        for index, source_step in enumerate(row["trajectory"], start=1):
            action = source_step["action"]
            _, _, done, _ = env.step(action)
            episode = env.get_episode_info()
            replayed_steps = env.get_trajectory()
            if episode is None or episode.verifier_events or episode.escalation is not None:
                raise ReplayRejectedError("verifier_intervention")
            if len(replayed_steps) != index:
                raise ReplayRejectedError("replay_step_count_changed")
            current = replayed_steps[-1]
            if current.action != action or bool(current.done) != bool(source_step["done"]):
                raise ReplayRejectedError("action_or_terminal_flag_changed")
            if index < len(row["trajectory"]):
                if done or source_step["done"]:
                    raise ReplayRejectedError("premature_episode_end")
                source_payload = _observation_payload(
                    source_step["observation"], step=index, budget=10,
                )
                runtime_payload = _observation_payload(
                    current.observation, step=index, budget=MAX_STEPS,
                )
                if source_payload != runtime_payload:
                    raise ReplayRejectedError("changed_tool_output_payload")
            elif not done or current.observation != source_step["observation"]:
                raise ReplayRejectedError("changed_submission_result")
        episode = env.get_episode_info()
        assert episode is not None
        if episode.final_reward is None:
            raise ReplayRejectedError("missing_final_reward")
        steps = [s.model_dump(exclude_none=True) for s in env.get_trajectory()]
        for step in steps:
            step["raw_action"] = step["action"]
        migrated = dict(row)
        migrated.update({
            "trajectory": steps,
            "initial_observation": initial,
            "initial_observation_sha256": _text_hash(initial),
            "system_prompt_sha256": _text_hash(QASPER_SYSTEM_PROMPT),
            "runtime_protocol": RUNTIME_PROTOCOL,
            "runtime_replay_seconds": round(time.monotonic() - started, 3),
            "source_trajectory_sha256": original_hash,
            "source_review": row["review"],
            "review": {
                "basis": (
                    "Source-visible model-assisted review transferred only after "
                    "exact tool-output replay"
                ),
                "reviewer_type": "model-assisted, not independent human review",
                "replay_verified": True,
                "evidence_checked": row["review"]["evidence_checked"],
                "stopping_checked": row["review"]["stopping_checked"],
                "source_trajectory_sha256": original_hash,
            },
        })
        migrated["review"]["trajectory_sha256"] = trajectory_hash(migrated)
        return migrated
    finally:
        env.close()


def _docker_image_id() -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "rlm-sandbox", "--format", "{{.Id}}"],
        text=True, capture_output=True, check=True,
    )
    return result.stdout.strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _journal_digest(record: dict) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return _text_hash(payload)


def _journal_context(rows: list[dict], source_manifest: dict, image_id: str) -> dict:
    return {
        "type": "header",
        "schema_version": "qasper-envoybench-replay-journal-v1",
        "source": {
            "candidates": _identity(SOURCE_CANDIDATES),
            "manifest": _identity(SOURCE_MANIFEST),
            "train": _identity(SOURCE_TRAIN), "val": _identity(SOURCE_VAL),
            "benchmark": _identity(SOURCE_BENCHMARK),
            "semantic_review": _identity(ROOT / source_manifest["source"]["semantic_review"]),
        },
        "holdout_lock": _identity(HOLDOUT_LOCK),
        "corpus_content_hash": content_hash(SOURCE_CORPUS),
        "runtime": {
            "protocol": RUNTIME_PROTOCOL,
            "system_prompt_sha256": _text_hash(QASPER_SYSTEM_PROMPT),
            "initial_observation_sha256_by_question": {
                row["question_id"]: _text_hash(initial_observation(
                    row["question"], require_evidence=True, include_preamble=True,
                )) for row in rows
            },
            "max_steps": MAX_STEPS, "use_docker": True,
            "docker_image_id": image_id, "require_evidence": True,
            "include_preamble": True, "evidence_verifier": True,
            "verifier_feedback_budget": 1, "escalate_after_verifier_failure": True,
        },
        "implementation": {
            name: _identity(ROOT / name)["sha256"] for name in (
                "scripts/replay_qasper_envoybench_sft.py",
                "src/env/document_env.py", "src/env/repl.py", "src/env/tools.py",
                "src/env/evidence_state.py", "src/env/reward.py",
                "src/policies/code_execution.py",
            )
        },
        "question_ids": [row["question_id"] for row in rows],
    }


def _append_journal(path: Path, record: dict, previous_sha256: str | None) -> str:
    item = dict(record)
    if previous_sha256 is not None:
        item["previous_sha256"] = previous_sha256
    digest = _journal_digest(item)
    item["entry_sha256"] = digest
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest


def _read_journal(
    path: Path, context: dict, rows: list[dict],
) -> tuple[list[dict], list[dict], str]:
    records = _load_jsonl(path)
    if not records:
        raise ValueError("Empty replay journal")
    previous = None
    accepted, rejected = [], []
    for position, item in enumerate(records):
        expected = item.get("entry_sha256")
        content = {key: value for key, value in item.items() if key != "entry_sha256"}
        if expected != _journal_digest(content):
            raise ValueError(f"Tampered replay journal entry: {position}")
        if position == 0:
            if content != context:
                raise ValueError("Replay journal source, runtime, or holdout hash changed")
        else:
            if content.get("previous_sha256") != previous:
                raise ValueError(f"Broken replay journal chain: {position}")
            index = position - 1
            if index >= len(rows) or content.get("source_index") != index:
                raise ValueError(f"Out-of-order replay journal entry: {position}")
            row = rows[index]
            if (content.get("question_id") != row["question_id"] or
                    content.get("source_trajectory_sha256") != trajectory_hash(row)):
                raise ValueError(f"Replay journal source row changed: {position}")
            if content.get("type") == "accepted":
                migrated = content["candidate"]
                if (migrated.get("source_trajectory_sha256") != trajectory_hash(row)
                        or migrated.get("review", {}).get("trajectory_sha256")
                        != trajectory_hash(migrated)
                        or migrated.get("runtime_protocol") != RUNTIME_PROTOCOL):
                    raise ValueError(f"Replay journal candidate changed: {position}")
                accepted.append(migrated)
            elif content.get("type") == "rejected":
                rejected.append({
                    "question_id": content["question_id"],
                    "source_trajectory_sha256": content["source_trajectory_sha256"],
                    "reason": content["reason"],
                })
            else:
                raise ValueError(f"Unexpected replay journal outcome: {position}")
        previous = expected
    return accepted, rejected, previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=int, help="Docker smoke replay only; never writes train/val",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume a hash-checked partial replay",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out_dir}")
    rows, questions, assignment, source_manifest = _load_sources()
    if args.limit is not None:
        if not 1 <= args.limit <= len(rows):
            raise SystemExit("--limit must be between 1 and 137")
        rows = rows[:args.limit]
    corpus = Corpus(corpus_path=str(SOURCE_CORPUS))
    corpus.load(build_index=False)
    image_id = _docker_image_id()
    output = args.out_dir.resolve()
    journal = output.with_name(output.name + ".journal.jsonl")
    context = _journal_context(rows, source_manifest, image_id)
    if args.resume:
        if not journal.exists():
            raise SystemExit(f"No replay journal to resume: {journal}")
        accepted, rejected, previous_sha256 = _read_journal(journal, context, rows)
        completed = len(accepted) + len(rejected)
        print(f"Resuming replay after {completed}/{len(rows)} journaled outcomes", flush=True)
    else:
        if journal.exists():
            raise SystemExit(f"Replay journal already exists; use --resume: {journal}")
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.touch(exist_ok=False)
        previous_sha256 = _append_journal(journal, context, None)
        accepted, rejected, completed = [], [], 0
    for index, row in enumerate(rows[completed:], start=completed + 1):
        qid = row["question_id"]
        try:
            migrated = replay_row(row, questions[qid], corpus, SOURCE_CORPUS)
        except ReplayRejectedError as exc:
            outcome = {
                "question_id": qid, "source_trajectory_sha256": trajectory_hash(row),
                "reason": str(exc),
            }
            rejected.append(outcome)
            previous_sha256 = _append_journal(journal, {
                "type": "rejected", "source_index": index - 1, **outcome,
            }, previous_sha256)
            print(f"[{index}/{len(rows)}] REJECT {qid}: {exc}", flush=True)
        else:
            accepted.append(migrated)
            previous_sha256 = _append_journal(journal, {
                "type": "accepted", "source_index": index - 1,
                "question_id": qid, "source_trajectory_sha256": trajectory_hash(row),
                "candidate": migrated,
            }, previous_sha256)
            print(f"[{index}/{len(rows)}] replayed {qid}", flush=True)
    if not accepted:
        raise RuntimeError("No replayed candidates survived")
    if _journal_context(rows, source_manifest, _docker_image_id()) != context:
        raise RuntimeError("Replay context changed during execution; refusing to finalize")
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "candidates.jsonl", accepted)
    split_report = {}
    if args.limit is None:
        for split in ("train", "val"):
            part = [r for r in accepted if assignment[r["question_id"]] == split]
            if not part:
                raise RuntimeError(f"Replay emptied the {split} split")
            _write_jsonl(output / f"{split}.jsonl", [_conversation(r) for r in part])
            split_report[split] = {
                "count": len(part), "question_ids": [r["question_id"] for r in part],
                "paper_ids": sorted({known_paper_id(r["question"]) for r in part}),
                **_identity(output / f"{split}.jsonl"),
            }
        if set(split_report["train"]["paper_ids"]) & set(split_report["val"]["paper_ids"]):
            raise RuntimeError("Replay produced overlapping paper splits")
    manifest = {
        "schema_version": "qasper-envoybench-sft-replay-v1",
        "status": "complete" if args.limit is None else "smoke_only_no_training_splits",
        "runtime_protocol": RUNTIME_PROTOCOL,
        "source": {
            "candidates": _identity(SOURCE_CANDIDATES),
            "manifest": _identity(SOURCE_MANIFEST),
            "train": _identity(SOURCE_TRAIN), "val": _identity(SOURCE_VAL),
            "benchmark": _identity(SOURCE_BENCHMARK),
            "semantic_review": _identity(ROOT / source_manifest["source"]["semantic_review"]),
            "reviewer": source_manifest["reviewer"],
            "reviewer_type": source_manifest["reviewer_type"],
        },
        "holdout_lock": _identity(HOLDOUT_LOCK),
        "corpus": {
            "path": str(SOURCE_CORPUS.relative_to(ROOT)),
            "content_hash": content_hash(SOURCE_CORPUS),
        },
        "runtime": {
            "system_prompt_sha256": _text_hash(QASPER_SYSTEM_PROMPT),
            "initial_observation_sha256_by_question": {
                r["question_id"]: r["initial_observation_sha256"] for r in accepted
            },
            "max_steps": MAX_STEPS, "use_docker": True,
            "docker_image_id": image_id, "require_evidence": True,
            "include_preamble": True, "evidence_verifier": True,
            "verifier_feedback_budget": 1, "escalate_after_verifier_failure": True,
        },
        "implementation": {
            name: _identity(ROOT / name)["sha256"] for name in (
                "scripts/replay_qasper_envoybench_sft.py",
                "src/env/document_env.py", "src/env/repl.py", "src/env/tools.py",
                "src/env/evidence_state.py", "src/env/reward.py",
                "src/policies/code_execution.py",
            )
        },
        "source_count": source_manifest["accepted"],
        "attempted_count": len(rows), "accepted_count": len(accepted),
        "rejected_count": len(rejected), "rejections": rejected,
        "answerability": dict(Counter(r["expected_answerability"] for r in accepted)),
        "teacher_behaviors": dict(Counter(r["teacher_behavior"] for r in accepted)),
        "candidates": _identity(output / "candidates.jsonl"),
        "splits": split_report,
        "claim_boundary": (
            "Actions and tool-output payloads replayed under the EnvoyBench runtime. "
            "Inherited source-visible model-assisted review is not independent human review "
            "and this data migration is not a model-improvement result."
        ),
        "journal": _identity(journal),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(accepted)} accepted, {len(rejected)} rejected -> {output}")


if __name__ == "__main__":
    main()
