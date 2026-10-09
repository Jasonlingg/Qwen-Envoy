"""CPU fixture, lexical baseline, and transcript diagnostics for memory-worker readiness.

This is a synthetic development screen. It does not judge semantic answer support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_ai_paper_id_diagnostics import tokenize  # noqa: E402
from scripts.validate_learning_loop_pilot import validate as validate_anchors  # noqa: E402
from src.env.evidence_state import FAILED_OBSERVATION_MARKERS, parse_document_calls  # noqa: E402
from src.eval.artifacts import content_hash  # noqa: E402
from src.eval.research_review import load_corpus, materialize_evidence  # noqa: E402
from src.research.benchmark import load_benchmark  # noqa: E402
from src.research.vault import import_vault  # noqa: E402

SAMPLE_VAULT = ROOT / "data/product_memory/sample_vault"
QUESTIONS = ROOT / "data/product_memory/questions_v2.json"
COLLECTION = "Learning"
BASELINE_NAME = "bm25_passage_baseline"


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _validate_snapshot(snapshot: Path, benchmark_path: Path = QUESTIONS) -> tuple[dict, dict]:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    benchmark = load_benchmark(benchmark_path, manifest)
    if content_hash(snapshot / "corpus") != manifest["corpus_hash"]:
        raise ValueError("snapshot corpus content has changed")
    return manifest, benchmark


def freeze(output: Path) -> dict:
    """Import once, or check that an existing frozen import still matches source files."""
    if not output.exists():
        import_vault(SAMPLE_VAULT, COLLECTION, output)
    manifest, benchmark = _validate_snapshot(output)
    if manifest.get("status") != "complete":
        raise ValueError("the frozen sample import has failures")
    for item in manifest["papers"]:
        source = SAMPLE_VAULT / item["source_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError(f"sample vault changed after freeze: {item['source_path']}")
    source_check, lines = validate_anchors(QUESTIONS, output)
    check_path = output / "source-check-v2.md"
    if not check_path.exists():
        check_path.write_text("\n".join(lines) + "\n")
    return {
        "status": "source_checked_agent_authored_human_review_pending",
        "snapshot": str(output),
        "benchmark_id": benchmark["benchmark_id"],
        "corpus_hash": manifest["corpus_hash"],
        "documents": len(manifest["papers"]),
        "questions": source_check["question_count"],
        "source_check": str(check_path),
    }


def _passages(
    documents: dict[str, dict], *, chunk_size: int = 512, overlap: int = 64
) -> list[dict]:
    rows = []
    for doc_id, doc in sorted(documents.items()):
        for start in range(0, len(doc["text"]), chunk_size - overlap):
            end = min(start + chunk_size, len(doc["text"]))
            rows.append(
                {
                    "doc_id": doc_id,
                    "start": start,
                    "end": end,
                    "title": doc.get("title", doc_id),
                    "text": doc["text"][start:end],
                }
            )
    return rows


def bm25_evidence(
    question: str,
    passages: list[dict],
    index: BM25Okapi,
    *,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve exact spans from question text only; diversify overlapping windows."""
    scores = index.get_scores(tokenize(question))
    ranked = sorted(
        ((float(score), row) for score, row in zip(scores, passages) if score > 0),
        key=lambda item: (-item[0], item[1]["doc_id"], item[1]["start"]),
    )
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for score, row in ranked:
        doc_id = row["doc_id"]
        if counts.get(doc_id, 0) >= 2:
            continue
        if any(
            item["doc_id"] == doc_id and row["start"] < item["end"] and row["end"] > item["start"]
            for item in selected
        ):
            continue
        selected.append(
            {"doc_id": doc_id, "start": row["start"], "end": row["end"], "score": round(score, 6)}
        )
        counts[doc_id] = counts.get(doc_id, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def baseline(snapshot: Path, output: Path, *, top_k: int = 5) -> dict:
    manifest, benchmark = _validate_snapshot(snapshot)
    documents = load_corpus(snapshot / "corpus")
    passages = _passages(documents)
    index = BM25Okapi(
        [tokenize(f"{row['title']} {row['title']} {row['text']}") for row in passages]
    )
    rows = []
    for question in benchmark["questions"]:
        started = time.monotonic()
        evidence = bm25_evidence(question["question"], passages, index, top_k=top_k)
        rows.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "policy": BASELINE_NAME,
                "run_label": BASELINE_NAME,
                "status": "retrieval_only",
                "predicted_answer": "",
                "predicted_citations": sorted({item["doc_id"] for item in evidence}),
                "predicted_evidence": [
                    {key: item[key] for key in ("doc_id", "start", "end")} for item in evidence
                ],
                "duration_seconds": time.monotonic() - started,
                "trajectory": [],
            }
        )
    _write_new(output, rows)
    _write_new(
        output.with_suffix(".manifest.json"),
        {
            "run_label": BASELINE_NAME,
            "kind": "retrieval_only_no_generated_answer",
            "benchmark_id": benchmark["benchmark_id"],
            "question_ids": [q["id"] for q in benchmark["questions"]],
            "questions_sha256": content_hash(QUESTIONS),
            "corpus_sha256": manifest["corpus_hash"],
            "retrieval": {
                "method": "BM25 over 512-character passages, 64-character overlap, title twice",
                "top_k": top_k,
                "max_two_spans_per_document": True,
                "positive_score_only": True,
                "uses_reference_answers_or_source_anchors": False,
            },
            "semantic_support_review": "pending; retrieval-only packet has no generated answer",
        },
    )
    return {
        "baseline": str(output),
        "questions": len(rows),
        "corpus_hash": manifest["corpus_hash"],
        "kind": "retrieval_only",
    }


def _step_diagnostics(row: dict) -> dict:
    successful_tool_steps = 0
    tool_steps = 0
    errors = 0
    repeats = 0
    seen: set[str] = set()
    for step in row.get("trajectory", []):
        action = str(step.get("action", "")).strip()
        observation = str(step.get("observation", ""))
        observation_lower = observation.lower()
        failed = "traceback" in observation_lower or any(
            marker in observation_lower for marker in FAILED_OBSERVATION_MARKERS
        )
        if not action.upper().startswith("SUBMIT:"):
            if action in seen:
                repeats += 1
            seen.add(action)
        calls = parse_document_calls(action)
        if calls:
            tool_steps += 1
            if not failed:
                successful_tool_steps += 1
        if failed:
            errors += 1
    return {
        "document_tool_action_steps": tool_steps,
        "successful_document_tool_action_steps": successful_tool_steps,
        "execution_error_steps": errors,
        "repeated_action_steps": repeats,
    }


def _run_metrics(rows: list[dict], benchmark: dict, documents: dict[str, dict]) -> dict:
    question_by_id = {q["id"]: q for q in benchmark["questions"]}
    per_question = []
    for row in rows:
        q = question_by_id[row["question_id"]]
        evidence = materialize_evidence(row.get("predicted_evidence"), documents)
        evidence_docs = {item["doc_id"] for item in evidence if item["valid"]}
        required = set(q["required_doc_ids"])
        anchors = q.get("source_anchors", [])
        anchors_covered = sum(
            any(
                item["valid"]
                and item["doc_id"] == anchor["doc_id"]
                and anchor["needle"] in item["quote"]
                for item in evidence
            )
            for anchor in anchors
        )
        tool = _step_diagnostics(row)
        per_question.append(
            {
                "question_id": q["id"],
                "expected_answerability": q["expected_answerability"],
                "status": row.get("status"),
                "submitted_answer": bool(str(row.get("predicted_answer", "")).strip()),
                "evidence_spans": len(evidence),
                "valid_evidence_spans": sum(item["valid"] for item in evidence),
                "required_note_hits": len(evidence_docs & required),
                "required_note_count": len(required),
                "strict_reference_anchor_hits": anchors_covered,
                "strict_reference_anchor_count": len(anchors),
                "duration_seconds": round(float(row.get("duration_seconds") or 0), 4),
                "escalated": row.get("status") == "escalated" or bool(row.get("escalation")),
                **tool,
            }
        )
    total_spans = sum(item["evidence_spans"] for item in per_question)
    required_total = sum(item["required_note_count"] for item in per_question)
    anchor_total = sum(item["strict_reference_anchor_count"] for item in per_question)
    count = len(per_question)
    return {
        "question_count": count,
        "summary": {
            "submitted_answer_count": sum(item["submitted_answer"] for item in per_question),
            "valid_evidence_span_rate": (
                sum(item["valid_evidence_spans"] for item in per_question) / total_spans
                if total_spans
                else None
            ),
            "required_note_recall": (
                sum(item["required_note_hits"] for item in per_question) / required_total
                if required_total
                else None
            ),
            "strict_reference_anchor_recall_proxy": (
                sum(item["strict_reference_anchor_hits"] for item in per_question) / anchor_total
                if anchor_total
                else None
            ),
            "document_tool_action_steps": sum(
                item["document_tool_action_steps"] for item in per_question
            ),
            "successful_document_tool_action_steps": sum(
                item["successful_document_tool_action_steps"] for item in per_question
            ),
            "execution_error_episodes": sum(
                item["execution_error_steps"] > 0 for item in per_question
            ),
            "repeated_action_steps": sum(item["repeated_action_steps"] for item in per_question),
            "escalated_episodes": sum(item["escalated"] for item in per_question),
            "mean_duration_seconds": sum(item["duration_seconds"] for item in per_question) / count,
        },
        "per_question": per_question,
    }


def diagnose(questions: Path, corpus: Path, runs: list[Path], output: Path) -> dict:
    benchmark = load_benchmark(questions)
    questions_hash = content_hash(questions)
    corpus_hash = content_hash(corpus)
    if corpus_hash != benchmark["corpus_hash"]:
        raise ValueError("question file and corpus hash differ")
    documents = load_corpus(corpus)
    expected = {q["id"] for q in benchmark["questions"]}
    question_text = {q["id"]: q["question"] for q in benchmark["questions"]}
    expected_order = [q["id"] for q in benchmark["questions"]]
    systems = {}
    for path in runs:
        manifest_path = path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"run manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("questions_sha256") != questions_hash:
            raise ValueError(f"run question file hash differs: {path}")
        if manifest.get("corpus_sha256") != corpus_hash:
            raise ValueError(f"run corpus hash differs: {path}")
        if manifest.get("question_ids") != expected_order:
            raise ValueError(f"run question order or IDs differ: {path}")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError(f"run is incomplete or malformed: {path}")
        ids = [row.get("question_id") for row in rows]
        if set(ids) != expected or len(set(ids)) != len(ids):
            raise ValueError(f"run question IDs differ from benchmark: {path}")
        if any(row.get("question") != question_text[row["question_id"]] for row in rows):
            raise ValueError(f"run question text differs from benchmark: {path}")
        labels = {row.get("run_label") or row.get("policy") for row in rows}
        if len(labels) != 1:
            raise ValueError(f"mixed system labels: {path}")
        label = labels.pop()
        if manifest.get("run_label") != label:
            raise ValueError(f"run label differs from manifest: {path}")
        if label in systems:
            raise ValueError(f"duplicate system label: {label}")
        systems[label] = {
            "path": str(path),
            "kind": "retrieval_only" if label == BASELINE_NAME else "model_episode",
            **_run_metrics(rows, benchmark, documents),
        }
    result = {
        "schema_version": "personal-memory-readiness-diagnostics-v1",
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_status": benchmark.get("status"),
        "questions_sha256": questions_hash,
        "corpus_sha256": corpus_hash,
        "systems": systems,
        "semantic_support": (
            "pending independent human review; spans and gold-note recall "
            "are proxies only"
        ),
        "interpretation": (
            "If this is the historical eight-paper development set, these are paper-task tool "
            "diagnostics, not live Qwen results on personal memory. A synthetic-fixture result "
            "is a development screen, not private-vault transfer evidence."
        ),
    }
    _write_new(output, result)
    return {
        "diagnostics": str(output),
        "benchmark_id": benchmark["benchmark_id"],
        "systems": list(systems),
        "semantic_support": "pending",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    baseline_parser = commands.add_parser("baseline")
    baseline_parser.add_argument("--snapshot", type=Path, required=True)
    baseline_parser.add_argument("--output", type=Path, required=True)
    baseline_parser.add_argument("--top-k", type=int, default=5)
    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument("--questions", type=Path, required=True)
    diagnose_parser.add_argument("--corpus", type=Path, required=True)
    diagnose_parser.add_argument("--run", type=Path, action="append", required=True)
    diagnose_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(args.output)
        elif args.command == "baseline":
            if not 1 <= args.top_k <= 5:
                raise ValueError("top-k must be between 1 and 5")
            result = baseline(args.snapshot, args.output, top_k=args.top_k)
        else:
            result = diagnose(args.questions, args.corpus, args.run, args.output)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
