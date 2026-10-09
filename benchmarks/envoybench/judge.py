"""Identity-blind, model-assisted judging of an unreviewed QASPER bundle.

This is a provisional diagnostic, never an independent human review. The
command reads only the anonymous ``review.json``; it does not open the blind
key, automatic scores, run results, or model identities. Its default mode is
an offline cost preview. ``--execute`` is required for paid model calls.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable, Iterator

from benchmarks.envoybench.score import REVIEW_VERSION, RUBRIC_VERSION, _review_content
from src.eval.artifacts import configuration_hash

DEFAULT_MODEL = "claude-sonnet-5"
JUDGE_VERSION = "envoybench-provisional-judge-v1"
CHECKPOINT_VERSION = "envoybench-provisional-judge-checkpoint-v1"
QUESTION_RECORD_VERSION = "envoybench-provisional-judge-question-v1"
MAX_OUTPUT_TOKENS = 2000
DEFAULT_MAX_REQUEST_CHARS = 100_000
DEFAULT_MAX_TOTAL_INPUT_CHARS = 250_000
ABSENCE_NOTE = "No supporting passage is supplied by the frozen QASPER reference or candidate."
VERDICTS = {"pass", "partial", "fail"}
ROW_FIELDS = {
    "blind_id", "question_id", "question", "expected_answerability",
    "reference_answer", "grader_notes", "status", "answer", "citations",
    "evidence", "verdict", "notes", "relevant_source_passage",
}
SYSTEM_PROMPT = """You are a strict, identity-blind evaluator of scientific-paper answers.
These are provisional QASPER-derived references, not independently validated gold.
Judge each anonymous answer independently against the supplied reference answer,
QASPER annotated evidence, and the candidate's OWN submitted citations and
exact candidate quotations. A real or exact quote does not by itself support
an answer. Do not guess model identities or compare candidates. For an answerable
question, PASS requires a submitted, materially correct answer whose important
claims are supported by the candidate's OWN valid, relevant cited evidence.
Reference evidence cannot rescue an answer with missing, invalid, or misleading
candidate citations/evidence; such an answer is FAIL. PARTIAL means a useful,
partly supported answer misses an important detail or qualification without a
material false claim. FAIL also covers wrong, misleading, empty, or false refusal.
Any run status other than submitted, or an empty answer, is FAIL even when the
QASPER annotation says unanswerable; an execution error is not a correct refusal.
For a QASPER-unanswerable question, PASS means an honest refusal without invented
claims; this refusal may omit citations/evidence. An unsupported substantive
answer is FAIL. When the reference is ambiguous, explain the uncertainty in
notes; do not silently change its label. Give a nonempty reason for EVERY verdict,
including PASS.
Return only JSON: {"verdicts":[{"blind_id":"R001","verdict":"pass|partial|fail",
"notes":"brief explanation","relevant_source_passage":"exact supplied quote or absence note"}]}.
For every verdict, notes must explain the judgment. The source passage must be
an exact substring of the supplied reference evidence or valid candidate quote;
if none supports the needed claim, use exactly: """ + ABSENCE_NOTE


def load_prepared_review(path: Path) -> dict:
    """Load only a fresh provisional review file, never the identity key."""
    if path.name != "review.json" or path.is_symlink():
        raise ValueError("input must be a regular, anonymous review.json file")
    review = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("review must be a JSON object")
    if review.get("schema_version") != REVIEW_VERSION:
        raise ValueError("unsupported blind review schema")
    if review.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("unsupported rubric version")
    if review.get("source_reference_status") != "unreviewed_qasper":
        raise ValueError("judge accepts only explicitly unreviewed QASPER references")
    if review.get("reference_review_hash") is not None:
        raise ValueError("provisional review must not claim a human reference review")
    if review.get("status") != "incomplete" or review.get("reviewer_kind") != "model_assisted":
        raise ValueError("judge needs a fresh incomplete model_assisted bundle")
    if review.get("reviewer_id") or review.get("reviewed_at"):
        raise ValueError("fresh review must have no reviewer or review date")
    for field in ("benchmark_id", "benchmark_hash", "corpus_hash", "results_hash"):
        if not isinstance(review.get(field), str) or not review[field]:
            raise ValueError(f"review is missing {field}")
    if review.get("excluded_questions") != []:
        raise ValueError("provisional QASPER judging must cover every frozen question")
    rows = review.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("review has no anonymous answers")
    by_question: OrderedDict[str, list[dict]] = OrderedDict()
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("anonymous answer row has an unexpected schema")
        blind_id, question_id = row["blind_id"], row["question_id"]
        if not isinstance(blind_id, str) or not blind_id or blind_id in ids:
            raise ValueError("blind IDs must be unique, nonempty strings")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question ID must be a nonempty string")
        ids.add(blind_id)
        if row["verdict"] is not None or row["notes"] or row["relevant_source_passage"]:
            raise ValueError("judge needs fresh, undecided answers")
        if row["expected_answerability"] not in {"sufficient", "insufficient"}:
            raise ValueError("unknown QASPER answerability")
        if not isinstance(row["answer"], str) or not isinstance(row["reference_answer"], str):
            raise ValueError("answer and reference must be strings")
        if (not isinstance(row["grader_notes"], list)
                or any(not isinstance(item, str) for item in row["grader_notes"])):
            raise ValueError("QASPER reference evidence must be a list of strings")
        if not isinstance(row["citations"], list) or not isinstance(row["evidence"], list):
            raise ValueError("citations and evidence must be lists")
        by_question.setdefault(question_id, []).append(row)
    sizes = {len(group) for group in by_question.values()}
    if len(sizes) != 1 or next(iter(sizes)) < 2:
        raise ValueError("review must contain a complete paired matrix")
    for group in by_question.values():
        first = group[0]
        if any(any(row[field] != first[field] for field in (
            "question", "expected_answerability", "reference_answer", "grader_notes"
        )) for row in group[1:]):
            raise ValueError("paired answers have different frozen references")
    return review


def _question_groups(review: dict) -> list[list[dict]]:
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for row in review["rows"]:
        grouped.setdefault(row["question_id"], []).append(row)
    return list(grouped.values())


def prompt_for_question(rows: list[dict]) -> str:
    """Include the QASPER annotation and every anonymous submitted evidence span."""
    first = rows[0]
    payload = {
        "question_id": first["question_id"],
        "question": first["question"],
        "qasper_expected_answerability_unreviewed": first["expected_answerability"],
        "qasper_reference_answer_unreviewed": first["reference_answer"],
        "qasper_annotated_evidence_unreviewed": first["grader_notes"],
        "anonymous_candidates": [{
            "blind_id": row["blind_id"],
            "run_status": row["status"],
            "answer": row["answer"],
            "citations": row["citations"],
            "submitted_evidence": row["evidence"],
        } for row in rows],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def plan(review: dict, max_questions: int | None = None) -> dict:
    groups = _question_groups(review)
    if max_questions is not None and max_questions < 1:
        raise ValueError("--max-questions must be positive")
    if max_questions is not None and max_questions >= len(groups):
        raise ValueError("--max-questions smoke must be smaller than the full question count")
    selected = groups[:max_questions] if max_questions is not None else groups
    prompts = [prompt_for_question(rows) for rows in selected]
    lengths = [len(SYSTEM_PROMPT) + len(prompt) for prompt in prompts]
    return {
        "total_questions": len(groups),
        "selected_questions": len(selected),
        "selected_answer_rows": sum(len(rows) for rows in selected),
        "estimated_input_characters": sum(lengths),
        "maximum_request_input_characters": max(lengths),
        "maximum_request_question_id": selected[lengths.index(max(lengths))][0]["question_id"],
        "rough_input_tokens_at_four_chars_per_token": (
            sum(lengths) + 3
        ) // 4,
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "paid_request_count": len(selected),
        "will_be_complete": len(selected) == len(groups),
    }


def enforce_input_bounds(
    summary: dict, *, max_request_chars: int, max_total_chars: int,
) -> None:
    """Fail before any paid request; never silently shorten evidence packets."""
    if max_request_chars < 1 or max_total_chars < 1:
        raise ValueError("input character limits must be positive")
    if summary["maximum_request_input_characters"] > max_request_chars:
        raise ValueError(
            f"question {summary['maximum_request_question_id']} has "
            f"{summary['maximum_request_input_characters']} input characters, exceeding "
            f"--max-input-chars-per-request={max_request_chars}"
        )
    if summary["estimated_input_characters"] > max_total_chars:
        raise ValueError(
            f"selected prompts have {summary['estimated_input_characters']} input characters, "
            f"exceeding --max-total-input-chars={max_total_chars}"
        )


def parse_verdicts(text: str, rows: list[dict]) -> list[dict]:
    """Refuse missing/extra answers and unsupported fabricated passage strings."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"verdicts"}:
        raise ValueError("judge response must contain only verdicts")
    verdicts = data["verdicts"]
    if not isinstance(verdicts, list) or len(verdicts) != len(rows):
        raise ValueError("judge must grade every anonymous answer for the question")
    expected = {row["blind_id"]: row for row in rows}
    received: set[str] = set()
    accepted = []
    for item in verdicts:
        if not isinstance(item, dict) or set(item) != {
            "blind_id", "verdict", "notes", "relevant_source_passage"
        }:
            raise ValueError("judge verdict schema is invalid")
        blind_id = item["blind_id"]
        if blind_id not in expected or blind_id in received:
            raise ValueError("judge returned a duplicate or unknown blind ID")
        received.add(blind_id)
        verdict = item["verdict"]
        notes, passage = item["notes"], item["relevant_source_passage"]
        if verdict not in VERDICTS or not isinstance(notes, str) or not isinstance(passage, str):
            raise ValueError("judge returned an invalid verdict, note, or passage")
        if not notes.strip():
            raise ValueError("every judge verdict needs a nonempty explanation")
        if verdict in {"partial", "fail"} and not passage.strip():
            raise ValueError("partial/fail needs a passage or absence note")
        row = expected[blind_id]
        if (row["status"] != "submitted" or not row["answer"].strip()) and verdict != "fail":
            raise ValueError("unsubmitted or empty answer must be FAIL")
        if verdict == "pass":
            if row["expected_answerability"] == "sufficient" and not any(
                isinstance(evidence, dict)
                and evidence.get("valid") is True
                and isinstance(evidence.get("quote"), str)
                and evidence["quote"].strip()
                and evidence.get("doc_id") in row["citations"]
                for evidence in row["evidence"]
            ):
                raise ValueError("answerable PASS needs candidate's own valid cited evidence")
        quotes = row["grader_notes"] + [
            evidence["quote"] for evidence in row["evidence"]
            if isinstance(evidence, dict) and evidence.get("valid") is True
            and isinstance(evidence.get("quote"), str)
        ]
        if passage.strip() and passage.strip() != ABSENCE_NOTE and not any(
            passage.strip() in quote for quote in quotes
        ):
            raise ValueError("judge passage is not present in supplied source text")
        accepted.append({
            "blind_id": blind_id, "verdict": verdict,
            "notes": notes.strip(), "relevant_source_passage": passage.strip(),
        })
    return accepted


def _checkpoint_header(review: dict, model: str, max_questions: int | None) -> dict:
    groups = _question_groups(review)
    selected = groups[:max_questions] if max_questions is not None else groups
    return {
        "schema_version": CHECKPOINT_VERSION,
        "judge_protocol_version": JUDGE_VERSION,
        "input_review_hash": configuration_hash(review),
        "requested_model": model,
        "max_questions": max_questions,
        "selected_question_ids": [rows[0]["question_id"] for rows in selected],
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "temperature": "unspecified_provider_default",
    }


def _validate_question_record(record: dict, rows: list[dict], index: int) -> dict:
    expected_fields = {
        "schema_version", "index", "question_id", "prompt_sha256", "verdicts",
        "reported_model", "reported_models", "input_tokens", "output_tokens",
        "attempt_count", "previous_hash", "record_hash",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise ValueError("judge checkpoint question record schema is invalid")
    expected_prompt_hash = hashlib.sha256(prompt_for_question(rows).encode()).hexdigest()
    if (record["schema_version"] != QUESTION_RECORD_VERSION
            or type(record["index"]) is not int or record["index"] != index
            or record["question_id"] != rows[0]["question_id"]
            or record["prompt_sha256"] != expected_prompt_hash):
        raise ValueError("judge checkpoint differs from selected frozen question or prompt")
    if (not isinstance(record["reported_model"], str) or not record["reported_model"]
            or not isinstance(record["reported_models"], list)
            or not record["reported_models"]
            or any(not isinstance(item, str) or not item for item in record["reported_models"])
            or record["reported_model"] != record["reported_models"][-1]
            or type(record["attempt_count"]) is not int
            or record["attempt_count"] not in {1, 2}
            or len(record["reported_models"]) != record["attempt_count"]
            or type(record["input_tokens"]) is not int or record["input_tokens"] < 0
            or type(record["output_tokens"]) is not int or record["output_tokens"] < 0):
        raise ValueError("judge checkpoint has invalid model or token metadata")
    canonical = parse_verdicts(json.dumps({"verdicts": record["verdicts"]}), rows)
    if canonical != record["verdicts"]:
        raise ValueError("judge checkpoint verdict text is not normalized")
    unhashed = {key: value for key, value in record.items() if key != "record_hash"}
    if record["record_hash"] != configuration_hash(unhashed):
        raise ValueError("judge checkpoint record hash mismatch")
    return record


def _write_checkpoint_header(path: Path, header: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_question_record(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_checkpoint(
    path: Path, review: dict, *, model: str, max_questions: int | None = None
) -> list[dict]:
    """Validate every saved judgment against the frozen input before resuming."""
    if path.name != "judgments.jsonl" or path.is_symlink() or not path.is_file():
        raise ValueError("resume needs a regular judgments.jsonl checkpoint")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("judge checkpoint is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("judge checkpoint header is malformed") from exc
    if header != _checkpoint_header(review, model, max_questions):
        raise ValueError("judge checkpoint is bound to different input, model, or settings")
    groups = _question_groups(review)
    selected = groups[:max_questions] if max_questions is not None else groups
    if len(lines) - 1 > len(selected):
        raise ValueError("judge checkpoint has more answers than selected questions")
    records = []
    previous_hash = configuration_hash(header)
    for index, line in enumerate(lines[1:]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"judge checkpoint question {index + 1} is malformed") from exc
        _validate_question_record(record, selected[index], index)
        if record["previous_hash"] != previous_hash:
            raise ValueError("judge checkpoint hash chain is broken")
        previous_hash = record["record_hash"]
        records.append(record)
    return records


def _call_and_validate(
    rows: list[dict], call: Callable[[str], tuple[str, dict]], index: int,
    previous_hash: str,
) -> dict:
    prompt = prompt_for_question(rows)
    input_tokens = output_tokens = 0
    reported_models = []
    for attempt in range(2):
        text, metadata = call(prompt)
        if not isinstance(metadata, dict):
            raise ValueError("judge call metadata must be an object")
        in_tokens = metadata.get("input_tokens")
        out_tokens = metadata.get("output_tokens")
        reported_model = metadata.get("model")
        if not isinstance(reported_model, str) or not reported_model:
            raise ValueError("judge must report its actual model/version")
        if (type(in_tokens) is not int or in_tokens < 0
                or type(out_tokens) is not int or out_tokens < 0):
            raise ValueError("judge must report nonnegative token usage")
        input_tokens += in_tokens
        output_tokens += out_tokens
        reported_models.append(reported_model)
        try:
            verdicts = parse_verdicts(text, rows)
            break
        except (ValueError, json.JSONDecodeError):
            if attempt == 1:
                raise
    record = {
        "schema_version": QUESTION_RECORD_VERSION,
        "index": index,
        "question_id": rows[0]["question_id"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "verdicts": verdicts,
        "reported_model": reported_models[-1],
        "reported_models": reported_models,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "attempt_count": len(reported_models),
        "previous_hash": previous_hash,
    }
    record["record_hash"] = configuration_hash(record)
    return _validate_question_record(record, rows, index)


def judge_review(
    review: dict,
    call: Callable[[str], tuple[str, dict]],
    *,
    model: str,
    max_questions: int | None = None,
    prior_records: list[dict] | None = None,
    on_question: Callable[[dict], None] | None = None,
) -> tuple[dict, dict]:
    """Judge all selected groups; only a full matrix can receive complete status."""
    summary = plan(review, max_questions)
    groups = _question_groups(review)[:summary["selected_questions"]]
    judged = copy.deepcopy(review)
    initial_hashes = {
        row["blind_id"]: configuration_hash(_review_content(row)) for row in review["rows"]
    }
    row_by_id = {row["blind_id"]: row for row in judged["rows"]}
    records = list(prior_records or [])
    if len(records) > len(groups):
        raise ValueError("more checkpoint records than selected questions")
    previous_hash = configuration_hash(_checkpoint_header(review, model, max_questions))
    for index, record in enumerate(records):
        _validate_question_record(record, groups[index], index)
        if record["previous_hash"] != previous_hash:
            raise ValueError("prior judge records have a broken hash chain")
        previous_hash = record["record_hash"]
    for index in range(len(records), len(groups)):
        record = _call_and_validate(groups[index], call, index, previous_hash)
        if on_question is not None:
            on_question(record)
        records.append(record)
        previous_hash = record["record_hash"]
    for record in records:
        for item in record["verdicts"]:
            row_by_id[item["blind_id"]].update({
                "verdict": item["verdict"], "notes": item["notes"],
                "relevant_source_passage": item["relevant_source_passage"],
            })
    input_tokens = sum(record["input_tokens"] for record in records)
    output_tokens = sum(record["output_tokens"] for record in records)
    calls = [{
        "question_id": record["question_id"],
        "blind_ids": [item["blind_id"] for item in record["verdicts"]],
        "reported_model": record["reported_model"],
        "input_tokens": record["input_tokens"],
        "output_tokens": record["output_tokens"],
        "attempt_count": record["attempt_count"],
    } for record in records]
    if any(configuration_hash(_review_content(row)) != initial_hashes[row["blind_id"]]
           for row in judged["rows"]):
        raise ValueError("judge changed immutable blind answer content")
    complete = all(row["verdict"] in VERDICTS for row in judged["rows"])
    if complete != summary["will_be_complete"]:
        raise ValueError("judge completion disagrees with planned question count")
    judged["status"] = "complete" if complete else "incomplete"
    judged["reviewer_kind"] = "model_assisted"
    judged["reviewer_id"] = f"anthropic:{model}"
    judged["reviewed_at"] = date.today().isoformat() if complete else ""
    report = {
        "schema_version": "envoybench-provisional-judge-report-v1",
        "judge_protocol_version": JUDGE_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "temperature": "unspecified_provider_default",
        "source_fields_supplied": [
            "qasper_expected_answerability_unreviewed",
            "qasper_reference_answer_unreviewed",
            "qasper_annotated_evidence_unreviewed",
            "anonymous_candidates.answer",
            "anonymous_candidates.citations",
            "anonymous_candidates.submitted_evidence",
        ],
        "source_reference_status": "unreviewed_qasper",
        "review_provenance": "identity-blind model-assisted; not human review",
        "requested_model": model,
        "reported_models": sorted({call["reported_model"] for call in calls
                                   if isinstance(call["reported_model"], str)}),
        "benchmark_id": review["benchmark_id"],
        "benchmark_hash": review["benchmark_hash"],
        "corpus_hash": review["corpus_hash"],
        "results_hash": review["results_hash"],
        "input_review_hash": configuration_hash(review),
        "judged_review_hash": configuration_hash(judged),
        "status": judged["status"],
        "question_count": len(groups),
        "answer_count": sum(row["verdict"] in VERDICTS for row in judged["rows"]),
        "total_question_count": summary["total_questions"],
        "total_answer_count": len(judged["rows"]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_request_count": sum(record["attempt_count"] for record in records),
        "reused_checkpoint_questions": len(prior_records or []),
        "calls": calls,
        "limitations": (
            "QASPER references were not independently validated; the blind packet supplies "
            "one canonical answer and annotated evidence, not every alternative annotation. "
            "A model judge may be wrong or biased. Exact quotes do not prove semantic support. "
            "This is not a human-reviewed held-out promotion result."
        ),
    }
    return judged, report


def _anthropic_call(model: str) -> Callable[[str], tuple[str, dict]]:
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for --execute")
    client = Anthropic().with_options(max_retries=1, timeout=120)

    def call(prompt: str) -> tuple[str, dict]:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(block.text for block in response.content if getattr(block, "text", None))
        return text, {
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    return call


def _write_new(path: Path, value: dict) -> None:
    """Install a complete JSON file atomically, without replacing an existing one."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive install on the same filesystem.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@contextmanager
def _run_lock(directory: Path) -> Iterator[None]:
    """Prevent concurrent resumes from paying twice for the same question."""
    with (directory / ".judge.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if path.is_symlink():
            raise ValueError(f"existing judge output differs from completed result: {path.name}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        # This execution-only count may differ when rebuilding the same report
        # after an interruption between writing report.json and review.json.
        def stable(item: dict) -> dict:
            return {
                key: content for key, content in item.items()
                if key != "reused_checkpoint_questions"
            }
        if not isinstance(existing, dict) or stable(existing) != stable(value):
            raise ValueError(f"existing judge output differs from completed result: {path.name}")
    else:
        _write_new(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True,
                        help="fresh provisional bundle review.json; never blind-key.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for judged review.json and judge-report.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-questions", type=int,
        help="smoke test: judge the first N questions; output stays incomplete",
    )
    parser.add_argument("--execute", action="store_true", help="make paid model calls")
    parser.add_argument("--resume", action="store_true",
                        help="resume an existing validated judgments.jsonl checkpoint")
    parser.add_argument(
        "--max-input-chars-per-request", type=int, default=DEFAULT_MAX_REQUEST_CHARS,
        help="paid-call safety limit; no prompt or evidence is truncated",
    )
    parser.add_argument(
        "--max-total-input-chars", type=int, default=DEFAULT_MAX_TOTAL_INPUT_CHARS,
        help="paid-run safety limit across selected questions",
    )
    args = parser.parse_args()
    review = load_prepared_review(args.review)
    summary = plan(review, args.max_questions)
    try:
        enforce_input_bounds(
            summary, max_request_chars=args.max_input_chars_per_request,
            max_total_chars=args.max_total_input_chars,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_dir.is_symlink():
        parser.error("--output-dir must not be a symlink")
    if args.resume:
        if not args.output_dir.is_dir():
            parser.error("--resume requires an existing output directory")
        if (args.output_dir / "review.json").exists():
            parser.error("judge output already finalized; resume cannot overwrite it")
        prior = load_checkpoint(
            args.output_dir / "judgments.jsonl", review,
            model=args.model, max_questions=args.max_questions,
        )
    else:
        if args.output_dir.exists():
            parser.error("--output-dir must not already exist; use --resume for a checkpoint")
        prior = []
    print(json.dumps({
        "mode": "execute_paid_calls" if args.execute else "dry_run_no_model_calls",
        "model": args.model,
        **summary,
        "checkpointed_questions": len(prior),
        "remaining_paid_requests": summary["selected_questions"] - len(prior),
        "max_input_chars_per_request": args.max_input_chars_per_request,
        "max_total_input_chars": args.max_total_input_chars,
        "note": "Character/4 tokens is a rough preview, not a billed token or dollar estimate.",
    }, indent=2))
    if not args.execute:
        return 0
    call = _anthropic_call(args.model)
    checkpoint = args.output_dir / "judgments.jsonl"
    if not args.resume:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with _run_lock(args.output_dir):
        if args.resume:
            if (args.output_dir / "review.json").exists():
                raise ValueError("judge output finalized during resume")
            prior = load_checkpoint(
                checkpoint, review, model=args.model, max_questions=args.max_questions,
            )
        else:
            header = _checkpoint_header(review, args.model, args.max_questions)
            _write_checkpoint_header(checkpoint, header)
        judged, report = judge_review(
            review, call, model=args.model,
            max_questions=args.max_questions,
            prior_records=prior,
            on_question=lambda record: _append_question_record(checkpoint, record),
        )
        _write_or_verify(args.output_dir / "judge-report.json", report)
        _write_new(args.output_dir / "review.json", judged)
    print(json.dumps({
        "output": str(args.output_dir / "review.json"),
        "report": str(args.output_dir / "judge-report.json"),
        "status": report["status"],
        "input_tokens": report["input_tokens"],
        "output_tokens": report["output_tokens"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
