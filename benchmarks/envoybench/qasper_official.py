"""Official QASPER Answer F1 on the selected 40-question executable-agent study.

Scoring uses only the standard library and packaged original annotations:
  python -m benchmarks.envoybench.qasper_official score --run-dir PATH --output FILE

Only exporting references from the pinned local Arrow cache needs ``datasets``:
  python -m benchmarks.envoybench.qasper_official export-references --arrow PATH

This is token-overlap Answer F1, not semantic support review, not the MuSiQue
reward, and not an evaluation on the full official QASPER test split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCES = ROOT / "release/qasper-agent-study/references.json"
VENDOR = Path(__file__).with_name("vendor") / "qasper"
EVALUATOR_COMMIT = "e996b6c7b1b5f95d9308a74e3586416c6e780df1"
EVALUATOR_SHA256 = "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"
EVALUATOR_URL = (
    f"https://github.com/allenai/qasper-led-baseline/blob/{EVALUATOR_COMMIT}/scripts/evaluator.py"
)
SOURCE_REVISION = "13b496d2a5359329b110e3419628de3cf791843b"
SOURCE_ARROW_SHA256 = "9462ccbd216bb2e6d9fba739a4fd349dd7da1e15116b09e0c4be85302d2b6ba4"
SOURCE_ROWS_SHA256 = "5854e52c9b84a7a4d26f795dc9d4bc03668201350d4ad00c7d2eb2be9c959c65"
SELECTION_MANIFEST_SHA256 = "d57e8121e234dae6fefd7dae50f40bb38e320da63e1036c3d50fcef1eea8c2d9"
REFERENCES_SHA256 = "e0b9c8dac3bac1e472183daadfd4ea8eebc3fe97493ed9ff1fba033e739efa63"
REFERENCE_VERSION = "qasper-agent-study-references-v1"
SCORE_VERSION = "qasper-agent-study-score-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def official_evaluator():
    """Load the unmodified, hash-pinned upstream implementation without extras."""
    path = VENDOR / "evaluator.py"
    _require(file_sha256(path) == EVALUATOR_SHA256, "official evaluator SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("qasper_pinned_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _records(columns: dict) -> list[dict]:
    """Reconstruct Arrow's columnar lists without choosing or changing annotations."""
    lengths = {len(values) for values in columns.values()}
    _require(len(lengths) == 1, "unequal source annotation column lengths")
    return [dict(zip(columns, values)) for values in zip(*columns.values())]


def export_references(arrow: Path, output: Path = DEFAULT_REFERENCES) -> dict:
    """Export every source answer annotation for the already selected 40 IDs."""
    _require(file_sha256(arrow) == SOURCE_ARROW_SHA256, "pinned QASPER Arrow SHA256 mismatch")
    from datasets import Dataset  # Optional, export only; never imported while scoring.

    source_rows = list(Dataset.from_file(str(arrow)))
    canonical = hashlib.sha256()
    for row in sorted(source_rows, key=lambda item: item["id"]):
        canonical.update(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )
    _require(canonical.hexdigest() == SOURCE_ROWS_SHA256, "pinned QASPER row hash mismatch")
    selection_path = ROOT / "release/envoybench-v0.1/run/manifest.json"
    _require(
        file_sha256(selection_path) == SELECTION_MANIFEST_SHA256,
        "frozen question selection manifest changed",
    )
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {qid.removeprefix("qasper_test_"): qid for qid in manifest["question_ids"]}
    questions = {}
    for paper in source_rows:
        for qa in _records(paper["qas"]):
            source_id = qa["question_id"]
            if source_id not in selected:
                continue
            _require(source_id not in questions, "duplicate original source question")
            questions[source_id] = {
                "question_id": selected[source_id],
                "source_question_id": source_id,
                "source_paper_id": paper["id"],
                "paper_title": paper["title"],
                "question": qa["question"],
                "answers": [
                    {"annotation_id": item["annotation_id"], "answer": item["answer"]}
                    for item in _records(qa["answers"])
                ],
            }
    _require(set(questions) == set(selected), "selected question missing from original QASPER")
    bundle = {
        "schema_version": REFERENCE_VERSION,
        "source": {
            "dataset": "allenai/qasper",
            "revision": SOURCE_REVISION,
            "split": "test",
            "url": f"https://huggingface.co/datasets/allenai/qasper/tree/{SOURCE_REVISION}",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "citation": "Dasigi et al. (2021), A Dataset of Information-Seeking Questions and "
            "Answers Anchored in Research Papers, NAACL.",
            "citation_url": "https://aclanthology.org/2021.naacl-main.365/",
            "arrow_sha256": SOURCE_ARROW_SHA256,
            "rows_sha256": SOURCE_ROWS_SHA256,
            "source_paper_count": len(source_rows),
            "export_note": "All original answer objects and annotation IDs retained "
            "in source order "
            "for selected questions; Arrow columns restructured into records. Worker "
            "metadata and unselected questions/paper bodies omitted. No one-best "
            "answer selection, annotation repair, or new judgment.",
        },
        "selection": {
            "manifest_sha256": SELECTION_MANIFEST_SHA256,
            **{
                field: manifest[field]
                for field in (
                    "benchmark_id",
                    "benchmark_hash",
                    "corpus_hash",
                    "question_ids",
                    "split_status",
                )
            },
            "question_count": len(selected),
            "scope": "40 selected QASPER test questions; known-paper code-execution agent study; "
            "not the full official test split",
        },
        "questions": [questions[source_id] for source_id in selected],
    }
    load_gold(bundle)
    _write(output, bundle)
    return bundle


def load_gold(bundle: dict) -> tuple[dict, dict]:
    """Use official annotation precedence, including every reference answer."""
    _require(bundle.get("schema_version") == REFERENCE_VERSION, "unknown reference schema")
    questions = bundle["questions"]
    by_id = {row["question_id"]: row for row in questions}
    source_ids = {row["source_question_id"] for row in questions}
    selected = bundle["selection"]["question_ids"]
    _require(
        len(questions) == len(by_id) == len(source_ids) == len(selected)
        and set(by_id) == set(selected),
        "reference question coverage is inconsistent",
    )
    original_data = {}
    for row in questions:
        _require(bool(row["answers"]), "source question has no reference annotations")
        original_data.setdefault(row["source_paper_id"], {"qas": []})["qas"].append(
            {
                "question_id": row["source_question_id"],
                "answers": row["answers"],
            }
        )
    gold = official_evaluator().get_answers_and_evidence(original_data, text_evidence_only=False)
    return gold, by_id


def score_predictions(bundle: dict, predictions: dict[str, str]) -> tuple[dict, list[dict]]:
    """Score submitted strings keyed by protocol ID; absent predictions score zero.

    An empty submitted answer is still a prediction: upstream token F1 returns
    zero even for empty/empty. Missing predictions remain in the overall mean
    but upstream excludes them from its answer-type subgroups.
    """
    gold, references = load_gold(bundle)
    _require(set(predictions) <= set(references), "prediction has an unknown question ID")
    _require(
        all(isinstance(value, str) for value in predictions.values()),
        "submitted predictions must be strings",
    )
    evaluator = official_evaluator()
    # Empty evidence is only a placeholder for the upstream evaluate() signature.
    # Its Evidence F1 result is deliberately discarded, never reported.
    official_predictions = {
        references[qid]["source_question_id"]: {"answer": answer, "evidence": []}
        for qid, answer in predictions.items()
    }
    evaluated = evaluator.evaluate(gold, official_predictions)
    rows = []
    type_denominators = Counter()
    for qid in bundle["selection"]["question_ids"]:
        source_id = references[qid]["source_question_id"]
        source_answers = gold[source_id]
        missing = qid not in predictions
        if missing:
            answer_f1, matched_type, matched_index = 0.0, None, None
        else:
            scores = [
                evaluator.token_f1_score(predictions[qid], ref["answer"]) for ref in source_answers
            ]
            matched_index = max(range(len(scores)), key=scores.__getitem__)
            answer_f1 = scores[matched_index]
            matched_type = source_answers[matched_index]["type"]
            type_denominators[matched_type] += 1
        rows.append(
            {
                "question_id": qid,
                "source_question_id": source_id,
                "predicted_answer": predictions.get(qid),
                "prediction_missing": missing,
                "answer_f1": answer_f1,
                "matched_answer_type": matched_type,
                "matched_reference_index": matched_index,
                "reference_count": len(source_answers),
                "reference_answer_types": [ref["type"] for ref in source_answers],
            }
        )
    metrics = {
        "answer_f1": evaluated["Answer F1"],
        "answer_f1_percent": evaluated["Answer F1"] * 100,
        "answer_f1_by_type": evaluated["Answer F1 by type"],
        "answer_f1_by_type_denominators": {
            kind: type_denominators[kind] for kind in evaluated["Answer F1 by type"]
        },
        "missing_predictions": evaluated["Missing predictions"],
        "submitted_predictions": len(predictions),
        "question_count": len(gold),
    }
    return metrics, rows


def score_run(
    run_dir: Path,
    *,
    references: Path = DEFAULT_REFERENCES,
    output: Path | None = None,
) -> dict:
    """Score each declared model against the same frozen selected source IDs."""
    run_dir, references = Path(run_dir), Path(references)
    manifest_path, results_path = run_dir / "manifest.json", run_dir / "results.json"
    _require(
        file_sha256(references) == REFERENCES_SHA256, "frozen original references SHA256 mismatch"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    bundle = json.loads(references.read_text(encoding="utf-8"))
    load_gold(bundle)
    selection = bundle["selection"]
    question_ids = selection["question_ids"]
    _require(len(question_ids) == 40, "this study requires the selected 40 questions")
    _require(
        len(manifest["question_ids"]) == 40 and set(manifest["question_ids"]) == set(question_ids),
        "run does not declare the same 40 source questions",
    )
    for field in ("benchmark_id", "benchmark_hash", "corpus_hash"):
        _require(manifest[field] == selection[field], f"run/reference {field} mismatch")
    model_keys = [model["key"] for model in manifest["models"]]
    _require(
        bool(model_keys) and len(model_keys) == len(set(model_keys)), "invalid model declarations"
    )
    by_pair = {}
    for row in results:
        pair = (row["model_key"], row["question_id"])
        _require(
            pair[0] in model_keys and pair[1] in question_ids and pair not in by_pair,
            "unknown or duplicate model/question result",
        )
        for field in ("run_id", "benchmark_id", "benchmark_hash", "corpus_hash"):
            _require(row[field] == manifest[field], f"result/manifest {field} mismatch")
        _require(
            row["status"] in {"submitted", "no_submission", "error", "escalated"},
            "unknown result status",
        )
        by_pair[pair] = row
    models, scored_rows = {}, []
    for model in model_keys:
        submitted = {
            qid: row["predicted_answer"]
            for (key, qid), row in by_pair.items()
            if key == model and row["status"] == "submitted"
        }
        models[model], rows = score_predictions(bundle, submitted)
        models[model]["recorded_episodes"] = sum(key == model for key, _ in by_pair)
        for row in rows:
            saved = by_pair.get((model, row["question_id"]))
            row.update({"model_key": model, "status": saved["status"] if saved else "not_recorded"})
        scored_rows.extend(rows)
    report = {
        "schema_version": SCORE_VERSION,
        "metric": "Official QASPER Answer F1",
        "scope": selection["scope"],
        **{
            field: manifest[field]
            for field in ("run_id", "benchmark_id", "benchmark_hash", "corpus_hash")
        },
        "question_ids": question_ids,
        "results_hash": configuration_hash({"results": results}),
        "results_sha256": file_sha256(results_path),
        "prediction_policy": "literal_submitted_answer_v1",
        "models": models,
        "rows": scored_rows,
        "provenance": {
            "evaluator_url": EVALUATOR_URL,
            "evaluator_commit": EVALUATOR_COMMIT,
            "evaluator_sha256": EVALUATOR_SHA256,
            "evaluator_license": "Apache-2.0",
            "scorer_sha256": file_sha256(Path(__file__)),
            "references_sha256": file_sha256(references),
            "source": bundle["source"],
            "selection": selection,
            "run_manifest_sha256": file_sha256(manifest_path),
            "run_status": manifest.get("status"),
            "models": manifest["models"],
        },
        "limitations": [
            "Official token-overlap metric on 40 selected source questions, not full QASPER test; "
            "known-paper prompts and code-execution conditions differ from "
            "published full-test systems.",
            "All original reference annotations are scored and the maximum answer F1 is used. "
            "Literal submitted strings receive only official SQuAD-style normalization; no "
            "protocol abstention paraphrase is rewritten to Unanswerable.",
            "No-submission, error, escalated, or unrecorded episodes are missing predictions: zero "
            "in the 40-question overall mean. Empty submitted strings also score zero. Official "
            "answer-type means omit missing predictions and choose the first maximizing reference.",
            "No Evidence F1 is reported: character spans were not converted to QASPER paragraphs. "
            "Answer F1 is not semantic support review or a validated model-improvement gate.",
        ],
    }
    if output is not None:
        _write(Path(output), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export-references", help="read pinned local Arrow; no download")
    export.add_argument("--arrow", type=Path, required=True)
    export.add_argument("--output", type=Path, default=DEFAULT_REFERENCES)
    score = subparsers.add_parser(
        "score", help="score saved runs offline with the standard library"
    )
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    score.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export-references":
            result = export_references(args.arrow, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "questions": len(result["questions"]),
                        "sha256": file_sha256(args.output),
                    }
                )
            )
        else:
            result = score_run(args.run_dir, references=args.references, output=args.output)
            print(
                json.dumps(
                    result
                    if args.output is None
                    else {
                        "output": str(args.output),
                        "models": result["models"],
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
