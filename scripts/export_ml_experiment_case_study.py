"""Export bounded September 30 development evidence; never execute saved actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reproduce_case_study import recorded_error_episode, reproduce  # noqa: E402
from src.eval.artifacts import content_hash  # noqa: E402
from src.product.ml_experiment_measurements import Measurement  # noqa: E402

SELECTED = {
    "wrong_span": "qasper_test_7ae95716977d39d96e871e552c35ca0753115229",
    "shallow_answer": "qasper_test_0b5a7ccf09810ff5a86162d502697d16b3536249",
    "false_refusal": "qasper_test_a71fb012631e6a8854d5945b6d0ab2ab8e7b7ee6",
    "supported_control": "qasper_test_2d5d0b0c54105717bf48559b914fefd0c94964a6",
    "refusal_control": "qasper_test_fd556a038c36abc88a800d9d4f2cfa0aef6f5aba",
}


def build_export(root: Path = ROOT) -> dict:
    report = reproduce(root / "release/envoybench-v0.1")
    manifest = json.loads((root / "release/envoybench-v0.1/run/manifest.json").read_text())
    paths = {"results": "release/envoybench-v0.1/run/results.json",
             "review": "release/envoybench-v0.1/review-model-assisted/review.json",
             "key": "release/envoybench-v0.1/review-prepared/blind-key.json",
             "prior_retrieval": "release/qasper-agent-study/retrieval-baseline/official-score.json"}
    data = {key: json.loads((root / name).read_text()) for key, name in paths.items()}
    by_pair = {(r["model_key"], r["question_id"]): (i, r)
               for i, r in enumerate(data["results"])}
    reviews = {r["blind_id"]: r for r in data["review"]["rows"]}
    review_pairs = {(a["system"], a["question_id"]): reviews[a["blind_id"]]
                    for a in data["key"]["assignments"]}
    corpus = root / "benchmarks/envoybench/data/test_candidate/corpus"
    if content_hash(corpus) != manifest["corpus_hash"]:
        raise ValueError("Frozen corpus hash mismatch; rebuild the pinned corpus first")
    cases = {}
    for label, question_id in SELECTED.items():
        index, row = by_pair["qwen_v5", question_id]
        review = review_pairs["qwen_v5", question_id]
        excerpts = []
        for span in row["predicted_evidence"]:
            path = corpus / (span["doc_id"] + ".json")
            document = json.loads(path.read_text())
            if not 0 <= span["start"] < span["end"] <= len(document["text"]):
                raise ValueError("Saved evidence has invalid offsets")
            excerpts.append({**span,
                "document_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "excerpt": document["text"][span["start"]:min(span["end"], span["start"] + 250)],
                "excerpt_truncated": span["end"] - span["start"] > 250})
        cases[label] = {"question_id": question_id, "question": row["question"],
            "answer": row["predicted_answer"], "status": row["status"],
            "steps": row["steps"], "result_pointer": f"/{index}",
            "actions": [step["action"] for step in row["trajectory"]],
            "observation_prefixes": [step["observation"][:220] for step in row["trajectory"]],
            "observation_prefix_limit": 220, "observations_may_be_truncated": True,
            "submitted_evidence": excerpts, "existing_review": {
                "kind": "model_assisted", "verdict": review["verdict"],
                "expected_answerability": review["expected_answerability"],
                "notes": review["notes"]},
            "selection": "Post-hoc illustrative development case; not representative or held out"}
        if len(json.dumps(cases[label], ensure_ascii=False)) > 4_000:
            raise ValueError("Selected case exceeds evidence attachment bound")
    measurements, timing = {}, {}
    for key in ("qwen_base", "qwen_v5"):
        rows = [r for r in data["results"] if r["model_key"] == key]
        valid_times = [r["duration_seconds"] for r in rows if r["duration_seconds"] > 0]
        timing[key] = {"positive_duration_count": len(valid_times), "scheduled_cases": len(rows),
                       "zero_placeholder_count": len(rows) - len(valid_times),
                       "terminal_failure_ids": [r["question_id"] for r in rows
                                                if r["status"] != "submitted"]}
        metrics = {"provisional_pass_rate": ("percent", "pass count / 40 * 100",
                        report["models"][key]["provisional_passes"] / len(rows) * 100),
                   "error_episodes": ("count", "episodes with recorded error or no submission",
                        float(sum(recorded_error_episode(r) or r["status"] != "submitted"
                                  for r in rows))),
                   "human_supported_rate": (
                       "percent", "human-supported pass count / 40 * 100", None),
                   "latency": ("seconds", "mean full episode latency", None),
                   "cost": ("USD", "all-in session allocation per question", None)}
        reasons = {"human_supported_rate": "Independent source-checked human review is absent",
                   "latency": "Incomplete base timing and no matched lifecycle timing scope",
                   "cost": "Loading, idle, storage, cleanup and host accounting are incomplete"}
        measurements[key] = {}
        for metric_id, (unit, aggregation, value) in metrics.items():
            measured = {"metric_id": metric_id, "unit": unit, "aggregation": aggregation,
                "population": "40 September 30 QASPER development questions", "value": value,
                "sample_count": len(rows) if value is not None else None,
                "coverage": "complete" if value is not None else "unknown"}
            if value is None:
                measured["unknown_reason"] = reasons[metric_id]
            if metric_id == "provisional_pass_rate":
                measured.update(numerator=report["models"][key]["provisional_passes"],
                                denominator=len(rows), review={"kind": "model_assisted"})
            measurements[key][metric_id] = Measurement.model_validate(measured).model_dump(
                mode="json")
    prior = data["prior_retrieval"]
    parents = {**paths, "manifest": "release/envoybench-v0.1/run/manifest.json",
               "original_protocol": "benchmarks/envoybench/PROVISIONAL_COMPARISON_2026_09_30.md"}
    return {"schema_version": "ml-experiment-qwen-development-export-v1",
        "parents": {key: {"path": name,
                         "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                    for key, name in parents.items()},
        "recipe": "scripts/export_ml_experiment_case_study.py; fixed SELECTED IDs; v5 actions "
                  "as data; observation prefixes 220 chars; submitted evidence prefixes 250 chars; "
                  "counts recomputed from frozen rows and bound existing reviews",
        "lineage": "Exporter checks parents/corpus; MCP verifies only selected export bytes",
        "summary": {"models": report["models"], "timing": timing,
                    "review_kind": "model_assisted", "human_review": "pending",
                    "scope": "Already-seen selected development questions; no improvement claim"},
        "run_identity": {key: manifest[key] for key in ["question_ids", "corpus_hash", "models",
            "created_at_utc", "completed_at_utc", "git_commit", "system_prompt_sha256",
            "tool_preamble_sha256", "implementation_sha256", "seed_note"]},
        "original_decision_rule": (root / parents["original_protocol"]).read_text().split(
            "**Decision rule.** ", 1)[1].strip(),
        "cases": cases, "measurements": measurements,
        "prior_retrieval": {"metric": prior["metric"], "models": prior["models"],
                            "meaning": "Token overlap; semantic support remains unreviewed"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_export()
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(args.output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "model_requests": 0}))


if __name__ == "__main__":
    main()
