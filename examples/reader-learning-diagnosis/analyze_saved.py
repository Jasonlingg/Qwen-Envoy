"""Reproduce a development-only diagnostic without models or execution of actions."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "examples/base-v5-reader-comparison"
BUNDLE = ROOT / "benchmarks/research_reader_v1"
HERE = Path(__file__).resolve().parent


def analyze():
    hashes = {}

    def load(path):
        raw = path.read_bytes()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    cases = load(BUNDLE / "cases/development.json")
    case_map = {case["case_id"]: case for case in cases}
    original = load(DATA / "per-question-results.json")
    annotations = load(HERE / "annotations.json")
    labels = {(r["case_id"], r["arm"]): r for r in annotations["rows"]}
    expected = {(case["case_id"], arm) for case in cases for arm in ("base", "v5")}
    rows = original["rows"]
    if (len(rows) != 24 or {(r["case_id"], r["arm"]) for r in rows} != expected
            or len(labels) != 24 or len(annotations["rows"]) != 24 or set(labels) != expected):
        raise ValueError("complete matching development rows and manual annotations required")
    output, sources = [], {}
    for row in rows:
        key = row["case_id"], row["arm"]
        case = case_map[row["case_id"]]
        packet = load(DATA / "live" / f"{row['case_id']}--{row['arm']}.packet.json")
        version = case["paper_version"]
        if version not in sources:
            sources[version] = load(BUNDLE / "sources" / f"{version}.json")
        source = sources[version]
        if (packet["question"] != case["question"] or packet["execution"] != "docker_only"
                or packet["model_identity"]["adapter_applied"] != (row["arm"] == "v5")
                or packet["paper_assignment"]["arxiv_id"] != version
                or row["kind"] != case["kind"]):
            raise ValueError("packet identity differs from the matched development question")
        for item in packet["evidence"]:
            start, end = item["start"], item["end"]
            if (item["doc_id"] != source["doc_id"] or type(start) is not int
                    or type(end) is not int or not 0 <= start < end <= len(source["text"])
                    or item["quote"] != source["text"][start:end]):
                raise ValueError("packet quote differs from its pinned source")
        output.append({"case_id": row["case_id"], "arm": row["arm"], "kind": row["kind"],
            "question": case["question"], "original_model_assisted_verdict": row["verdict"],
            "original_review_notes": row["notes"], "diagnosis": labels[key],
            "candidate_claims": packet["candidate_claims"],
            "submitted_evidence": packet["evidence"], "research_gaps": packet["research_gaps"],
            "requests": row["requests"], "execution_error_steps": row["execution_error_steps"],
            "submission_rejection_steps": row["submission_rejection_steps"],
            "actions": [{"step": t["step"], "action": t["action"],
                         "source_inspection": t.get("source_inspection", False)}
                        for t in packet["trajectory"]]})
    summary = {}
    for arm in ("base", "v5"):
        selected = [r for r in output if r["arm"] == arm]
        tags = Counter(tag for r in selected for tag in
                       [r["diagnosis"]["primary"], *r["diagnosis"]["secondary"]])
        summary[arm] = {"episodes": len(selected), "requests": sum(r["requests"] for r in selected),
            "recorded_execution_error_episodes": sum(bool(r["execution_error_steps"])
                                                       for r in selected),
            "original_answerable_passes": sum(r["kind"] != "target_gap" and
                r["original_model_assisted_verdict"] == "pass" for r in selected),
            "original_gap_passes": sum(r["kind"] == "target_gap" and
                r["original_model_assisted_verdict"] == "pass" for r in selected),
            "primary_counts": dict(Counter(r["diagnosis"]["primary"] for r in selected)),
            "nonexclusive_tag_counts": dict(sorted(tags.items()))}
    return {"schema_version": "reader-learning-diagnosis-v1", "scope": "saved development only",
            "review_kind": "implementing_assistant_manual_diagnosis; not a new semantic grade",
            "human_review": "not_performed", "rewards_computed": False,
            "historical_grades_changed": False, "reserved_evaluation_opened": False,
            "model_requests": 0, "gpu_activations": 0,
            "tag_count_note": "Manual diagnostic tags can overlap; not new pass/fail labels.",
            "input_sha256": dict(sorted(hashes.items())), "summary": summary, "rows": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
