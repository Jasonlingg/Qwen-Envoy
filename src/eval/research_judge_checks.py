"""Offline fixture integrity and judge-input export, never a semantic judge or reward."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

VERSIONS = {"2307.08691v1", "2306.00978v6"}
GROUPS = {"critical_negative": 12, "complete": 8, "partial": 4,
          "valid_gap": 4, "blanket_abstention": 4}
INSPECTION = "assumed_complete_inspection_for_synthetic_fixture_not_an_executed_trace"
REPO = Path(__file__).resolve().parents[2]
CONTRACTS = {f"examples/reader-learning-diagnosis/{name}.json" for name in (
    "judge-validation-plan", "reward-proposal")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError("fixture path escapes root")
    path = root / name
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file() or path.stat().st_size > 2_000_000):
        raise ValueError("missing, linked or oversized fixture file")
    return path


def _load(root, name):
    return json.loads(_file(root, name).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _spans(spans, text):
    _require(bool(spans), "source spans required")
    for s in spans:
        a, b = s["start"], s["end"]
        _require(type(a) is int and type(b) is int and 0 <= a < b <= len(text)
                 and s["text"] == text[a:b], "source span must match exact characters")


def check_bundle(root, *, repo=REPO):
    """Check artifact bindings and authored label arithmetic, not label truth."""
    root, repo = Path(root), Path(repo)
    files = _load(root, "freeze-lock.json")["files"]
    required = {"manifest.json", "questions.json", "cases.json", "labels.json",
                "reviews/human-template.json", "reviews/source-review.json"}
    required |= {f"sources/{v}.json" for v in VERSIONS}
    _require(set(files) == required, "exact frozen fixture inventory required")
    for name, digest in files.items():
        _require(sha256(_file(root, name)) == digest, f"frozen fixture changed: {name}")
    m = _load(root, "manifest.json")
    _require(m["schema_version"] == "research-judge-checks-v1"
             and m["status"] == "fixtures_only_judge_not_run"
             and m["judge_identity"] is None and m["actual_results"] is None
             and m["human_review"] == "not_performed"
             and m["model_inference_requests"] == 0 and m["gpu_activations"] == 0
             and m["training_approved"] is False,
             "fixtures cannot assert judge or training results")
    _require(set(m["prior_contracts"]) == CONTRACTS, "unexpected prior contract paths")
    for name, digest in m["prior_contracts"].items():
        _require(sha256(_file(repo, name)) == digest, "prior contract changed")
    _require(sha256(_file(repo, "benchmarks/research_reader_v1/freeze-lock.json"))
             == m["reader_bundle_lock_sha256"], "reader lock changed")
    plan = _load(repo, "examples/reader-learning-diagnosis/judge-validation-plan.json")
    _require({v.split("v")[0] for v in VERSIONS}
             == set(plan["judge_validation_partition_paper_ids"]), "wrong development partition")
    _require(set(m["sources"]) == VERSIONS, "only two development validation papers allowed")
    reader_lock = _load(repo, "benchmarks/research_reader_v1/freeze-lock.json")["files"]
    texts = {}
    for version, s in m["sources"].items():
        name = f"sources/{version}.json"
        _require(s["path"] == name and s["sha256"] == files[name]
                 and files[name] == reader_lock[name], "source binding differs")
        doc = _load(root, name)
        _require(doc["metadata"]["arxiv_id"] == version, "source identity differs")
        texts[version] = doc["text"]
    qs, rows, labels = (_load(root, name) for name in (
        "questions.json", "cases.json", "labels.json"))
    _require(len(qs) == 7, "seven fixed question rubrics required")
    for q in qs.values():
        _require(q["paper_version"] in VERSIONS and q["task"] in {"answerable", "target_gap"}
                 and q["question"].strip(), "invalid question")
        facts = q["requested_facts"]
        _require(bool(facts) and [f["fact_id"] for f in facts]
                 == [f"f{i+1}" for i in range(len(facts))]
                 and all(f["criterion"].strip() for f in facts), "invalid fixed requested facts")
        _spans(q["reference_contexts"], texts[q["paper_version"]])
    ids = [r["case_id"] for r in rows]
    _require(len(rows) == 32 and ids == [f"j{i:02d}" for i in range(1, 33)]
             and [label["case_id"] for label in labels] == ids, "32 aligned unique cases required")
    _require(Counter(label["group"] for label in labels) == GROUPS, "predeclared groups differ")
    _require(Counter(label["critical_type"] for label in labels
                     if label["group"] == "critical_negative")
             == {t: 2 for t in plan["case_requirements"]["critical_types_with_two_examples_each"]},
             "two of each critical type required")
    for row, label in zip(rows, labels):
        q = qs[row["question_id"]]
        _require(row["inspection_precondition"] == INSPECTION, "not a real execution trace")
        candidate = row["candidate"]
        gap = candidate["gap_report"]
        _require(isinstance(candidate["answer"], str) and candidate["answer"].strip()
                 and len(candidate["answer"]) <= 20_000, "candidate answer must be bounded text")
        _require(gap is None or (isinstance(gap, dict)
                 and set(gap) == {"missing", "follow_up_query"}
                 and all(isinstance(v, str) and 0 < len(v.strip()) <= 20_000
                         for v in gap.values())), "gap fields must be bounded nonempty text")
        _spans(candidate["evidence"], texts[q["paper_version"]])
        judgments = label["fact_judgments"]
        _require([j["fact_id"] for j in judgments] == [f["fact_id"] for f in q["requested_facts"]]
                 and all(type(j["correct_and_supported"]) is bool for j in judgments)
                 and type(label["all_material_assertions_supported"]) is bool,
                 "complete explicit fact labels and material gate required")
        covered = sum(j["correct_and_supported"] for j in judgments)
        c = covered / len(judgments)
        expected = (float(c == 1) if q["task"] == "target_gap" else c)
        if not label["all_material_assertions_supported"]:
            expected = 0
        _require(type(label["expected_score"]) in {int, float}
                 and label["expected_score"] == expected and label["rationale"].strip(),
                 "authored score must follow fixed facts and material gate")
        group = label["group"]
        _require((group != "complete" or expected == 1)
                 and (group != "partial" or 0 < expected < 1)
                 and (group not in {"critical_negative", "blanket_abstention"} or expected == 0),
                 "score inconsistent with fixture group")
        _require((q["task"] == "target_gap") == (group == "valid_gap" or
                 label["critical_type"] in {"invented target result",
                    "valid target gap plus unsupported partial literature claim"}),
                 "question task differs from planned fixture group")
        if group == "valid_gap" or label["critical_type"] == (
                "valid target gap plus unsupported partial literature claim"):
            _require(gap and set(gap) == {"missing", "follow_up_query"}
                     and all(isinstance(v, str) and v.strip() for v in gap.values()),
                     "gap report and useful query required")
        _require(group != "valid_gap" or expected == 1, "valid gap must score one")
        if group == "blanket_abstention":
            _require(covered == 0 and row["candidate"]["answer"] == "Unanswerable"
                     and gap is None, "blanket abstention must supply no facts")
    review = _load(root, "reviews/source-review.json")
    _require(review["review_kind"] == "model_assisted" and review["reviewer_identity"]
             and review["human_review"] == "not_performed" and not review["open_material_findings"]
             and review["reviewed_file_sha256"] == {n: h for n, h in files.items()
                 if n != "reviews/source-review.json"}
             and [r["case_id"] for r in review["rows"]] == ids
             and all(r["verdict"] == "accepted_provisional" and r["rationale"]
                     for r in review["rows"]), "complete independent provisional review required")
    human = _load(root, "reviews/human-template.json")
    _require(human["reviewer_identity"] is None and human["reviewed_at"] is None
             and [r["case_id"] for r in human["rows"]] == ids
             and all(r["verdict"] is None and r["notes"] is None for r in human["rows"]),
             "human review must remain blank")
    return {"status": "mechanical_checks_passed_not_judge_validation", "cases": len(rows),
            "groups": dict(GROUPS), "judge_results": None, "human_review": "not_performed",
            "lock_sha256": sha256(_file(root, "freeze-lock.json"))}


def judge_inputs(root):
    """Allowlist grader inputs. Reference criteria are not candidate expected verdicts.

    This is for a future judge, never a reader policy. No model calls are made.
    Public file separation cannot prevent a caller from deliberately opening labels.
    """
    check_bundle(root)
    root = Path(root)
    questions = _load(root, "questions.json")
    sources = {v: _load(root, f"sources/{v}.json")["text"] for v in VERSIONS}
    def spans(items):
        return [{k: s[k] for k in ("start", "end", "text")} for s in items]

    exported = []
    for row in _load(root, "cases.json"):
        q, c = questions[row["question_id"]], row["candidate"]
        exported.append({"case_id": row["case_id"], "paper_version": q["paper_version"],
            "question": q["question"],
            "requested_facts": [{k: f[k] for k in ("fact_id", "criterion")}
                                for f in q["requested_facts"]],
            "reference_contexts": spans(q["reference_contexts"]),
            "source_text": sources[q["paper_version"]],
            "candidate": {"answer": c["answer"], "evidence": spans(c["evidence"]),
                          "gap_report": None if c["gap_report"] is None else {
                              k: c["gap_report"][k] for k in ("missing", "follow_up_query")}},
            "inspection_precondition": row["inspection_precondition"]})
    return exported
