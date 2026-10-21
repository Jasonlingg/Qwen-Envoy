"""Offline integrity checks for source-reviewed paper questions; no semantic grading."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from src.product.arxiv_ids import validate_arxiv_id

SPLITS = ("development", "evaluation")
KINDS = {"supported", "condition_binding", "target_gap"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("bundle paths must stay inside the root")
    path = root / name
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
        raise ValueError("linked bundle paths are not accepted")
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError(f"missing or oversized bundle file: {name}")
    return path


def _load(root, name):
    return json.loads(_path(root, name).read_text())


def check_bundle(root):
    """Validate frozen bytes, split isolation and exact spans, never answer quality."""
    root = Path(root).resolve(strict=True)
    lock = _load(root, "freeze-lock.json")
    files = lock["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("freeze lock must list bundle files")
    for name, expected in files.items():
        if sha256(_path(root, name)) != expected:
            raise ValueError(f"frozen bundle changed: {name}")
    manifest = _load(root, "manifest.json")
    if (manifest["schema_version"] != "research-reader-eval-v1"
            or manifest["status"] != "frozen_pending_human_review_not_run"
            or manifest["human_review"] != "not_performed"
            or manifest["training_approved"] is not False
            or manifest["model_inference_requests"] != 0
            or manifest["model_visible_fields"] != ["question", "paper_version"]):
        raise ValueError("this bundle is preparation, not training or evaluated answers")
    required = {"manifest.json", "protocol.json", "reviews/human-answer-template.json",
                "reviews/source-review.json"}
    documents = {}
    for version, source in manifest["sources"].items():
        if validate_arxiv_id(version) != version or "v" not in version:
            raise ValueError("source versions must be pinned")
        name = f"sources/{version}.json"
        if source["path"] != name or source["sha256"] != files.get(name):
            raise ValueError("source path or hash differs")
        doc = _load(root, name)
        if (doc["metadata"]["arxiv_id"] != version
                or doc["metadata"]["source_url"] != f"https://arxiv.org/abs/{version}"
                or doc["metadata"]["coverage"] not in {"html_paragraphs", "pdf_text"}
                or doc["metadata"]["coverage"] != source["coverage"]
                or not isinstance(doc["text"], str) or not 0 < len(doc["text"]) <= 250_000):
            raise ValueError("invalid source identity or full-text coverage")
        required.add(name)
        documents[version] = doc
    if len(documents) != 8 or len({v.split("v")[0] for v in documents}) != 8:
        raise ValueError("eight distinct paper families required")
    ids, papers = {}, {}
    for split in SPLITS:
        spec = manifest["splits"][split]
        expected_names = {key: f"{key}/{split}.json" for key in ("cases", "references")}
        if any(spec[key] != name for key, name in expected_names.items()):
            raise ValueError("unexpected split paths")
        required.update(expected_names.values())
        cases, refs = (_load(root, spec[key]) for key in ("cases", "references"))
        ids[split] = [case["case_id"] for case in cases]
        papers[split] = {case["paper_version"].split("v")[0] for case in cases}
        if (len(cases) != 12 or len(set(ids[split])) != 12
                or len(papers[split]) != 4
                or any(sum(c["paper_version"].split("v")[0] == p for c in cases) != 3
                       for p in papers[split])
                or [ref["case_id"] for ref in refs] != ids[split]
                or any(sum(c["kind"] == kind for c in cases) != 4 for kind in KINDS)):
            raise ValueError("split needs twelve aligned cases with four of each kind")
        for case, ref in zip(cases, refs):
            if (not re.fullmatch(r"[a-z][a-z0-9-]+", case["case_id"])
                    or not isinstance(case["question"], str)
                    or not 0 < len(case["question"]) <= 1500):
                raise ValueError("invalid case identity or bounded question")
            doc = documents[case["paper_version"]]
            contexts = ref["source_contexts"]
            context_ids = {context["context_id"] for context in contexts}
            if not contexts or len(context_ids) != len(contexts):
                raise ValueError("unique source contexts required")
            for context in contexts:
                start, end = context["start"], context["end"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(doc["text"])
                        or context["text"] != doc["text"][start:end]):
                    raise ValueError("reference must be an exact bounded source span")
            criteria = ref["facts"]
            if (not criteria or len({f["fact_id"] for f in criteria}) != len(criteria)
                    or any(not f["criterion"].strip()
                           or not set(f["context_ids"]) <= context_ids for f in criteria)):
                raise ValueError("unique facts and existing context references required")
    if set(ids["development"]) & set(ids["evaluation"]) or papers["development"] & papers[
            "evaluation"]:
        raise ValueError("question IDs and paper families must be split disjoint")
    if set(files) != required:
        raise ValueError("freeze lock differs from required bundle files")
    review = _load(root, "reviews/source-review.json")
    if (review["review_kind"] != "model_assisted"
            or review["human_review"] != "not_performed" or not review["reviewer_identity"]
            or {row["case_id"] for row in review["rows"]} != set(ids["evaluation"])
            or len(review["rows"]) != 12
            or any(row["verdict"] != "accepted" or not row["notes"]
                   for row in review["rows"])):
        raise ValueError("complete attributed provisional source review required")
    template = _load(root, "reviews/human-answer-template.json")
    if (any(template[key] is not None for key in ("reviewer_kind", "reviewer_identity",
            "reviewed_at", "model_identity", "run_manifest_sha256"))
            or len(template["rows"]) != 24
            or {(r["split"], r["case_id"]) for r in template["rows"]}
            != {(s, cid) for s in SPLITS for cid in ids[s]}
            or any(row[key] is not None for row in template["rows"] for key in (
                "answer_verdict", "condition_error", "unsupported_numeric", "notes"))):
        raise ValueError("human answer review template must stay blank; copy it to review a run")
    protocol = _load(root, "protocol.json")
    if (protocol["execution"]["paid_execution_approved"] is not False
            or "decision_rule" not in protocol or "split_policy" not in protocol):
        raise ValueError("preparation cannot authorize execution")
    reviewed = review["reviewed_file_sha256"]
    if reviewed != {name: digest for name, digest in files.items()
                   if name != "reviews/source-review.json"}:
        raise ValueError("source review differs from frozen questions, rubric or sources")
    return {"status": "mechanical_checks_passed_not_semantic_grades",
            "split_counts": {s: len(ids[s]) for s in SPLITS}, "source_count": len(documents),
            "human_review": "not_performed", "lock_sha256": sha256(root / "freeze-lock.json")}


def model_cases(root, split):
    """Allowlist model-facing fields; graders and gap labels never enter prompts."""
    if split not in SPLITS:
        raise ValueError("explicit development or evaluation split required")
    check_bundle(root)
    root = Path(root)
    return [{key: row[key] for key in ("question", "paper_version")}
            for row in _load(root, f"cases/{split}.json")]
