"""Annotation-assisted passage demonstrations; no inference or runtime changes."""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from pathlib import Path

from src.eval.artifacts import content_hash
from src.eval.reader_improvement import digest, file_hash, inside, load, write
from src.product.passage_scope import PASSAGES_V2, contains, make_scope
from src.research.qasper import (
    QASPER_REVISION,
    _convert_question,
    _doc_id,
    _paper_document,
    _records,
)

ROOT = Path(__file__).resolve().parents[2]
DOC = re.compile(r"\bqasper_[0-9]{4}_[0-9]{4,5}\b")
QID = re.compile(r"\bqasper_(?:train|test|validation)_[0-9a-f]{40}\b")
PAPER = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$")
SKIP = {"corpus", "documents", "snapshots", "archive", "hf-upload", "site-packages", ".git"}
ID_KEYS = {"id", "case_id", "context_id", "question_id", "question_ids",
           "selected_question_ids", "doc_id", "doc_ids",
           "target_doc_ids", "required_doc_ids", "reserved_paper_ids", "paper_ids",
           "source_paper_id", "qasper_paper_id", "selected_target_paper_ids", "question"}
REQUIRED = ["data/sft/qasper-v2/train.jsonl", "data/sft/qasper-v2/val.jsonl",
    "data/sft/qasper-v5/train.jsonl", "data/sft/qasper-v5/val.jsonl",
    "out/research/qasper-scale-sft-v1/train.jsonl", "out/research/qasper-scale-sft-v1/val.jsonl",
    "out/research/qasper-annotation-scale-v1/raw/candidates.jsonl",
    "out/research/qasper-annotation-scale-v1/clean/candidates.jsonl",
    "out/research/qasper-synthetic-reliability-v1/accepted/candidates.jsonl",
    "out/research/qasper-v6-runtime-replayed-20261013/candidates.jsonl",
    "out/diagnostics/qwen-learning-20260918/data/train.jsonl",
    "data/research/qasper-v6-holdout-lock.json",
    "benchmarks/envoybench/data/dev/benchmark.json",
    "benchmarks/envoybench/data/test_candidate/benchmark.json",
    "out/research/qasper-rl-v1/train-v2/benchmark.json",
    "out/research/qasper-annotation-scale-v1/source/benchmark.json",
    "out/research/qasper-train-pilot-v2/questions.json",
    "data/rl/reader-facts-v1/cases.json", "data/rl/reader-facts-v1/source-review.json",
    "data/rl/reader-facts-v1/historical-evaluation-reservations/questions.json"]


def metadata_ids(value):
    """Extract identifier fields only; never return references/answers/source text."""
    docs, questions = set(), set()
    # Old benchmark copies can retain a full corpus reservation and stale parent
    # question_ids. Only their actual selected questions establish evaluation use.
    lock_docs = (value.get("reserved_doc_ids", []) if isinstance(value, dict)
                 and "lock" in value.get("schema_version", "") else [])
    docs.update(d for d in lock_docs if isinstance(d, str) and DOC.fullmatch(d))
    if isinstance(value, dict) and isinstance(value.get("questions"), list):
        value = {"questions": value["questions"]}
    def visit(item, key=None):
        if isinstance(item, dict):
            for k, child in item.items():
                if k not in {"corpus", "documents", "papers", "full_text", "text", "evidence",
                             "gold_evidence", "answer_annotations", "references", "trajectory"}:
                    visit(child, k)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str) and key in ID_KEYS:
            docs.update(DOC.findall(item))
            questions.update(QID.findall(item))
            if PAPER.fullmatch(item):
                docs.add(_doc_id(item))
    visit(value)
    return docs, questions


def inventory(root=ROOT):
    for name in REQUIRED:
        if not (root / name).is_file():
            raise ValueError(f"missing required prior-data inventory source: {name}")
    files, all_docs, all_questions = [], set(), set()
    for area in ("data/sft", "data/rl", "data/research", "out/research", "out/diagnostics",
                 "benchmarks/envoybench", "examples", "release"):
        for base, dirs, names in os.walk(root / area):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and d != "research_reader_v1")
            for name in sorted(names):
                path = Path(base) / name
                relative = str(path.relative_to(root))
                training = (name.endswith(".jsonl") and any(
                    s in name.lower() for s in ("train", "val", "candidate")))
                eval_hint = any(s in relative.lower() for s in (
                    "test", "dev", "eval", "confirm", "holdout", "oracle", "abstention"))
                metadata = name.endswith(".json") and (
                    name.endswith(".manifest.json") or name in {
                        "results.json", "cases.json", "source-review.json"}
                    or (area == "data/research")
                    or name in {"benchmark.json", "questions.json"}
                    or (eval_hint and any(s in name for s in (
                        "benchmark", "questions", "selection", "manifest", "models"))))
                if not training and not metadata:
                    continue
                if training:
                    raw = path.read_text()
                    docs, questions = set(DOC.findall(raw)), set(QID.findall(raw))
                else:
                    docs, questions = metadata_ids(load(path))
                kind = "prepared_training_pool" if training else "evaluation_metadata"
                files.append({"path": relative, "sha256": file_hash(path), "kind": kind,
                              "paper_ids": sorted(docs), "question_ids": sorted(questions)})
                all_docs |= docs
                all_questions |= questions
    scanned = {f["path"] for f in files}
    if not set(REQUIRED) <= scanned:
        raise ValueError("inventory rules missed required sources")
    return {"schema_version": "passage-pilot-exclusions-v3", "files": files,
            "excluded_doc_ids": sorted(all_docs), "excluded_question_ids": sorted(all_questions),
            "rules": "Conservative union of prepared training/candidate conversations and selected "
            "evaluation identifier fields in declared workspace roots. Indexed corpus files, "
            "weights, archives and reserved research_reader_v1 are not read. Prepared data is "
            "excluded even when optimizer use is unproven. No pretraining/remote-history claim.",
            "human_review": None}


def check_registry(registry, root=ROOT):
    if registry.get("schema_version") != "passage-pilot-exclusions-v3":
        raise ValueError("legacy or unknown exclusion registry; regenerate with structured "
                         "benchmark training-pool and RL role/rejection coverage")
    if not set(REQUIRED) <= {f["path"] for f in registry["files"]}:
        raise ValueError("incomplete exclusion registry")
    docs, questions = set(), set()
    for row in registry["files"]:
        path = inside(root, row["path"])
        if file_hash(path) != row["sha256"]:
            raise ValueError("prior inventory source changed")
        training = path.suffix == ".jsonl" and any(
            s in path.name.lower() for s in ("train", "val", "candidate"))
        if training:
            raw = path.read_text()
            actual = set(DOC.findall(raw)), set(QID.findall(raw))
        else:
            actual = metadata_ids(load(path))
        if actual != (set(row["paper_ids"]), set(row["question_ids"])):
            raise ValueError("registry identifiers differ from their recorded sources")
        docs.update(row["paper_ids"])
        questions.update(row["question_ids"])
    if sorted(docs) != registry["excluded_doc_ids"] or sorted(questions) != registry[
            "excluded_question_ids"]:
        raise ValueError("registry union changed")


def paired_windows(text, annotation):
    """Propose controlled scopes; exact-string absence is not a semantic label."""
    evidence = annotation["evidence"]
    if len(text) < 6400 or len(evidence) != 1:
        return None
    span = evidence[0]
    if not 40 <= span["end"] - span["start"] <= 1000:
        return None
    start = min(max(0, span["start"] - 300), len(text) - 1600)
    present = {"start": start, "end": start + 1600}
    if not contains([present], span["start"], span["end"]):
        return None
    answer_parts = annotation["extractive_spans"]
    candidates = [{"start": i, "end": i + 1600}
                  for i in range(0, len(text) - 1599, 1600)]
    candidates = [w for w in candidates if (w["end"] <= present["start"] or
        w["start"] >= present["end"]) and not any(
        part.casefold() in text[w["start"]:w["end"]].casefold() for part in answer_parts)]
    if len(candidates) < 2:
        return None
    # Shared distractor plus one replaced window; ordered source ranges in both states.
    shared, absent = candidates[-1], candidates[0]
    return {"present": sorted([present, shared], key=lambda w: w["start"]),
            "missing": sorted([absent, shared], key=lambda w: w["start"])}


def select(rows, split, registry, count):
    excluded_docs, excluded_qs = set(registry["excluded_doc_ids"]), set(
        registry["excluded_question_ids"])
    pool, rejected = [], Counter()
    for row in rows:
        if _doc_id(row["id"]) in excluded_docs:
            rejected["prior_paper_exposure"] += 1
            continue
        document, locations = _paper_document(row, split)
        document["metadata"]["source_revision"] = QASPER_REVISION
        for qa in _records(row.get("qas")):
            q = _convert_question(qa, document, locations, split)
            if q["id"] in excluded_qs or q["conversion_issues"] or q[
                    "ambiguous_text_evidence_count"] or q["expected_answerability"] != "sufficient":
                rejected["identity_or_annotation_mapping"] += 1
                continue
            annotations = q["answer_annotations"]
            if (any(a["answer_type"] != "extractive" for a in annotations)
                    or len({a["answer_text"].casefold() for a in annotations}) != 1):
                rejected["not_single_agreed_extractive_answer"] += 1
                continue
            a = annotations[0]
            if (not 4 <= len(a["answer_text"]) <= 200 or len(a["extractive_spans"]) > 3
                    or any(len(p) < 3 for p in a["extractive_spans"])):
                rejected["answer_length_or_complexity"] += 1
                continue
            windows = paired_windows(document["text"], a)
            if windows is None:
                rejected["paired_scope_feasibility"] += 1
                continue
            pool.append((hashlib.sha256(("42:" + q["id"]).encode()).hexdigest(),
                         q, document, windows))
    selected, used = [], set()
    for _, question, document, windows in sorted(pool, key=lambda r: r[0]):
        if document["doc_id"] in used:
            continue
        used.add(document["doc_id"])
        selected.append((question, document, windows))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible distinct {split} papers; need {count}")
    return selected, {"eligible_questions": len(pool), "selected_papers": len(selected),
                      "rejection_counts": dict(rejected)}


def check_prepared(prepared):
    manifest = load(prepared / "manifest.json")
    for name, key in (("cases.json", "cases_sha256"), ("selection.json", "selection_sha256"),
                      ("exclusions.json", "exclusions_sha256")):
        if file_hash(prepared / name) != manifest[key]:
            raise ValueError("prepared cases, selection or exclusion registry changed")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(inside(prepared, name)) != expected:
            raise ValueError("prepared source snapshot changed")
    cases = load(prepared / "cases.json")
    registry = load(prepared / "exclusions.json")
    check_registry(registry)
    if (len(cases) != 36 or len({c["case_id"] for c in cases}) != 36
            or Counter(c["role"] for c in cases) != {"train": 24, "development": 12}):
        raise ValueError("expected 24 train and 12 development cases")
    papers = {role: {c["doc_id"] for c in cases if c["role"] == role}
              for role in ("train", "development")}
    if papers["train"] & papers["development"]:
        raise ValueError("train/development paper overlap")
    if len(papers["train"]) != 12 or len(papers["development"]) != 6:
        raise ValueError("expected twelve training and six development papers")
    pairs = {}
    for c in cases:
        if not QID.fullmatch(c["question_id"]) or c["case_id"] != (
                c["question_id"] + "--" + c["scope_answerability"]):
            raise ValueError("invalid case identity")
        expected_split = "train" if c["role"] == "train" else "validation"
        if c["source_split"] != expected_split or not c["question_id"].startswith(
                "qasper_" + expected_split + "_"):
            raise ValueError("source split differs from data role")
        snapshot = inside(prepared, c["snapshot"])
        names = [str((snapshot / name).relative_to(prepared.resolve())) for name in (
            "manifest.json", "corpus/" + c["doc_id"] + ".json")]
        if not all(name in manifest["source_sha256"] for name in names):
            raise ValueError("case snapshot is not hash-bound")
        from src.research.agent import load_snapshot
        _, docs = load_snapshot(snapshot)
        windows = c["scope"]["windows"]
        if (len(windows) != 2 or any(w["end"] - w["start"] != 1600 for w in windows)
                or c["scope"] != make_scope(docs, windows, version=PASSAGES_V2)
                or set(docs) != {c["doc_id"]}):
            raise ValueError("case scope differs from pinned two-window source")
        pairs.setdefault(c["question_id"], []).append(c)
        if (c["doc_id"] in registry["excluded_doc_ids"]
                or c["question_id"] in registry["excluded_question_ids"]):
            raise ValueError("previously exposed paper/question in pilot data")
    if len(pairs) != 18:
        raise ValueError("one question per selected paper required")
    for pair in pairs.values():
        if (len(pair) != 2 or {c["scope_answerability"] for c in pair} != {"present", "missing"}
                or any(pair[0][key] != pair[1][key] for key in (
                    "question", "doc_id", "snapshot", "role", "annotation_for_review_only"))
                or len({(w["start"], w["end"]) for c in pair
                        for w in c["scope"]["windows"]}) != 3):
            raise ValueError("invalid paired evidence states/shared window")
    return manifest, cases


def reviewed_rejections(review, depth=0):
    """Retain all earlier source-review exclusions when proposing replacements."""
    if review is None:
        return []
    if depth >= 20:
        raise ValueError("too many source-review replacement rounds")
    parent = inside(ROOT, review["prepared_path"])
    _, cases = check_prepared(parent)
    if (review["cases_sha256"] != file_hash(parent / "cases.json")
            or review.get("reviewer_kind") != "model_assisted"
            or not review.get("reviewer_identity")):
        raise ValueError("unbound data-review rejections")
    parents = {c["question_id"]: c for c in cases}
    rows = review["rows"]
    if not rows:
        raise ValueError("empty data-review rejection round")
    for row in rows:
        if (row["question_id"] not in parents or not row.get("reason")
                or row["doc_id"] != parents[row["question_id"]]["doc_id"]):
            raise ValueError("rejection does not identify a reviewed parent case")
    previous = load(parent / "selection.json").get("data_review_rejections")
    return reviewed_rejections(previous, depth + 1) + rows


def prepare(registry_path, output, rejections_path=None):
    from benchmarks.envoybench.build_data import _load_source
    registry = load(registry_path)
    check_registry(registry)
    selection_registry = {**registry, "excluded_doc_ids": list(registry["excluded_doc_ids"]),
                          "excluded_question_ids": list(registry["excluded_question_ids"])}
    rejections = load(rejections_path) if rejections_path else None
    for row in reviewed_rejections(rejections):
        selection_registry["excluded_doc_ids"].append(row["doc_id"])
        selection_registry["excluded_question_ids"].append(row["question_id"])
    index = load(ROOT / "benchmarks/envoybench/data/splits.json")["source"]
    source_rows, provenance = {}, {}
    for split in ("train", "validation"):
        arrow = (Path.home() / ".cache/huggingface/datasets/allenai___qasper/qasper/0.0.0" /
                 QASPER_REVISION / f"qasper-{split}.arrow")
        if not arrow.is_file():
            raise ValueError("pinned local QASPER Arrow missing; no download fallback")
        source_rows[split], provenance[split] = _load_source(
            split, index["splits"][split], arrow.parent)
    selected, selection = {}, {}
    for split, count in (("train", 12), ("validation", 6)):
        selected[split], selection[split] = select(
            source_rows[split], split, selection_registry, count)
    papers = [{d["doc_id"] for _, d, _ in rows} for rows in selected.values()]
    if papers[0] & papers[1]:
        raise ValueError("training/development papers overlap")
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for split, rows in selected.items():
        for question, document, windows in rows:
            qid = question["id"]
            snapshot = output / "snapshots" / qid
            corpus = snapshot / "corpus"
            corpus.mkdir(parents=True)
            write(corpus / (document["doc_id"] + ".json"), document)
            write(snapshot / "manifest.json", {"schema_version": "research-snapshot-v1",
                "source_dataset": "allenai/qasper", "source_revision": QASPER_REVISION,
                "source_split": split, "corpus_hash": content_hash(corpus)})
            for state in ("present", "missing"):
                scope = make_scope({document["doc_id"]: document}, windows[state],
                                   version=PASSAGES_V2)
                cases.append({"case_id": qid + "--" + state, "question_id": qid,
                    "role": "train" if split == "train" else "development",
                    "source_split": split, "question": question["source_question"],
                    "doc_id": document["doc_id"], "snapshot": str(snapshot.relative_to(output)),
                    "scope": scope, "scope_answerability": state,
                    "original_qasper_answerability": question["expected_answerability"],
                    "annotation_for_review_only": question["answer_annotations"][0],
                    "review_status": "pending_source_review", "training_ready": False})
    write(output / "cases.json", cases)
    write(output / "exclusions.json", registry)
    write(output / "selection.json", {"seed": 42, "rank": "sha256(42:<question ID>)",
        "source": provenance, "splits": selection, "model_outputs_used": False,
        "data_review_rejections": rejections,
        "rejections_sha256": file_hash(rejections_path) if rejections_path else None})
    review = {"reviewer_kind": None, "reviewer_identity": None, "reviewed_at": None,
              "cases_sha256": file_hash(output / "cases.json"), "rows": [{
                  "case_id": c["case_id"], "case_sha256": digest(c), "decision": None,
                  "source_support_reviewed": None, "notes": "", "final_action": None,
                  "supporting_spans": []} for c in cases]}
    write(output / "review-template.json", review)
    write(output / "manifest.json", {"schema_version": "passage-pilot-prepared-v1",
        "status": "pending_source_review", "cases_sha256": file_hash(output / "cases.json"),
        "selection_sha256": file_hash(output / "selection.json"),
        "exclusions_sha256": file_hash(output / "exclusions.json"),
        "source_sha256": {str(p.relative_to(output)): file_hash(p)
                          for p in sorted((output / "snapshots").rglob("*.json"))},
        "human_review": None, "new_model_calls": 0, "training_performed": False})
    return {"cases": len(cases), "train": 24, "development": 12,
            "status": "pending_source_review", "new_model_calls": 0}
