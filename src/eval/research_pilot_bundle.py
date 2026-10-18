"""Freeze a cached development pilot without fetching sources or calling models."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.eval.artifacts import configuration_hash
from src.product.arxiv_ids import validate_arxiv_id
from src.product.arxiv_library import ArxivLibrary
from src.research.agent import load_snapshot

SCHEMA = "research-workflow-pilot-bundle-v1"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offline_fetch(*args, **kwargs):
    raise RuntimeError("Frozen pilot forbids source network access; prepare cached papers first")


def offline_library(path: Path) -> ArxivLibrary:
    return ArxivLibrary(path, fetch=offline_fetch, fetch_pdf=offline_fetch)


def source_identity(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    paths = [*root.glob("src/eval/research_pilot*.py"),
             *root.glob("src/product/*.py"), *root.glob("src/policies/*.py"),
             *root.glob("src/env/*.py"), *root.glob("src/research/*.py"),
             root / "scripts/run_research_workflow_pilot.py", root / "pyproject.toml"]
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "implementation_files": hashes, "implementation_sha256": configuration_hash(hashes),
            "python": platform.python_version(), "platform": platform.platform()}


def validate_questions(data: dict) -> None:
    if data.get("schema_version") != "research-workflow-pilot-v1":
        raise ValueError("Unknown pilot question schema")
    papers = data.get("papers")
    questions = data.get("questions")
    if (not isinstance(papers, list) or not papers
            or not isinstance(questions, list) or not questions):
        raise ValueError("Pilot requires papers and questions")
    versions = [validate_arxiv_id(p["arxiv_id"]) for p in papers]
    if any("v" not in v for v in versions) or len(set(versions)) != len(versions):
        raise ValueError("Pilot papers must be unique version-pinned IDs")
    ids = set()
    for q in questions:
        if (not isinstance(q.get("id"), str) or q["id"] in ids
                or not isinstance(q.get("question"), str) or not q["question"].strip()
                or len(q["question"]) > 4_000 or not isinstance(q.get("arxiv_ids"), list)
                or not 1 <= len(q["arxiv_ids"]) <= 5
                or not set(q["arxiv_ids"]) <= set(versions)):
            raise ValueError("Invalid or duplicate pilot question")
        ids.add(q["id"])
    if set(data.get("review_strata", {})) != ids:
        raise ValueError("Every question requires an evaluator-only review stratum")
    if set(data["review_strata"].values()) - {"answerable", "deployment_insufficient"}:
        raise ValueError("Unknown review stratum")


def prepare(question_file: Path, source_root: Path, destination: Path, repo: Path) -> dict:
    if destination.exists():
        raise ValueError("Preparation destination already exists; choose a fresh directory")
    data = json.loads(question_file.read_text(encoding="utf-8"))
    validate_questions(data)
    source = offline_library(source_root)
    documents = []
    for paper in data["papers"]:
        document = source.document(paper["arxiv_id"])
        digest = hashlib.sha256(document["text"].encode()).hexdigest()
        if paper.get("text_sha256") != digest:
            raise ValueError(f"Question set text hash differs for {paper['arxiv_id']}")
        documents.append({"arxiv_id": paper["arxiv_id"], "text_sha256": digest,
                          "document_sha256": sha256(source.papers_dir / paper["arxiv_id"]
                                                    / "document.json"),
                          "coverage": document["metadata"]["coverage"]})
    destination.mkdir(parents=True)
    library = offline_library(destination / "library")
    try:
        for paper in documents:
            shutil.copytree(source.papers_dir / paper["arxiv_id"],
                            library.papers_dir / paper["arxiv_id"])
        view = library.view([p["arxiv_id"] for p in documents])
        snapshot, _ = load_snapshot(view)
        write_json(destination / "questions.json", data)
        manifest = {"schema_version": SCHEMA, "status": "prepared_not_executed",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "questions_sha256": sha256(destination / "questions.json"),
                    "corpus_hash": snapshot["corpus_hash"], "papers": documents,
                    "source_identity": source_identity(repo), "seed": 42,
                    "actual_model_requests": 0, "semantic_review": "not_performed",
                    "plan": [{"question_id": q["id"], "arm": arm}
                             for i, q in enumerate(data["questions"])
                             for arm in (("direct", "delegated") if i % 2 == 0
                                         else ("delegated", "direct"))]}
        manifest["bundle_sha256"] = configuration_hash(manifest)
        write_json(destination / "manifest.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(destination)
        raise


def load_bundle(path: Path) -> tuple[dict, dict, ArxivLibrary]:
    manifest = json.loads((path / "manifest.json").read_text())
    digest = manifest.pop("bundle_sha256", None)
    if manifest.get("schema_version") != SCHEMA or configuration_hash(manifest) != digest:
        raise ValueError("Pilot manifest identity changed")
    manifest["bundle_sha256"] = digest
    if sha256(path / "questions.json") != manifest["questions_sha256"]:
        raise ValueError("Pilot questions changed")
    data = json.loads((path / "questions.json").read_text())
    validate_questions(data)
    library = offline_library(path / "library")
    for paper in manifest["papers"]:
        document = library.document(paper["arxiv_id"])
        if (sha256(library.papers_dir / paper["arxiv_id"] / "document.json")
                != paper["document_sha256"] or hashlib.sha256(document["text"].encode()).hexdigest()
                != paper["text_sha256"]):
            raise ValueError("Frozen paper changed")
    snapshot, _ = load_snapshot(library.view([p["arxiv_id"] for p in manifest["papers"]]))
    if snapshot["corpus_hash"] != manifest["corpus_hash"]:
        raise ValueError("Frozen corpus identity changed")
    return manifest, data, library
