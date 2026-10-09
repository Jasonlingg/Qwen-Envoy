"""Materialize the frozen EnvoyBench QASPER question sets and paper corpora.

The checked-in split manifest contains IDs and hashes, not copies of QASPER's
paper text. This command verifies the pinned source before writing run inputs.
It performs no inference and does not produce benchmark scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from src.eval.artifacts import content_hash
from src.research.benchmark import validate_benchmark
from src.research.qasper import (
    CODE_EXEC_CONVERTER_VERSION,
    CONVERTER_VERSION,
    QASPER_CONFIG,
    QASPER_DATASET,
    QASPER_REVISION,
    _convert_question,
    _paper_document,
    _records,
    _to_code_exec_question,
)

SCHEMA_VERSION = "envoybench-splits-v1"
MANIFEST_VERSION = "envoybench-data-manifest-v1"
SELECTION_VERSION = "sha256-ranked-balanced-paper-disjoint-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_rows_sha256(rows: Iterable[dict]) -> str:
    """Hash source records independent of Arrow serialization/layout."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["id"]):
        digest.update(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _load_source(split: str, source: dict, arrow_dir: Path | None) -> tuple[list[dict], dict]:
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned QASPER dependency: pip install -e '.[qasper]'"
        ) from exc

    if arrow_dir is None:
        arrow_dir = (
            Path.home()
            / ".cache/huggingface/datasets/allenai___qasper/qasper/0.0.0"
            / QASPER_REVISION
        )
    arrow_path = arrow_dir / f"qasper-{split}.arrow"
    if arrow_path.is_file():
        if sha256_file(arrow_path) != source["arrow_sha256"]:
            raise ValueError(f"{split} Arrow SHA-256 differs from frozen source")
        rows = list(Dataset.from_file(str(arrow_path)))
    else:
        # A different datasets version may serialize Arrow differently. The
        # canonical row hash below is the authority for source equivalence.
        dataset = load_dataset(
            QASPER_DATASET, QASPER_CONFIG, revision=QASPER_REVISION, split=split
        )
        rows = list(dataset)
    actual_rows_hash = canonical_rows_sha256(rows)
    if actual_rows_hash != source["rows_sha256"]:
        raise ValueError(f"{split} QASPER rows differ from frozen source")
    return rows, {
        "source_split": split,
        "source_revision": QASPER_REVISION,
        "source_rows_sha256": actual_rows_hash,
        "source_arrow_sha256": source["arrow_sha256"],
        "paper_count": len(rows),
    }


def _question_catalogue(rows: list[dict], split: str) -> dict[str, tuple[dict, dict]]:
    catalogue = {}
    for row in rows:
        document, locations = _paper_document(row, split)
        for qa in _records(row.get("qas")):
            converted = _convert_question(qa, document, locations, split)
            if converted["id"] in catalogue:
                raise ValueError(f"Duplicate QASPER question ID: {converted['id']}")
            catalogue[converted["id"]] = (converted, document)
    return catalogue


def select_candidate_ids(
    catalogue: dict[str, tuple[dict, dict]],
    *,
    seed: int,
    per_answerability: int,
    excluded_paper_ids: set[str],
) -> list[str]:
    """Select a predeclared 50/50 stress set with one question per unseen paper."""
    eligible = []
    for question_id, (question, _) in catalogue.items():
        if (
            question["source_paper_id"] in excluded_paper_ids
            or question["conversion_issues"]
            or question["ambiguous_text_evidence_count"]
            or question["expected_answerability"] not in {"sufficient", "insufficient"}
        ):
            continue
        rank = hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
        eligible.append((rank, question_id, question))
    eligible.sort()
    chosen = {"sufficient": [], "insufficient": []}
    used_papers: set[str] = set()
    for _, question_id, question in eligible:
        label = question["expected_answerability"]
        paper_id = question["source_paper_id"]
        if len(chosen[label]) == per_answerability or paper_id in used_papers:
            continue
        chosen[label].append(question_id)
        used_papers.add(paper_id)
        if all(len(items) == per_answerability for items in chosen.values()):
            break
    if any(len(items) != per_answerability for items in chosen.values()):
        raise ValueError("Too few eligible distinct test papers for frozen selection")
    return sorted(chosen["sufficient"] + chosen["insufficient"])


def _reference_answer(annotation: dict) -> dict:
    return {
        "annotation_id": annotation["annotation_id"],
        "answer_type": annotation["answer_type"],
        "answer_text": annotation["answer_text"],
        "unanswerable": annotation["unanswerable"],
        "evidence": annotation["evidence"],
    }


def _prepare_question(converted: dict) -> dict:
    question = _to_code_exec_question(converted)
    if question is None or converted["conversion_issues"]:
        raise ValueError(f"Frozen question {converted['id']} no longer passes conversion")
    question.update(
        {
            "source_question_id": converted["source_question_id"],
            "source_paper_id": converted["source_paper_id"],
            "source_annotation_ids": [
                item["annotation_id"] for item in converted["answer_annotations"]
            ],
            "source_question": converted["source_question"],
            "reference_answers": [
                _reference_answer(item) for item in converted["answer_annotations"]
            ],
            "gold_evidence": converted["gold_evidence"],
            "gold_status": "QASPER annotations; independent EnvoyBench review pending",
        }
    )
    return question


def _build_split(
    name: str,
    spec: dict,
    source: dict,
    rows: list[dict],
    source_audit: dict,
    output: Path,
) -> dict:
    split = spec["source_split"]
    catalogue = _question_catalogue(rows, split)
    selected_ids = spec["question_ids"]
    if len(selected_ids) != len(set(selected_ids)) or sorted(selected_ids) != selected_ids:
        raise ValueError(f"{name} IDs must be unique and sorted")
    if set(selected_ids) - catalogue.keys():
        raise ValueError(f"{name} has question IDs absent from pinned QASPER source")
    if name == "test_candidate":
        expected = select_candidate_ids(
            catalogue,
            seed=spec["selection_seed"],
            per_answerability=spec["per_answerability"],
            excluded_paper_ids=set(spec["excluded_prior_paper_ids"]),
        )
        if selected_ids != expected:
            raise ValueError("Candidate test IDs do not match the frozen selection algorithm")

    source_paper_ids = {row["id"] for row in rows}
    excluded_corpus_ids = set(spec["corpus_excluded_paper_ids"])
    if not excluded_corpus_ids <= source_paper_ids:
        raise ValueError(f"{name} excludes a nonexistent paper")
    corpus_paper_ids = source_paper_ids - excluded_corpus_ids
    selected_paper_ids = {catalogue[qid][0]["source_paper_id"] for qid in selected_ids}
    if not selected_paper_ids <= corpus_paper_ids:
        raise ValueError(f"{name} target paper was excluded from its corpus")
    if name == "test_candidate":
        forbidden = set(spec["excluded_prior_paper_ids"])
        if excluded_corpus_ids != forbidden:
            raise ValueError("Candidate corpus must exclude all previously used target papers")
        if selected_paper_ids & forbidden:
            raise ValueError("Candidate test includes a previously used target paper")

    split_dir = output / name
    if split_dir.exists():
        raise FileExistsError(
            f"{split_dir} already exists; remove only after preserving any local runs"
        )
    split_dir.mkdir(parents=True)
    corpus_dir = split_dir / "corpus"
    corpus_dir.mkdir()
    paper_manifest = []
    for row in sorted(rows, key=lambda item: item["id"]):
        if row["id"] not in corpus_paper_ids:
            continue
        document, _ = _paper_document(row, split)
        path = corpus_dir / f"{document['doc_id']}.json"
        _write_json(path, document)
        paper_manifest.append(
            {
                "doc_id": document["doc_id"],
                "source_paper_id": row["id"],
                "sha256": sha256_file(path),
            }
        )
    corpus_hash = content_hash(corpus_dir)
    questions = [_prepare_question(catalogue[qid][0]) for qid in selected_ids]
    documents = {
        paper["doc_id"]: json.loads(
            (corpus_dir / f"{paper['doc_id']}.json").read_text()
        )
        for paper in paper_manifest
    }
    for question in questions:
        for evidence in question["gold_evidence"]:
            text = documents[evidence["doc_id"]]["text"]
            if text[evidence["start"] : evidence["end"]] != evidence["text"]:
                raise ValueError(f"Evidence span moved in {question['id']}")
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": f"envoybench-{name.replace('_', '-')}-v1",
        "status": spec["status"],
        "domain": "Executable known-paper evidence QA over QASPER research papers",
        "corpus_hash": corpus_hash,
        "reserved_doc_ids": [paper["doc_id"] for paper in paper_manifest],
        "training_exclusion": (
            "All evaluation target papers are excluded from Envoy QASPER SFT; "
            "inspect data/manifest.json for split-level evidence and limitations."
        ),
        "converter_version": CODE_EXEC_CONVERTER_VERSION,
        "source_dataset": QASPER_DATASET,
        "source_revision": QASPER_REVISION,
        "source_split": split,
        "selection_seed": spec["selection_seed"],
        "questions": questions,
    }
    validate_benchmark(benchmark)
    benchmark_path = split_dir / "benchmark.json"
    _write_json(benchmark_path, benchmark)
    manifest = {
        "schema_version": "research-snapshot-v1",
        "benchmark_id": benchmark["benchmark_id"],
        "parser_version": CONVERTER_VERSION,
        "source_dataset": QASPER_DATASET,
        "source_config": QASPER_CONFIG,
        "source_revision": QASPER_REVISION,
        **source_audit,
        "split_status": spec["status"],
        "human_review_status": "pending",
        "papers": paper_manifest,
        "paper_count": len(paper_manifest),
        "selected_target_paper_ids": sorted(selected_paper_ids),
        "selected_question_ids": selected_ids,
        "corpus_hash": corpus_hash,
        "benchmark_sha256": sha256_file(benchmark_path),
        "status": "complete",
    }
    _write_json(split_dir / "manifest.json", manifest)
    validate_benchmark(benchmark, manifest)
    return {
        "benchmark": f"{name}/benchmark.json",
        "corpus": f"{name}/corpus",
        "manifest": f"{name}/manifest.json",
        "benchmark_sha256": manifest["benchmark_sha256"],
        "corpus_hash": corpus_hash,
        "source_rows_sha256": source["rows_sha256"],
        "question_count": len(questions),
        "target_paper_count": len(selected_paper_ids),
        "corpus_paper_count": len(paper_manifest),
        "answerability": {
            label: sum(q["expected_answerability"] == label for q in questions)
            for label in ("sufficient", "insufficient")
        },
        "status": spec["status"],
        "human_review_status": "pending",
    }


def materialize(splits_path: Path, output: Path, arrow_dir: Path | None = None) -> dict:
    spec = json.loads(splits_path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected {SCHEMA_VERSION}")
    source = spec["source"]
    if source["dataset"] != QASPER_DATASET or source["revision"] != QASPER_REVISION:
        raise ValueError("Frozen QASPER source does not match converter pin")
    if set(spec["splits"]) != {"dev", "test_candidate"}:
        raise ValueError("Both dev and test_candidate splits are required")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in spec["splits"]):
        raise FileExistsError("Split output exists; choose a new --output directory")

    loaded = {}
    for source_split in {item["source_split"] for item in spec["splits"].values()}:
        loaded[source_split] = _load_source(
            source_split, source["splits"][source_split], arrow_dir
        )
    # Training paper leakage is checked against the complete official train
    # split, not only the much smaller set of SFT conversations.
    train_rows, train_audit = _load_source("train", source["splits"]["train"], arrow_dir)
    train_ids = {row["id"] for row in train_rows}
    train_ids_hash = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
    if train_ids_hash != spec["training_paper_ids_sha256"]:
        raise ValueError("Frozen QASPER training paper set changed")
    dev_ids = {row["id"] for row in loaded["validation"][0]}
    test_ids = {row["id"] for row in loaded["test"][0]}
    if train_ids & (dev_ids | test_ids) or dev_ids & test_ids:
        raise ValueError("Official QASPER source has paper overlap across splits")
    audited_v5_ids = set(spec["audit"]["v5_sft_training_paper_ids"])
    if not audited_v5_ids <= train_ids:
        raise ValueError("Audited v5 SFT paper is absent from QASPER train")
    audited_v5_hash = hashlib.sha256(
        "\n".join(sorted(audited_v5_ids)).encode()
    ).hexdigest()
    if audited_v5_hash != spec["audit"]["v5_sft_training_paper_ids_sha256"]:
        raise ValueError("Audited v5 SFT paper list changed")
    test_catalogue = _question_catalogue(loaded["test"][0], "test")
    prior_test_ids = spec["audit"]["prior_test_question_ids"]
    if (
        len(prior_test_ids) != len(set(prior_test_ids))
        or set(prior_test_ids) - test_catalogue.keys()
    ):
        raise ValueError("Prior-use audit contains duplicate or missing QASPER test IDs")
    mapped_prior_papers = {
        test_catalogue[qid][0]["source_paper_id"] for qid in prior_test_ids
    }
    if (
        mapped_prior_papers != set(spec["audit"]["prior_test_paper_ids"])
        or mapped_prior_papers != set(spec["splits"]["test_candidate"]["excluded_prior_paper_ids"])
    ):
        raise ValueError("Prior-use test paper exclusions do not match source annotations")

    index = {
        "schema_version": MANIFEST_VERSION,
        "source_dataset": QASPER_DATASET,
        "source_config": QASPER_CONFIG,
        "source_revision": QASPER_REVISION,
        "source_url": f"https://huggingface.co/datasets/{QASPER_DATASET}",
        "source_license": "CC BY 4.0",
        "source_citation": (
            "Dasigi et al., QASPER: A Dataset of Information-Seeking Questions "
            "and Answers Anchored in Research Papers (2021)"
        ),
        "frozen_splits_sha256": sha256_file(splits_path),
        "training_source": {
            **train_audit,
            "paper_ids_sha256": train_ids_hash,
        },
        "independent_human_review": "pending; benchmark scores must not be claimed yet",
        "splits": {},
    }
    for name in ("dev", "test_candidate"):
        split_spec = spec["splits"][name]
        rows, source_audit = loaded[split_spec["source_split"]]
        index["splits"][name] = _build_split(
            name, split_spec, source["splits"][split_spec["source_split"]],
            rows, source_audit, output,
        )
    dev_targets = set(
        json.loads((output / "dev/manifest.json").read_text())[
            "selected_target_paper_ids"
        ]
    )
    test_targets = set(
        json.loads((output / "test_candidate/manifest.json").read_text())[
            "selected_target_paper_ids"
        ]
    )
    if dev_targets & test_targets:
        raise ValueError("Dev and candidate test target papers overlap")
    _write_json(output / "manifest.json", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", type=Path,
        default=Path(__file__).resolve().parent / "data/splits.json",
        help="Frozen split specification (IDs, exclusions, source hashes)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for benchmark.json and QASPER corpus materialization",
    )
    parser.add_argument("--arrow-dir", type=Path, help="Directory containing pinned qasper-*.arrow")
    args = parser.parse_args()
    manifest = materialize(args.splits, args.output, args.arrow_dir)
    print(json.dumps({"output": str(args.output), "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
