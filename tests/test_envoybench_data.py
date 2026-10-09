"""Offline checks for EnvoyBench's frozen QASPER split packaging."""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmarks.envoybench import build_data
from src.research.qasper import QASPER_REVISION


def _paper(paper_id: str, question_id: str, *, unanswerable: bool = False) -> dict:
    evidence = f"The result in paper {paper_id} is two passages."
    annotation = {
        "annotation_id": f"annotation-{question_id}",
        "worker_id": "worker",
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": [] if unanswerable else ["two passages"],
            "yes_no": None,
            "free_form_answer": "",
            "evidence": [] if unanswerable else [evidence],
            "highlighted_evidence": [],
        },
    }
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "A study of retrieval.",
        "full_text": {"section_name": ["Results"], "paragraphs": [[evidence]]},
        "qas": {
            "question": ["How many passages are used?"],
            "question_id": [question_id],
            "nlp_background": ["five"],
            "topic_background": ["familiar"],
            "paper_read": ["yes"],
            "search_query": [""],
            "question_writer": ["writer"],
            "answers": [[annotation]],
        },
    }


def _rows() -> dict[str, list[dict]]:
    return {
        "train": [_paper("2001.00001", "train-s")],
        "validation": [_paper("2002.00001", "dev-s")],
        "test": [
            _paper("2003.00001", "test-s"),
            _paper("2003.00002", "test-u", unanswerable=True),
        ],
    }


def _spec(rows: dict[str, list[dict]]) -> dict:
    catalogue = build_data._question_catalogue(rows["test"], "test")
    selected = build_data.select_candidate_ids(
        catalogue, seed=7, per_answerability=1, excluded_paper_ids=set()
    )
    train_ids = sorted(row["id"] for row in rows["train"])
    return {
        "schema_version": build_data.SCHEMA_VERSION,
        "source": {
            "dataset": "allenai/qasper",
            "revision": QASPER_REVISION,
            "splits": {
                split: {
                    "arrow_sha256": "0" * 64,
                    "rows_sha256": build_data.canonical_rows_sha256(split_rows),
                }
                for split, split_rows in rows.items()
            },
        },
        "training_paper_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "audit": {
            "v5_sft_training_paper_ids": train_ids,
            "v5_sft_training_paper_ids_sha256": hashlib.sha256(
                "\n".join(train_ids).encode()
            ).hexdigest(),
            "prior_test_question_ids": [],
            "prior_test_paper_ids": [],
        },
        "splits": {
            "dev": {
                "source_split": "validation",
                "status": "previously_used_development_not_held_out",
                "selection_seed": 1,
                "question_ids": ["qasper_validation_dev-s"],
                "corpus_excluded_paper_ids": [],
            },
            "test_candidate": {
                "source_split": "test",
                "status": "unreviewed_candidate_not_scored",
                "selection_seed": 7,
                "per_answerability": 1,
                "question_ids": selected,
                "excluded_prior_paper_ids": [],
                "corpus_excluded_paper_ids": [],
            },
        },
    }


def _patch_source(monkeypatch, rows: dict[str, list[dict]]) -> None:
    def fake_load(split, source, arrow_dir):
        return rows[split], {
            "source_split": split,
            "source_revision": QASPER_REVISION,
            "source_rows_sha256": source["rows_sha256"],
            "source_arrow_sha256": source["arrow_sha256"],
            "paper_count": len(rows[split]),
        }

    monkeypatch.setattr(build_data, "_load_source", fake_load)


def test_candidate_selection_is_balanced_paper_disjoint_and_deterministic():
    rows = [
        _paper("2003.00001", "used-s"),
        _paper("2003.00002", "fresh-s"),
        _paper("2003.00003", "fresh-u", unanswerable=True),
    ]
    catalogue = build_data._question_catalogue(rows, "test")
    first = build_data.select_candidate_ids(
        catalogue, seed=9, per_answerability=1,
        excluded_paper_ids={"2003.00001"},
    )
    second = build_data.select_candidate_ids(
        catalogue, seed=9, per_answerability=1,
        excluded_paper_ids={"2003.00001"},
    )
    assert first == second == ["qasper_test_fresh-s", "qasper_test_fresh-u"]
    assert len({catalogue[qid][0]["source_paper_id"] for qid in first}) == 2


def test_materialized_splits_reproduce_hashes_and_annotation_spans(tmp_path, monkeypatch):
    rows = _rows()
    _patch_source(monkeypatch, rows)
    splits = tmp_path / "splits.json"
    build_data._write_json(splits, _spec(rows))
    first = build_data.materialize(splits, tmp_path / "first")
    second = build_data.materialize(splits, tmp_path / "second")
    assert first == second
    assert first["splits"]["dev"]["status"] == "previously_used_development_not_held_out"
    assert first["splits"]["test_candidate"]["human_review_status"] == "pending"
    for name in ("dev", "test_candidate"):
        split_dir = tmp_path / "first" / name
        benchmark = json.loads((split_dir / "benchmark.json").read_text())
        manifest = json.loads((split_dir / "manifest.json").read_text())
        assert benchmark["corpus_hash"] == manifest["corpus_hash"]
        assert benchmark["corpus_hash"] == build_data.content_hash(split_dir / "corpus")
        assert first["splits"][name]["benchmark_sha256"] == build_data.sha256_file(
            split_dir / "benchmark.json"
        )
        for question in benchmark["questions"]:
            assert question["source_annotation_ids"]
            for span in question["gold_evidence"]:
                document = json.loads((split_dir / "corpus" / f"{span['doc_id']}.json").read_text())
                assert document["text"][span["start"] : span["end"]] == span["text"]


def test_materialization_rejects_training_paper_overlap(tmp_path, monkeypatch):
    rows = _rows()
    rows["train"] = [_paper("2003.00001", "train-s")]
    _patch_source(monkeypatch, rows)
    splits = tmp_path / "splits.json"
    build_data._write_json(splits, _spec(rows))
    with pytest.raises(ValueError, match="paper overlap"):
        build_data.materialize(splits, tmp_path / "output")


def test_materialization_rejects_reselected_test_ids(tmp_path, monkeypatch):
    rows = _rows()
    _patch_source(monkeypatch, rows)
    spec = _spec(rows)
    spec["splits"]["test_candidate"]["question_ids"] = ["qasper_test_test-s"]
    splits = tmp_path / "splits.json"
    build_data._write_json(splits, spec)
    with pytest.raises(ValueError, match="selection algorithm"):
        build_data.materialize(splits, tmp_path / "output")
