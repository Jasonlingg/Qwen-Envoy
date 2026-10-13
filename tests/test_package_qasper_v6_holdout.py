"""Offline checks for the immutable QASPER v6 EnvoyBench holdout package."""

from __future__ import annotations

import json

import pytest

from benchmarks.envoybench import build_data, package_v6_holdout, score_v6_holdout
from benchmarks.envoybench.frozen_dataset import load_split
from src.eval.artifacts import configuration_hash, content_hash
from src.research.qasper import QASPER_REVISION, _paper_document, _to_code_exec_question


def _paper(index: int, *, unanswerable: bool) -> dict:
    paper_id = f"2003.{index:05d}"
    passage = f"Paper {paper_id} finds two passages."
    return {
        "id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "A retrieval study.",
        "full_text": {"section_name": ["Results"], "paragraphs": [[passage]]},
        "qas": {
            "question": ["How many passages are used?"],
            "question_id": [f"test-{index:02d}"],
            "nlp_background": ["five"],
            "topic_background": ["familiar"],
            "paper_read": ["yes"],
            "search_query": [""],
            "question_writer": ["writer"],
            "answers": [[{
                "annotation_id": f"annotation-{index}",
                "worker_id": "worker",
                "answer": {
                    "unanswerable": unanswerable,
                    "extractive_spans": [] if unanswerable else ["two passages"],
                    "yes_no": None,
                    "free_form_answer": "",
                    "evidence": [] if unanswerable else [passage],
                    "highlighted_evidence": [],
                },
            }]],
        },
    }


@pytest.fixture
def frozen_sources(tmp_path, monkeypatch):
    rows = [_paper(index, unanswerable=index > 17) for index in range(1, 24)]
    extra = _paper(24, unanswerable=True)
    for field, values in extra["qas"].items():
        rows[-1]["qas"][field].extend(values)
    catalogue = build_data._question_catalogue(rows, "test")
    question_ids = sorted(catalogue)
    corpus_dir = tmp_path / "source_corpus"
    corpus_dir.mkdir()
    for row in rows:
        document, _ = _paper_document(row, "test")
        build_data._write_json(corpus_dir / f"{document['doc_id']}.json", document)
    corpus_hash = content_hash(corpus_dir)
    questions = [_to_code_exec_question(catalogue[qid][0]) for qid in question_ids]
    benchmark = {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": "synthetic-locked-holdout",
        "corpus_hash": corpus_hash,
        "reserved_doc_ids": sorted(path.stem for path in corpus_dir.glob("*.json")),
        "questions": questions,
    }
    benchmark_path = tmp_path / "source_benchmark.json"
    build_data._write_json(benchmark_path, benchmark)
    benchmark_hash = build_data.sha256_file(benchmark_path)
    selection_path = tmp_path / "selection.json"
    build_data._write_json(selection_path, {
        "source_split": "test",
        "source_revision": QASPER_REVISION,
        "seed": 7,
        "question_ids": question_ids,
        "benchmark_sha256": benchmark_hash,
        "corpus_hash": corpus_hash,
        "exclusions": [],
    })
    lock = {
        "schema_version": "qasper-v6-holdout-lock-v1",
        "source_split": "test",
        "source_revision": QASPER_REVISION,
        "selection_seed": 7,
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_hash,
        "selection_path": str(selection_path),
        "selection_sha256": build_data.sha256_file(selection_path),
        "corpus_path": str(corpus_dir),
        "corpus_hash": corpus_hash,
        "question_ids": question_ids,
        "reserved_doc_ids": sorted({
            catalogue[qid][0]["target_doc_ids"][0] for qid in question_ids
        }),
        "answerability_counts": {"sufficient": 17, "insufficient": 7},
        "prior_exclusion_benchmarks": [],
        "overlap_with_prior_benchmarks": 0,
        "overlap_with_reviewed_sft_candidates": 0,
    }
    lock_path = tmp_path / "lock.json"
    build_data._write_json(lock_path, lock)
    source_splits_path = tmp_path / "source_splits.json"
    build_data._write_json(source_splits_path, {
        "source": {
            "dataset": "allenai/qasper",
            "revision": QASPER_REVISION,
            "splits": {"test": {
                "arrow_sha256": "0" * 64,
                "rows_sha256": build_data.canonical_rows_sha256(rows),
            }},
        },
    })

    def fake_load(split, source, arrow_dir):
        assert split == "test"
        return rows, {
            "source_split": split,
            "source_revision": QASPER_REVISION,
            "source_rows_sha256": source["rows_sha256"],
            "source_arrow_sha256": source["arrow_sha256"],
            "paper_count": len(rows),
        }

    monkeypatch.setattr(build_data, "_load_source", fake_load)
    return {
        "rows": rows,
        "lock": lock,
        "lock_path": lock_path,
        "corpus_dir": corpus_dir,
        "source_splits_path": source_splits_path,
    }


def _materialize(frozen_sources, output):
    return package_v6_holdout.materialize(
        frozen_sources["lock_path"], output,
        source_splits_path=frozen_sources["source_splits_path"],
    )


def test_package_reproduces_locked_ids_corpus_and_annotations(tmp_path, frozen_sources):
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert _materialize(frozen_sources, first) == _materialize(frozen_sources, second)
    benchmark, snapshot, _, corpus, entry = load_split(first, "holdout")
    lock = frozen_sources["lock"]
    assert [q["id"] for q in benchmark["questions"]] == lock["question_ids"]
    assert content_hash(corpus) == lock["corpus_hash"]
    assert entry["target_paper_count"] == 23
    assert entry["answerability"] == lock["answerability_counts"]
    assert snapshot["human_review_status"] == "pending"
    assert all(q["source_annotation_ids"] for q in benchmark["questions"])
    assert all(
        (corpus / path.name).read_bytes() == path.read_bytes()
        for path in frozen_sources["corpus_dir"].glob("*.json")
    )
    with pytest.raises(FileExistsError):
        _materialize(frozen_sources, first)


def test_package_rejects_changed_frozen_benchmark_or_corpus(tmp_path, frozen_sources):
    benchmark_path = tmp_path / "source_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    benchmark["questions"][0]["answer"] = "changed"
    build_data._write_json(benchmark_path, benchmark)
    with pytest.raises(ValueError, match="Frozen file hash differs"):
        _materialize(frozen_sources, tmp_path / "bad_benchmark")

    build_data._write_json(benchmark_path, {
        **benchmark, "questions": [
            _to_code_exec_question(
                build_data._question_catalogue(frozen_sources["rows"], "test")[qid][0]
            )
            for qid in frozen_sources["lock"]["question_ids"]
        ],
    })
    corpus_file = next(frozen_sources["corpus_dir"].glob("*.json"))
    corpus_file.write_text(corpus_file.read_text() + "\n")
    with pytest.raises(ValueError, match="source corpus hash differs"):
        _materialize(frozen_sources, tmp_path / "bad_corpus")


def test_package_rejects_changed_pinned_question(tmp_path, frozen_sources):
    frozen_sources["rows"][0]["qas"]["question"][0] = "A different question?"
    with pytest.raises(ValueError, match="Frozen question differs from pinned QASPER"):
        _materialize(frozen_sources, tmp_path / "bad_question")


def test_package_rejects_prior_benchmark_target_overlap(tmp_path, frozen_sources):
    benchmark_path = tmp_path / "source_benchmark.json"
    prior = {"questions": [json.loads(benchmark_path.read_text())["questions"][0]]}
    prior_path = tmp_path / "prior.json"
    build_data._write_json(prior_path, prior)
    exclusion = {"path": str(prior_path), "sha256": build_data.sha256_file(prior_path)}
    selection_path = tmp_path / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["exclusions"] = [exclusion]
    build_data._write_json(selection_path, selection)
    lock = frozen_sources["lock"]
    lock["prior_exclusion_benchmarks"] = [exclusion]
    lock["selection_sha256"] = build_data.sha256_file(selection_path)
    build_data._write_json(frozen_sources["lock_path"], lock)
    with pytest.raises(ValueError, match="overlap a prior benchmark"):
        _materialize(frozen_sources, tmp_path / "bad_overlap")


def test_holdout_scorer_requires_complete_matrix_and_uses_original_annotations(
    tmp_path, frozen_sources,
):
    dataset = tmp_path / "dataset"
    _materialize(frozen_sources, dataset)
    benchmark, snapshot, benchmark_path, _, _ = load_split(dataset, "holdout")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_keys = ["base", "candidate"]
    results = []
    run_id = "synthetic-holdout-run"
    comparison_id = "synthetic-holdout-comparison"
    benchmark_hash = configuration_hash(benchmark)
    for model in model_keys:
        for question in benchmark["questions"]:
            answer = question["reference_answers"][0]["answer_text"]
            results.append({
                "schema_version": "envoybench-run-v1",
                "run_id": run_id, "comparison_id": comparison_id,
                "benchmark_id": benchmark["benchmark_id"],
                "benchmark_hash": benchmark_hash,
                "corpus_hash": benchmark["corpus_hash"],
                "split": "holdout", "split_status": snapshot["split_status"],
                "model_key": model, "question_id": question["id"],
                "question": question["question"],
                "status": "submitted",
                "predicted_answer": answer,
                "predicted_citations": [], "predicted_evidence": [],
                "trajectory": [{"action": f"SUBMIT: {answer} CITATIONS: [] EVIDENCE: []"}],
                "duration_seconds": 1.0,
            })
    build_data._write_json(run_dir / "results.json", results)
    manifest = {
        "schema_version": "envoybench-run-v1",
        "run_id": run_id, "comparison_id": comparison_id,
        "benchmark_id": benchmark["benchmark_id"],
        "status": "complete", "full_split": True, "split": "holdout",
        "subset_smoke": False, "split_question_count": len(benchmark["questions"]),
        "split_status": snapshot["split_status"],
        "question_ids": [q["id"] for q in benchmark["questions"]],
        "benchmark_hash": benchmark_hash,
        "benchmark_file_sha256": build_data.sha256_file(benchmark_path),
        "corpus_hash": benchmark["corpus_hash"],
        "models": [{"key": key} for key in model_keys],
    }
    build_data._write_json(run_dir / "manifest.json", manifest)
    options = {"holdout_lock_path": frozen_sources["lock_path"]}
    report = score_v6_holdout.score(dataset, run_dir, **options)
    assert report["question_count"] == 24
    assert report["systems"]["base"]["answer_f1_0_100"] == 100
    assert report["systems"]["candidate"]["literal_unanswerable_on_insufficient"] == 7
    assert "false_refusals" not in report["systems"]["candidate"]
    assert report["systems"]["candidate"]["refusal_behavior_review_required"] is True

    sufficient = next(q for q in benchmark["questions"]
                      if q["expected_answerability"] == "sufficient")
    changed = next(row for row in results
                   if row["model_key"] == "candidate" and row["question_id"] == sufficient["id"])
    changed["predicted_answer"] = "Unanswerable."
    changed["trajectory"][-1]["action"] = "SUBMIT: Unanswerable. CITATIONS: [] EVIDENCE: []"
    build_data._write_json(run_dir / "results.json", results)
    scored = score_v6_holdout.score(dataset, run_dir, **options)
    assert scored["systems"]["candidate"]["literal_unanswerable_on_sufficient"] == 1

    changed["predicted_answer"] = "I cannot answer from this paper."
    changed["trajectory"][-1]["action"] = (
        "SUBMIT: I cannot answer from this paper. CITATIONS: [] EVIDENCE: []"
    )
    build_data._write_json(run_dir / "results.json", results)
    scored = score_v6_holdout.score(dataset, run_dir, **options)
    assert scored["systems"]["candidate"]["literal_unanswerable_on_sufficient"] == 0
    assert scored["systems"]["candidate"]["possible_refusals_review_needed"] == 1
    flagged = next(row for row in scored["systems"]["candidate"]["question_rows"]
                   if row["question_id"] == sufficient["id"])
    assert flagged["refusal_diagnostic"] == "possible_refusal_review_needed"

    build_data._write_json(run_dir / "results.json", results[:-1])
    with pytest.raises(ValueError, match="complete paired model/question matrix"):
        score_v6_holdout.score(dataset, run_dir, **options)

    changed["trajectory"][-1]["action"] = "SUBMIT: different CITATIONS: [] EVIDENCE: []"
    build_data._write_json(run_dir / "results.json", results)
    with pytest.raises(ValueError, match="submitted fields differ from final action"):
        score_v6_holdout.score(dataset, run_dir, **options)

    changed["trajectory"][-1]["action"] = (
        "SUBMIT: I cannot answer from this paper. CITATIONS: [] EVIDENCE: []"
    )
    changed["run_id"] = "another-run"
    build_data._write_json(run_dir / "results.json", results)
    with pytest.raises(ValueError, match="run_id mismatch"):
        score_v6_holdout.score(dataset, run_dir, **options)

    changed["run_id"] = run_id
    build_data._write_json(run_dir / "results.json", results)
    manifest["models"].append({"key": "candidate"})
    build_data._write_json(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest model keys"):
        score_v6_holdout.score(dataset, run_dir, **options)

    manifest["models"].clear()
    build_data._write_json(run_dir / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest model keys"):
        score_v6_holdout.score(dataset, run_dir, **options)

    manifest["models"] = [{"key": key} for key in model_keys]
    build_data._write_json(run_dir / "manifest.json", manifest)
    frozen_sources["lock"]["question_ids"].pop()
    build_data._write_json(frozen_sources["lock_path"], frozen_sources["lock"])
    with pytest.raises(ValueError, match="pinned 24-question holdout lock"):
        score_v6_holdout.score(dataset, run_dir, **options)
