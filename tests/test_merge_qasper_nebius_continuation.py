"""The amended 40-question artifact must retain both attempts' provenance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.envoybench.qasper_official import score_run
from scripts import merge_qasper_nebius_continuation as merger
from src.eval.artifacts import configuration_hash


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def continuation(tmp_path: Path) -> tuple[Path, dict, list, dict]:
    original = merger.ORIGINAL_RUN
    original_manifest = json.loads((original / "manifest.json").read_text())
    original_rows = json.loads((original / "results.partial.json").read_text())
    original_usage = json.loads((original / "usage-budget.json").read_text())
    ids = original_manifest["question_ids"][-2:]

    manifest = copy.deepcopy(original_manifest)
    manifest.update(
        run_id="continuation-fixture",
        status="complete",
        question_ids=ids,
        full_split=False,
        subset_smoke=True,
        completed_at_utc="2026-10-10T00:00:00+00:00",
    )
    manifest["inference_budget"]["max_estimated_usd"] = 25.0
    manifest["implementation_sha256"]["benchmarks/envoybench/run.py"] = (
        merger.FORMATTED_RUNNER_SHA256
    )
    manifest["comparison_id"] = configuration_hash(
        {field: manifest[field] for field in merger.PROTOCOL_FIELDS}
    )

    rows = []
    for index, qid in enumerate(ids):
        row = copy.deepcopy(original_rows[38])
        row.update(
            run_id=manifest["run_id"], comparison_id=manifest["comparison_id"],
            question_id=qid,
            question=original_rows[38]["question"] if index == 0 else
            _frozen_question(qid),
            status="submitted", environment_status="completed", error=None,
            predicted_answer=f"fixture answer {index}",
            trajectory=[{
                "step": 1, "action": f"SUBMIT: fixture answer {index}",
                "observation": "Submitted", "done": True,
            }],
            steps=1, duration_seconds=1.0,
        )
        rows.append(row)

    usage = copy.deepcopy(original_usage)
    usage["config"] = copy.deepcopy(manifest["inference_budget"])
    usage.update(
        requests=original_usage["requests"] + 2,
        estimated_usd=original_usage["estimated_usd"] + 0.001,
        prompt_tokens=original_usage["prompt_tokens"] + 100,
        completion_tokens=original_usage["completion_tokens"] + 10,
        halted_reason=None,
    )
    source_hashes = {
        name: merger._file_sha256(original / name)
        for name in ("manifest.json", "results.partial.json", "usage-budget.json")
    }
    lineage = {
        "schema_version": merger.CONTINUATION_SCHEMA,
        "original_run_id": original_manifest["run_id"],
        "original_files_sha256": source_hashes,
        "original_usage": {
            field: original_usage[field] for field in merger.USAGE_COUNTERS
        },
        "continuation_question_ids": ids,
        "question_39_restarted_at_step": 1,
        "question_39_original_interrupted_after_steps": 11,
        "total_inference_budget": usage["config"],
        "continuation_usage": usage,
        "implementation_variance": {
            "file": "benchmarks/envoybench/run.py",
            "original_sha256": merger.ORIGINAL_RUNNER_SHA256,
            "continuation_sha256": merger.FORMATTED_RUNNER_SHA256,
            "reason": "exact line wrap of an unchanged RuntimeError statement",
        },
        "continuation_run_id": manifest["run_id"],
        "continuation_status": "complete",
    }
    path = tmp_path / "continuation"
    path.mkdir()
    _write(path / "manifest.json", manifest)
    _write(path / "results.json", rows)
    _write(path / "usage-budget.json", usage)
    _write(path / "continuation-lineage.json", lineage)
    return path, manifest, rows, usage


def _frozen_question(question_id: str) -> str:
    path = merger.DEFAULT_DATASET / "test_candidate/benchmark.json"
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    return next(question["question"] for question in benchmark["questions"]
                if question["id"] == question_id)


def test_merge_preserves_completed_rows_and_interrupted_trace_reference(
    continuation, tmp_path: Path,
) -> None:
    path, source_manifest, source_rows, source_usage = continuation
    original = merger.ORIGINAL_RUN
    before = {name: merger._file_sha256(original / name) for name in (
        "manifest.json", "results.partial.json", "usage-budget.json"
    )}
    output = tmp_path / "merged"
    manifest = merger.merge_continuation(path, output)
    rows = json.loads((output / "results.json").read_text())
    usage = json.loads((output / "usage-budget.json").read_text())
    original_rows = json.loads((original / "results.partial.json").read_text())

    assert manifest["status"] == "complete"
    assert len(rows) == len(set(row["question_id"] for row in rows)) == 40
    assert [row["question_id"] for row in rows] == manifest["question_ids"]
    assert manifest["run_id"] not in {
        json.loads((original / "manifest.json").read_text())["run_id"],
        source_manifest["run_id"],
    }
    assert manifest["comparison_id"] not in {
        json.loads((original / "manifest.json").read_text())["comparison_id"],
        source_manifest["comparison_id"],
    }
    for derived, source in zip(rows[:38], original_rows[:38]):
        assert {k: v for k, v in derived.items() if k not in merger.ROW_IDENTITY_FIELDS} == {
            k: v for k, v in source.items() if k not in merger.ROW_IDENTITY_FIELDS
        }
    for derived, source in zip(rows[38:], source_rows):
        assert derived["trajectory"] == source["trajectory"]
        assert derived["predicted_answer"] == source["predicted_answer"]
    interrupted = manifest["lineage"]["original"]["interrupted_question_39"]
    assert interrupted["row_sha256"] == merger._value_sha256(original_rows[38])
    assert interrupted["trajectory_sha256"] == merger._value_sha256(
        original_rows[38]["trajectory"]
    )
    assert manifest["protocol_amendment"]["original_no_retry_protocol_amended"] is True
    assert manifest["lineage"]["continuation"]["usage_delta"]["requests"] == 2
    assert usage == source_usage
    assert {name: merger._file_sha256(original / name) for name in before} == before
    official = score_run(output)
    assert official["run_id"] == manifest["run_id"]
    assert official["models"]["nemotron_ultra_nebius"]["recorded_episodes"] == 40


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "question IDs/order"),
    ("decoding", "models differs"),
    ("runner", "implementation hash differs"),
    ("spent_over_cap", "exceeds \\$25"),
    ("usage_not_carried", "does not carry forward"),
    ("trace_reused", "reuses the interrupted trace"),
    ("lineage_hash", "does not bind the original source files"),
])
def test_merge_rejects_incompatible_or_unbound_continuation(
    continuation, tmp_path: Path, mutation: str, match: str,
) -> None:
    path, manifest, rows, usage = continuation
    lineage = json.loads((path / "continuation-lineage.json").read_text())
    if mutation == "duplicate":
        rows[1]["question_id"] = rows[0]["question_id"]
    elif mutation == "decoding":
        manifest["models"][0]["decoding"]["temperature"] = 0.1
    elif mutation == "runner":
        manifest["implementation_sha256"]["benchmarks/envoybench/run.py"] = "0" * 64
    elif mutation == "spent_over_cap":
        usage["estimated_usd"] = 25.01
    elif mutation == "usage_not_carried":
        usage["requests"] = 1
    elif mutation == "trace_reused":
        original_rows = json.loads((merger.ORIGINAL_RUN / "results.partial.json").read_text())
        rows[0]["trajectory"] = original_rows[38]["trajectory"]
        rows[0]["steps"] = len(rows[0]["trajectory"])
    elif mutation == "lineage_hash":
        lineage["original_files_sha256"]["results.partial.json"] = "0" * 64
    _write(path / "manifest.json", manifest)
    _write(path / "results.json", rows)
    _write(path / "usage-budget.json", usage)
    _write(path / "continuation-lineage.json", lineage)
    with pytest.raises(ValueError, match=match):
        merger.merge_continuation(path, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("field", ["status", "environment_status"])
def test_merge_rejects_error_episode_despite_complete_subset_manifest(
    continuation, tmp_path: Path, field: str,
) -> None:
    path, _, rows, _ = continuation
    rows[0][field] = "error"
    _write(path / "results.json", rows)
    with pytest.raises(ValueError, match="continuation contains an error episode"):
        merger.merge_continuation(path, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()
