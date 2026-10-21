"""Actual saved-data derivation and complete stdio round trip; no paid work."""
import asyncio
import json
import shutil

import pytest

from scripts.demo_experiment_mode import run_demo
from scripts.export_serving_experiment import (
    DECISION,
    PROFILE,
    RESEARCH,
    ROOT,
    build_example,
)


@pytest.mark.parametrize("relative", [PROFILE + "/result.json",
                                     PROFILE + "/worker-identity.json",
                                     RESEARCH + "/research-note.json"])
def test_changed_saved_sources_fail_before_creating_a_record(tmp_path, relative):
    for folder in (PROFILE, RESEARCH):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    with (tmp_path / relative).open("ab") as stream:
        stream.write(b"\n")
    target = tmp_path / "new-record"
    with pytest.raises(ValueError, match="Saved source bytes differ"):
        asyncio.run(run_demo(target, source_root=tmp_path))
    assert not target.exists()


def test_real_source_derivation_preserves_unknown_latency_and_research_origin():
    bundle = build_example()
    export = bundle["serving-observations.json"]
    result = bundle["result-input.json"]
    assert export["raw_rounds"][0]["absolute_difference_seconds"] < 0.4
    assert export["raw_rounds"][1]["absolute_difference_seconds"] > 0.4
    assert export["operational"]["model_jobs_completed"] == 10
    for arm, overlap in (("baseline", 1), ("candidate", 2)):
        observed = {m["metric_id"]: m for m in result[arm]["measurements"]}
        assert observed["completed_calls"]["value"] == 4
        assert observed["engine_overlap"]["value"] == overlap
        assert observed["pair_seconds"]["value"] is None
        assert "polling margin" in observed["pair_seconds"]["unknown_reason"]
        assert not result[arm].get("failed_case_ids")
    assert bundle["research-note.json"]["origin"] == "host_research"
    assert "not a fresh paper-span verification" in export["saved_paper_research"]["provenance"]
    assert bundle["brief.json"]["candidate_outcomes_already_seen"] == "yes"
    assert bundle["brief.json"]["budget"]["max_model_requests"] == 0


def test_complete_recorded_loop_restart_retry_and_conflict_rejection(tmp_path):
    target = tmp_path / "demo"
    report = asyncio.run(run_demo(target))
    assert report["restart_verified"] and report["idempotent_retries_verified"]
    assert report["conflicting_result_and_frozen_edit_rejected"]
    assert report["assessment"]["overall"] == "inconclusive"
    assert report["assessment"]["effective_timing"] == "retrospective"
    assert report["research_note_count"] == 2 and report["result_count"] == 1
    assert report["model_calls"] == report["gpu_activations"] == 0
    assert report["network_source_calls"] == 0
    assert len(report["calls"]) == 15
    assert sum(c["expected_rejection"] for c in report["calls"]) == 2
    assert {c["session"] for c in report["calls"]} == {1, 2}
    resumed = json.loads((target / "resume.json").read_text())
    assert DECISION in resumed["context"]
    assert "artifact_measurement_verified" in resumed["context"]
    assert "unknown" in resumed["context"] and "semantic support" in resumed["warning"]
    full = json.loads((target / "complete-record.json").read_text())
    assert full["selected_result"]["assessment"] == report["assessment"]
    assert full["selected_research_note"]["note"]["semantic_support"] == "not_reviewed"
    assert full["selected_research_note"]["brief_revision"] == 2
    before = (target / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        asyncio.run(run_demo(target))
    assert (target / "report.json").read_bytes() == before
