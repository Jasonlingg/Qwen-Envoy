"""Compatibility and summary semantics for the outcome extension."""

from copy import deepcopy

from src.product.ml_experiment_io import decode, digest, encode
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_fixtures import complete_brief
from tests.ml_experiment_result_fixtures import paired_result, planned_brief


def test_pre_outcome_frozen_plan_remains_readable_with_original_digest(tmp_path):
    store = ExperimentStore(tmp_path)
    saved = store.save_brief(complete_brief())
    frozen = store.freeze_plan(saved["experiment_id"], 1)
    index_path = tmp_path / (saved["experiment_id"] + ".json")
    envelope = decode(index_path.read_bytes())

    def legacy_object(sha, *, plan_revision=None):
        raw = decode((tmp_path / ("object_" + sha + ".json")).read_bytes())
        for metric in raw["brief"]["metrics"]:
            for field in ("measurement_kind", "timing_scope", "failure_policy"):
                metric.pop(field)
        if plan_revision is not None:
            raw["revision_sha256"] = plan_revision
        body = encode(raw)
        old_sha = digest(body)
        (tmp_path / ("object_" + old_sha + ".json")).write_bytes(body)
        return old_sha

    revision = legacy_object(envelope["index"]["revisions"][0]["sha256"])
    plan = legacy_object(frozen["plan_sha256"], plan_revision=revision)
    envelope["index"]["revisions"][0]["sha256"] = revision
    envelope["index"]["frozen_plan_sha256"] = plan
    envelope["index"].pop("results")
    envelope["sha256"] = digest(encode(envelope["index"]))
    index_path.write_bytes(encode(envelope))
    restored = ExperimentStore(tmp_path).get(saved["experiment_id"])
    assert restored["plan_sha256"] == plan and restored["results"] == []
    outcome = store.record_result(saved["experiment_id"], plan, paired_result())
    assert outcome["plan_sha256"] == plan
    assert outcome["selected_result"]["assessment"]["overall"] == "inconclusive"
    assert "successor" in outcome["selected_result"]["assessment"]["criteria"][0]["reason"]


def test_required_failure_takes_precedence_over_unknown_and_list_tracks_results(tmp_path):
    brief = planned_brief()
    brief["metrics"].append({**deepcopy(brief["metrics"][0]), "metric_id": "missing"})
    brief["criteria"].append({**deepcopy(brief["criteria"][0]), "criterion_id": "unknown",
                              "metric_id": "missing"})
    store = ExperimentStore(tmp_path)
    saved = store.save_brief(brief)
    frozen = store.freeze_plan(saved["experiment_id"], 1)
    result = paired_result(brief)
    result["candidate"]["measurements"][0]["value"] = 95.0
    recorded = store.record_result(saved["experiment_id"], frozen["plan_sha256"], result)
    assessment = recorded["selected_result"]["assessment"]
    assert [c["status"] for c in assessment["criteria"]] == ["not_met", "unknown"]
    assert assessment["overall"] == "criteria_not_met"
    summary = store.list()["experiments"][0]
    assert summary["result_count"] == 1
    assert summary["updated_at"] == recorded["selected_result"]["recorded_at"]
