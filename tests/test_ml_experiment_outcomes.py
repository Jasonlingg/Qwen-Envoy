import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_io import decode, digest, encode
from src.product.ml_experiment_measurements import Measurement
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief


def frozen_store(tmp_path, brief=None):
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    resolver = EvidenceResolver(library_root=tmp_path / "library", artifact_root=exports)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief(brief or planned_brief())
    return store, store.freeze_plan(saved["experiment_id"], 1), exports


def attach_measurements(exports, result, *, full=True):
    selected = {}
    for label in ("baseline", "candidate"):
        measurement = result[label]["measurements"][0]
        selected[label] = (Measurement.model_validate(measurement).model_dump(mode="json")
                           if full else measurement["value"])
        measurement["reference_id"] = label + "_value"
    raw = json.dumps(selected).encode()
    path = exports / "measurements.json"
    path.write_bytes(raw)
    result["references"] = [{"kind": "local_artifact", "reference_id": label + "_value",
        "relative_path": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
        "json_pointer": "/" + label} for label in ("baseline", "candidate")]
    return path


def record(store, frozen, result=None):
    return store.record_result(frozen["experiment_id"], frozen["plan_sha256"],
                               result or paired_result())


def test_append_reload_select_and_freeze_preserve_original_plan(tmp_path):
    store, frozen, _ = frozen_store(tmp_path)
    first = record(store, frozen)
    second_result = paired_result(run_id="synthetic_second")
    second_result["candidate"]["measurements"][0]["value"] = 95.0
    second = record(store, frozen, second_result)
    assert second["selected_result"]["assessment"]["overall"] == "criteria_not_met"
    assert [row["run_id"] for row in second["results"]] == ["synthetic_run", "synthetic_second"]
    restarted = ExperimentStore(tmp_path / "records")
    old = restarted.get(frozen["experiment_id"], "synthetic_run")
    assert old["selected_result"] == first["selected_result"]
    assert old["frozen_plan"] == frozen["frozen_plan"]
    assert restarted.freeze_plan(frozen["experiment_id"], 1)["results"] == second["results"]
    with pytest.raises(ValueError, match="No recorded result"):
        restarted.get(frozen["experiment_id"], "no_such_run")
    with pytest.raises(ValueError, match="cannot be edited"):
        restarted.save_brief(planned_brief(), frozen["experiment_id"], 1)


def test_verified_artifact_retry_survives_deleted_source_but_conflict_rejected(tmp_path):
    store, frozen, exports = frozen_store(tmp_path)
    result = paired_result()
    path = attach_measurements(exports, result)
    first = record(store, frozen, result)
    assert set(first["selected_result"]["assessment"]["measurement_provenance"].values()) == {
        "artifact_measurement_verified"}
    path.unlink()
    assert record(store, frozen, result)["selected_result"] == first["selected_result"]
    assert store.get(frozen["experiment_id"])["selected_result"] == first["selected_result"]
    changed = deepcopy(result)
    changed["candidate"]["measurements"][0]["value"] = 50.0
    with pytest.raises(ValueError, match="different immutable"):
        record(store, frozen, changed)
    assert len(store.get(frozen["experiment_id"])["results"]) == 1


@pytest.mark.parametrize("full", [False, True])
def test_result_value_must_match_verified_source(tmp_path, full):
    store, frozen, exports = frozen_store(tmp_path)
    result = paired_result()
    attach_measurements(exports, result, full=full)
    result["candidate"]["measurements"][0]["value"] = 1.0
    with pytest.raises(ValueError, match="differs"):
        record(store, frozen, result)
    assert store.get(frozen["experiment_id"])["results"] == []


@pytest.mark.parametrize("review,attachment,expected", [
    ("model_assisted", "full", "inconclusive"),
    ("human_declared", "none", "inconclusive"),
    ("human_declared", "scalar", "inconclusive"),
    ("human_declared", "full", "criteria_met"),
])
def test_semantic_metric_needs_bound_human_declaration(tmp_path, review, attachment, expected):
    brief = planned_brief()
    brief["metrics"][0].update(measurement_kind="quality", requires_semantic_review=True)
    store, frozen, exports = frozen_store(tmp_path, brief)
    result = paired_result(brief)
    for label in ("baseline", "candidate"):
        result[label]["measurements"][0]["review"] = {"kind": review,
            "reviewer": "Synthetic reviewer declaration", "reviewed_at": "2026-10-01T00:00:00Z"}
    if attachment != "none":
        attach_measurements(exports, result, full=attachment == "full")
    saved = record(store, frozen, result)["selected_result"]
    assert saved["assessment"]["overall"] == expected
    assert any("not authenticated" in warning for warning in saved["assessment"]["warnings"])
    assert all(r["semantic_support"] == "not_reviewed" for r in saved["evidence_receipts"])


def test_artifact_review_metadata_cannot_be_overridden(tmp_path):
    store, frozen, exports = frozen_store(tmp_path)
    result = paired_result()
    attach_measurements(exports, result)
    result["candidate"]["measurements"][0]["review"] = {"kind": "human_declared",
        "reviewer": "Invented reviewer", "reviewed_at": "2026-10-01T00:00:00Z"}
    with pytest.raises(ValueError, match="differs"):
        record(store, frozen, result)


def test_exact_plan_required_and_unknown_metrics_rejected(tmp_path):
    store, frozen, _ = frozen_store(tmp_path)
    draft = store.save_brief({"question": "Not frozen"})
    with pytest.raises(ValueError, match="exact frozen"):
        store.record_result(draft["experiment_id"], frozen["plan_sha256"], paired_result())
    with pytest.raises(ValueError, match="exact frozen"):
        store.record_result(frozen["experiment_id"], "0" * 64, paired_result())
    other = store.save_brief(planned_brief())
    other_plan = store.freeze_plan(other["experiment_id"], 1)
    with pytest.raises(ValueError, match="exact frozen"):
        store.record_result(other["experiment_id"], frozen["plan_sha256"], paired_result())
    assert other_plan["plan_sha256"] != frozen["plan_sha256"]
    result = paired_result()
    result["candidate"]["measurements"][0]["metric_id"] = "unplanned"
    with pytest.raises(ValueError, match="not defined"):
        record(store, frozen, result)


def test_parallel_appends_do_not_lose_runs(tmp_path):
    store, frozen, _ = frozen_store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        views = list(pool.map(lambda run: record(ExperimentStore(tmp_path / "records"), frozen,
            paired_result(run_id=run)), ["synthetic_a", "synthetic_b"]))
    assert len(views) == 2
    assert {r["run_id"] for r in store.get(frozen["experiment_id"])["results"]} == {
        "synthetic_a", "synthetic_b"}


def test_interrupted_index_commit_leaves_plan_and_prior_result_intact(tmp_path, monkeypatch):
    store, frozen, _ = frozen_store(tmp_path)
    first = record(store, frozen)
    def interrupted(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(store, "_write_index", interrupted)
    with pytest.raises(ValueError, match="inaccessible"):
        record(store, frozen, paired_result(run_id="synthetic_second"))
    assert store.get(frozen["experiment_id"]) == first


def test_rehashed_assessment_tampering_is_detected_by_recomputation(tmp_path):
    store, frozen, _ = frozen_store(tmp_path)
    view = record(store, frozen)
    root = tmp_path / "records"
    index_path = root / (frozen["experiment_id"] + ".json")
    envelope = decode(index_path.read_bytes())
    outcome = deepcopy(view["selected_result"])
    outcome.pop("sha256")
    outcome["assessment"]["criteria"][0]["observed_value"] = 99.0
    raw = encode(outcome)
    sha = digest(raw)
    (root / ("object_" + sha + ".json")).write_bytes(raw)
    envelope["index"]["results"][0]["sha256"] = sha
    envelope["sha256"] = digest(encode(envelope["index"]))
    index_path.write_bytes(encode(envelope))
    with pytest.raises(ValueError, match="corrupt"):
        store.get(frozen["experiment_id"])
