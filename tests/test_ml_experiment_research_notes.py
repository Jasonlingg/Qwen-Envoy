"""Research survives resumption without changing the plan or grading claims."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.product.ml_experiment_io import decode, digest, encode
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief
from tests.test_ml_experiment_evidence import sources


def note(refs=(), note_id="memory_search"):
    return {"note_id": note_id, "question": "Which memory method fits?",
            "findings": ([{"claim": "Candidate finding, not independently reviewed.",
                           "reference_ids": [refs[0]["reference_id"]]}] if refs else []),
            "references": list(refs), "uncertainties": ["Needs a local measurement."],
            "proposed_test": "Compare memory while keeping the workload fixed.",
            "data_kind": "synthetic"}


def test_research_survives_revisions_restart_and_missing_sources(tmp_path):
    library, exports, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    first = store.save_brief({"question": "Reduce memory?"})
    key = first["experiment_id"]
    attached = store.record_research(key, 1, note(refs))
    stored = attached["selected_research_note"]
    assert attached["revision"] == 1 and attached["brief"] == first["brief"]
    assert stored["brief_sha256"] == first["revisions"][0]["sha256"]
    assert [r["verification"] for r in stored["evidence_receipts"]] == [
        "paper_span_verified", "artifact_value_verified", "reference_only"]
    updated = store.save_brief({"question": "Reduce memory with quality held fixed?"}, key, 1)
    assert updated["selected_research_note"] == stored
    (exports / "run.json").unlink()
    (library.papers_dir / "2609.12345v2" / "document.json").unlink()
    reopened = ExperimentStore(tmp_path / "records")
    assert reopened.get(key) == updated
    # An exact retry works even after a brief edit and removal of the original sources.
    assert reopened.record_research(key, 1, note(refs)) == updated
    assert stored["note"]["semantic_support"] == "not_reviewed"
    assert stored["note"]["proposal_status"] == "not_executed_or_validated"


def test_frozen_plan_and_supplied_result_stay_exactly_unchanged(tmp_path):
    store = ExperimentStore(tmp_path)
    saved = store.save_brief(planned_brief())
    key = saved["experiment_id"]
    frozen = store.freeze_plan(key, 1)
    recorded = store.record_result(key, frozen["plan_sha256"], paired_result())
    result = store.record_research(key, 1, note())
    for name in ("brief", "frozen_plan", "plan_sha256", "selected_result", "results", "revisions"):
        assert result[name] == recorded[name]
    assert store.freeze_plan(key, 1) == result


@pytest.mark.parametrize("revision", [None, 0, True, "1", 2])
def test_stale_research_never_writes(tmp_path, revision):
    store = ExperimentStore(tmp_path)
    original = store.save_brief({"question": "Memory?"})
    with pytest.raises(ValueError, match="expected_revision"):
        store.record_research(original["experiment_id"], revision, note())
    assert store.get(original["experiment_id"]) == original


def test_note_ids_are_immutable_and_selection_is_explicit(tmp_path):
    store = ExperimentStore(tmp_path)
    key = store.save_brief({"question": "Memory?"})["experiment_id"]
    first = store.record_research(key, 1, note())
    store.record_research(key, 1, note(note_id="second_search"))
    assert store.get(key, research_note_id="memory_search")["selected_research_note"] == (
        first["selected_research_note"])
    with pytest.raises(ValueError, match="immutable"):
        store.record_research(key, 1, {**note(), "question": "Changed"})
    with pytest.raises(ValueError, match="No research"):
        store.get(key, research_note_id="absent")


def test_concurrent_notes_are_not_lost_or_cross_linked(tmp_path):
    stores = [ExperimentStore(tmp_path), ExperimentStore(tmp_path)]
    key = stores[0].save_brief({"question": "Memory?"})["experiment_id"]
    other = stores[0].save_brief({"question": "Unrelated experiment"})["experiment_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: stores[i].record_research(key, 1, note(note_id=f"search_{i}")),
                      range(2)))
    assert {n["note_id"] for n in stores[0].get(key)["research_notes"]} == {"search_0", "search_1"}
    assert not stores[0].get(other)["research_notes"]


@pytest.mark.parametrize("change", ["quote", "source_symlink", "unreviewed", "missing_ref"])
def test_invalid_evidence_and_review_claims_cannot_be_saved(tmp_path, change):
    library, _, resolver, refs = sources(tmp_path)
    store = ExperimentStore(tmp_path / "records", evidence_resolver=resolver)
    saved = store.save_brief({"question": "Memory?"})
    value = note(refs[:1])
    if change == "quote":
        value["references"][0]["quote"] = "x" * 40
    elif change == "source_symlink":
        library.papers_dir.rename(library.root / "moved")
        library.papers_dir.symlink_to(library.root / "moved", target_is_directory=True)
    elif change == "unreviewed":
        value["semantic_support"] = "human_verified"
    else:
        value["findings"][0]["reference_ids"] = ["missing"]
    with pytest.raises(ValueError):
        store.record_research(saved["experiment_id"], 1, value)
    assert store.get(saved["experiment_id"]) == saved


def test_corrupt_and_hash_valid_cross_experiment_bindings_are_rejected(tmp_path):
    store = ExperimentStore(tmp_path)
    key = store.save_brief({"question": "Memory?"})["experiment_id"]
    other = store.save_brief({"question": "Latency?"})["experiment_id"]
    saved = store.record_research(key, 1, note())
    sha = saved["selected_research_note"]["sha256"]
    path = tmp_path / f"object_{sha}.json"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="corrupt"):
        store.get(key)
    data = decode(original)
    data["experiment_id"] = other
    raw = encode(data)
    replacement = digest(raw)
    (tmp_path / f"object_{replacement}.json").write_bytes(raw)
    index_path = tmp_path / f"{key}.json"
    envelope = decode(index_path.read_bytes())
    envelope["index"]["research"][0]["sha256"] = replacement
    envelope["sha256"] = digest(encode(envelope["index"]))
    index_path.write_bytes(encode(envelope))
    with pytest.raises(ValueError, match="corrupt"):
        store.get(key)
    assert store.get(other)["research_notes"] == []


def test_legacy_index_without_research_opens_without_rewriting_it(tmp_path):
    store = ExperimentStore(tmp_path)
    saved = store.save_brief({"question": "Old experiment"})
    path = tmp_path / f"{saved['experiment_id']}.json"
    envelope = decode(path.read_bytes())
    envelope["index"].pop("research")
    envelope["sha256"] = digest(encode(envelope["index"]))
    original = encode(envelope)
    path.write_bytes(original)
    assert ExperimentStore(tmp_path).get(saved["experiment_id"]) == saved
    assert path.read_bytes() == original


def test_failed_note_index_commit_keeps_the_original_experiment(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    saved = store.save_brief({"question": "Memory?"})
    def fail(*args):
        raise OSError("Interrupted note index commit")
    monkeypatch.setattr(store, "_write_index", fail)
    with pytest.raises(ValueError, match="inaccessible"):
        store.record_research(saved["experiment_id"], 1, note())
    assert ExperimentStore(tmp_path).get(saved["experiment_id"]) == saved
