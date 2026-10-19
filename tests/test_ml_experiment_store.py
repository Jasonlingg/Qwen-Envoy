import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_fixtures import complete_brief


def test_question_first_draft_survives_restart_and_keeps_revision_history(tmp_path):
    store = ExperimentStore(tmp_path)
    first = store.save_brief({"question": "Would a smaller model help?"})
    second = store.save_brief(complete_brief(), first["experiment_id"], expected_revision=1)
    restarted = ExperimentStore(tmp_path)
    assert restarted.get(first["experiment_id"]) == second
    assert second["revision"] == 2
    assert second["revisions"][0] == first["revisions"][0]
    assert second["status"] == "draft" and second["missing_fields"] == []
    assert "declarations" in second["reference_warning"]
    second["brief"]["question"] = "Caller mutation"
    assert store.get(first["experiment_id"])["brief"]["question"] != "Caller mutation"


def test_freeze_binds_exact_plan_and_protects_criteria(tmp_path):
    store = ExperimentStore(tmp_path)
    draft = store.save_brief(complete_brief())
    key = draft["experiment_id"]
    frozen = store.freeze_plan(key, expected_revision=1)
    assert frozen["status"] == "frozen" and frozen["plan_sha256"]
    assert frozen["frozen_plan"]["revision_sha256"] == draft["revisions"][0]["sha256"]
    assert "preregistration" in frozen["frozen_plan"]["warning"]
    assert ExperimentStore(tmp_path).freeze_plan(key, 1) == frozen
    changed = {**complete_brief(), "criteria": []}
    with pytest.raises(ValueError, match="Frozen"):
        store.save_brief(changed, key, 1)
    assert store.get(key) == frozen


def test_incomplete_question_cannot_be_frozen(tmp_path):
    store = ExperimentStore(tmp_path)
    draft = store.save_brief({"question": "What should I test?"})
    with pytest.raises(ValueError, match="Incomplete.*hypothesis"):
        store.freeze_plan(draft["experiment_id"], 1)
    assert store.get(draft["experiment_id"]) == draft


def test_changed_plan_uses_an_explicit_successor(tmp_path):
    store = ExperimentStore(tmp_path)
    prior = store.save_brief(complete_brief())
    store.freeze_plan(prior["experiment_id"], 1)
    successor = store.save_brief({**complete_brief(),
                                 "supersedes_experiment_id": prior["experiment_id"]})
    assert successor["experiment_id"] != prior["experiment_id"]
    assert store.get(prior["experiment_id"])["status"] == "frozen"
    with pytest.raises(ValueError):
        store.save_brief({**complete_brief(), "supersedes_experiment_id": "experiment_" + "a" * 32})


@pytest.mark.parametrize("revision", [None, 0, True, "1", 2])
def test_stale_or_invalid_edits_do_not_write_a_revision(tmp_path, revision):
    store = ExperimentStore(tmp_path)
    original = store.save_brief(complete_brief())
    with pytest.raises(ValueError, match="expected_revision"):
        store.save_brief(complete_brief(), original["experiment_id"], revision)
    assert store.get(original["experiment_id"]) == original


def test_two_stores_cannot_lose_a_simultaneous_edit(tmp_path):
    stores = [ExperimentStore(tmp_path), ExperimentStore(tmp_path)]
    original = stores[0].save_brief({"question": "Original question"})

    def edit(item):
        index, store = item
        try:
            return store.save_brief({"question": f"Revision from {index}"},
                                    original["experiment_id"], 1)["revision"]
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(edit, enumerate(stores)))
    assert sorted(outcomes, key=str) == [2, "stale"]
    assert stores[0].get(original["experiment_id"])["revision"] == 2


def test_failed_index_commit_preserves_previous_draft(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    original = store.save_brief({"question": "Original question"})
    write = store.files.write

    def fail_index(directory, name, raw, **kwargs):
        if name.startswith("experiment_"):
            raise OSError("Simulated interrupted commit")
        return write(directory, name, raw, **kwargs)

    monkeypatch.setattr(store.files, "write", fail_index)
    with pytest.raises(ValueError, match="inaccessible"):
        store.save_brief(complete_brief(), original["experiment_id"], 1)
    assert store.get(original["experiment_id"]) == original
    assert len(store.list()["experiments"]) == 1


@pytest.mark.parametrize("location", ["index", "revision", "frozen_plan"])
def test_modified_files_are_detected_by_get_and_list(tmp_path, location):
    store = ExperimentStore(tmp_path)
    draft = store.save_brief(complete_brief())
    record = store.freeze_plan(draft["experiment_id"], 1)
    key = record["experiment_id"]
    if location == "index":
        path = tmp_path / f"{key}.json"
        data = json.loads(path.read_text())
        data["index"]["frozen_plan_sha256"] = None
    else:
        sha = (record["plan_sha256"] if location == "frozen_plan"
               else record["revisions"][0]["sha256"])
        path = tmp_path / f"object_{sha}.json"
        data = json.loads(path.read_text())
        data["brief"]["criteria"][0]["threshold"] = -100.0
    path.write_text(json.dumps(data))
    for read in (lambda: store.get(key), store.list):
        with pytest.raises(ValueError, match="corrupt"):
            read()


def test_listing_is_bounded_and_corrupt_records_are_not_silently_skipped(tmp_path):
    store = ExperimentStore(tmp_path)
    first = store.save_brief({"question": "First question"})
    second = store.save_brief({"question": "Second question"})
    assert store.list(1) == {"experiments": [store.list()["experiments"][0]], "truncated": True}
    assert {row["experiment_id"] for row in store.list()["experiments"]} == {
        first["experiment_id"], second["experiment_id"]}
    (tmp_path / f"{first['experiment_id']}.json").write_bytes(b'{"broken":')
    with pytest.raises(ValueError):
        store.list(1)
    assert store.get(second["experiment_id"]) == second


@pytest.mark.parametrize("operation", ["index", "object", "lock"])
def test_symlinks_cannot_redirect_reads_or_writes(tmp_path, operation):
    root = tmp_path / "records"
    store = ExperimentStore(root)
    record = store.save_brief(complete_brief())
    external = tmp_path / "outside.json"
    external.write_text("untouched")
    name = {"index": f"{record['experiment_id']}.json", "lock": ".experiments.lock",
            "object": f"object_{record['revisions'][0]['sha256']}.json"}[operation]
    path = root / name
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(ValueError):
        store.get(record["experiment_id"])
    assert external.read_text() == "untouched"


@pytest.mark.parametrize("bad", ["../escape", "experiment_" + "a" * 32 + "/more", "", True])
def test_ids_cannot_be_paths(tmp_path, bad):
    with pytest.raises(ValueError, match="ID"):
        ExperimentStore(tmp_path).get(bad)
