import json
import os
from types import SimpleNamespace

import pytest

from src.product.ml_experiment_io import digest, encode
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_fixtures import complete_brief


def test_store_root_and_fifo_lock_are_rejected_without_following_them(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        ExperimentStore(linked)
    assert list(outside.iterdir()) == []
    root = tmp_path / "records"
    store = ExperimentStore(root)
    os.mkfifo(root / ".experiments.lock")
    with pytest.raises(ValueError, match="lock"):
        store.list()


def test_generated_id_collision_cannot_replace_an_existing_record(tmp_path, monkeypatch):
    from src.product import ml_experiment_store

    store = ExperimentStore(tmp_path)
    original = store.save_brief({"question": "Original question"})
    monkeypatch.setattr(ml_experiment_store, "uuid4", lambda: SimpleNamespace(
        hex=original["experiment_id"].removeprefix("experiment_")))
    with pytest.raises(ValueError, match="already exists"):
        store.save_brief({"question": "Accidental overwrite"})
    assert store.get(original["experiment_id"]) == original


def test_interrupted_freeze_stays_draft_until_index_commit(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path)
    draft = store.save_brief(complete_brief())
    original = store._write_index

    def interrupted(*args):
        raise OSError("Interrupted freeze")

    monkeypatch.setattr(store, "_write_index", interrupted)
    with pytest.raises(ValueError, match="inaccessible"):
        store.freeze_plan(draft["experiment_id"], 1)
    assert ExperimentStore(tmp_path).get(draft["experiment_id"]) == draft
    monkeypatch.setattr(store, "_write_index", original)
    assert store.freeze_plan(draft["experiment_id"], 1)["status"] == "frozen"


def test_hash_valid_but_wrong_revision_chain_is_rejected(tmp_path):
    store = ExperimentStore(tmp_path)
    first = store.save_brief({"question": "First question"})
    second = store.save_brief({"question": "Second question"}, first["experiment_id"], 1)
    path = tmp_path / f"{first['experiment_id']}.json"
    index = json.loads(path.read_bytes())
    revision_path = tmp_path / f"object_{second['revisions'][-1]['sha256']}.json"
    revision = json.loads(revision_path.read_bytes())
    revision["previous_sha256"] = "a" * 64
    raw = encode(revision)
    sha = digest(raw)
    (tmp_path / f"object_{sha}.json").write_bytes(raw)
    index["index"]["revisions"][-1]["sha256"] = sha
    index["sha256"] = digest(encode(index["index"]))
    path.write_bytes(encode(index))
    with pytest.raises(ValueError, match="corrupt"):
        store.get(first["experiment_id"])


@pytest.mark.parametrize("value", [True, 0, 101, "20"])
def test_invalid_listing_limits_do_not_read_records(tmp_path, value):
    with pytest.raises(ValueError, match="limit"):
        ExperimentStore(tmp_path).list(value)
