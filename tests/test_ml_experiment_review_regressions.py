"""Independent-review reproductions for directory boundaries and count domains."""

import os

import pytest
from pydantic import ValidationError

from src.product.ml_experiment_measurements import Measurement
from src.product.ml_experiment_sources import read_json
from src.product.ml_experiment_store import ExperimentStore
from tests.ml_experiment_result_fixtures import paired_result, planned_brief


def test_ancestor_replacement_after_store_creation_cannot_redirect_writes(tmp_path):
    selected = tmp_path / "selected"
    store = ExperimentStore(selected / "records")
    saved = store.save_brief({"question": "Original"})
    outside = tmp_path / "outside"
    (outside / "records").mkdir(parents=True)
    selected.rename(tmp_path / "original")
    selected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        store.save_brief({"question": "Must not escape"})
    assert list((outside / "records").iterdir()) == []
    original = ExperimentStore(tmp_path / "original/records")
    assert original.get(saved["experiment_id"])["brief"]["question"] == "Original"


def test_root_ancestor_swap_during_source_open_cannot_read_outside(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    root = selected / "exports"
    root.mkdir(parents=True)
    (root / "result.json").write_text('{"origin":"selected"}')
    outside = tmp_path / "outside"
    (outside / "exports").mkdir(parents=True)
    (outside / "exports/result.json").write_text('{"origin":"outside"}')
    original_open = os.open
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        # Swap after read_json's preliminary checks, just before its root open.
        if not swapped and os.fspath(path) in {root.anchor, str(root)}:
            swapped = True
            selected.rename(tmp_path / "original")
            selected.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(ValueError, match="unsafe"):
        read_json(root, "result.json")
    assert swapped


@pytest.mark.parametrize("value", [-999.5, -1.0, 0.5])
def test_invalid_count_cannot_enter_a_persisted_success(tmp_path, value):
    brief = planned_brief()
    brief["metrics"][0].update(unit="count", measurement_kind="other")
    store = ExperimentStore(tmp_path / "records")
    draft = store.save_brief(brief)
    frozen = store.freeze_plan(draft["experiment_id"], 1)
    result = paired_result(brief)
    for arm in (result["baseline"], result["candidate"]):
        arm["measurements"][0]["unit"] = "count"
    result["candidate"]["measurements"][0]["value"] = value
    with pytest.raises(ValidationError, match="Count measurement"):
        store.record_result(draft["experiment_id"], frozen["plan_sha256"], result)
    assert store.get(draft["experiment_id"])["results"] == []


def test_count_domain_allows_event_totals_and_summaries_keep_synthetic_label(tmp_path):
    measurement = paired_result()["baseline"]["measurements"][0]
    # Event totals may exceed the number of evaluated cases; signed scores use other units.
    assert Measurement.model_validate({**measurement, "unit": "count"}).value == 100.0
    assert Measurement.model_validate({**measurement, "unit": "signed_score", "value": -0.5})
    store = ExperimentStore(tmp_path)
    saved = store.save_brief(planned_brief())
    frozen = store.freeze_plan(saved["experiment_id"], 1)
    result = store.record_result(saved["experiment_id"], frozen["plan_sha256"], paired_result())
    assert result["results"][0]["data_kind"] == "synthetic"
