"""The retained frozen pool, source-bound reward and unavailable-evidence checks."""

import copy
from pathlib import Path

import pytest

from src.eval.reader_improvement import digest, load, write
from src.eval.reader_rl_validation import assess
from src.training.reader_data import (
    PILOT_REVIEW,
    load_bundle,
    reviewed_training_ids,
    score_episode,
)

BUNDLE = Path("data/rl/reader-facts-v1")


def test_actual_curated_pool_is_paper_disjoint_and_rejects_bad_upstream_labels():
    manifest, roles, contexts, cards = load_bundle(BUNDLE)
    assert [len(roles[r]) for r in ["train", "reward_validation", "model_evaluation_locked"]] == [
        82,
        8,
        12,
    ]
    assert manifest["human_review"] is None and manifest["training_pool_reuse"] is True
    assert len(load(BUNDLE / "rejections.json")) == 28
    assert len(load(BUNDLE / "history-exclusions.json")) == 11
    assert {c["case_id"] for c in roles["model_evaluation_locked"]}.isdisjoint(contexts)
    assert all(cards[c["case_id"]]["role"] == "train" for c in roles["train"])


def test_source_bound_keys_and_manifest_detect_one_byte_change(tmp_path):
    # Link the large immutable corpus files, copying only references we mutate.
    import os
    import shutil

    destination = tmp_path / "bundle"
    shutil.copytree(BUNDLE, destination, copy_function=os.link)
    source = destination / "keys.json"
    keys = copy.deepcopy(load(source))
    source.unlink()  # Never write through a hard link into the real source bundle.
    keys["cards"][0]["accepted_answers"][0]["answer"] += " invented"
    write(source, keys)
    with pytest.raises(ValueError, match="changed"):
        load_bundle(destination)


def test_unavailable_episode_never_becomes_a_zero_reward_training_example():
    with pytest.raises(ValueError, match="entire group"):
        score_episode({"infrastructure_ok": False}, {}, {})


def test_context_binding_changes_even_if_only_the_task_question_is_changed():
    _, roles, contexts, cards = load_bundle(BUNDLE)
    cid = roles["train"][0]["case_id"]
    modified = {**contexts[cid], "question": "a different experiment"}
    assert digest(modified) != cards[cid]["context_sha256"]


def test_all_84_saved_reference_contract_controls_still_reproduce():
    _, _, contexts, cards = load_bundle(BUNDLE)
    rows = load(Path("examples/reader-rl-ready/reference-validation/rows.json"))
    assert len(rows) == 84
    for row in rows:
        context, card = contexts[row["case_id"]], cards[row["case_id"]]
        inspected = context["windows"]
        if row["kind"] == "no_inspection":
            inspected = []
        elif row["kind"] == "uninspected_support":
            support = [s for claim in card["claims"] for s in claim["support"]]
            inspected = [
                next(
                    w
                    for w in inspected
                    if not any(w["start"] <= s["start"] < s["end"] <= w["end"] for s in support)
                )
            ]
        actual = assess(
            {"context_id": row["case_id"], "action": row["action"], "inspected_windows": inspected},
            context,
            card,
        )
        assert actual == row["actual"]
        assert actual["reward"] == row["expected"]


def test_pilot_holds_incomplete_alias_without_rewriting_frozen_reward():
    from src.eval.reader_rl_audit import submit

    _, roles, contexts, cards = load_bundle(BUNDLE)
    selected = reviewed_training_ids(BUNDLE, roles, cards)
    held = next(c for c in roles["train"] if c["case_id"] not in selected)
    card, context = cards[held["case_id"]], contexts[held["case_id"]]
    incomplete = "the public financial news dataset released by"
    assert incomplete in [v["answer"] for v in card["accepted_answers"]]
    support = [s for claim in card["claims"] for s in claim["support"]]
    spans = [{"doc_id": context["doc_id"], "start": s["start"], "end": s["end"]}
             for s in support]
    record = {"context_id": held["case_id"], "inspected_windows": context["windows"],
              "action": submit(incomplete, context["doc_id"], spans)}
    # The old finite contract still admits this bad alias. The new launch hold
    # prevents training it, without silently changing any frozen key or scorer.
    assert assess(record, context, card)["reward"] == 1
    assert len(selected) == 81 and held["case_id"] not in selected


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "heldout", "stale", "anonymous"])
def test_training_review_cannot_silently_omit_or_substitute_bound_labels(tmp_path, mutation):
    _, roles, _, cards = load_bundle(BUNDLE)
    review = copy.deepcopy(load(PILOT_REVIEW))
    if mutation == "missing":
        review["rows"].pop()
    elif mutation == "duplicate":
        review["rows"][-1] = copy.deepcopy(review["rows"][0])
    elif mutation == "heldout":
        review["rows"][0]["case_id"] = roles["model_evaluation_locked"][0]["case_id"]
    elif mutation == "stale":
        review["rows"][0]["card_sha256"] = "0" * 64
    else:
        review["reviewer_identity"] = ""
    path = tmp_path / "review.json"
    write(path, review)
    with pytest.raises(ValueError):
        reviewed_training_ids(BUNDLE, roles, cards, path)
