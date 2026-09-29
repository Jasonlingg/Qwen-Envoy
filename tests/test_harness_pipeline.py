"""Offline vertical slice: frozen source -> verified bundle -> draft -> inbox."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.artifacts import content_hash
from src.harness.draft import DRAFT_VERSION
from src.harness.pipeline import ReplayConfig, ReplayInvestigation, replay_question_draft
from src.harness.run_record import RunRecord
from src.harness.staging import PublishedButUnconfirmedError, stage_drafts


@pytest.fixture
def inputs(tmp_path):
    text = "The paper reports exact evidence.\nA second paragraph is not in the selected span."
    snapshot = tmp_path / "snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    arxiv_id = "2609.11111v1"
    doc_id = "arxiv_2609_11111v1"
    doc = {
        "doc_id": doc_id,
        "title": "Fixture paper",
        "text": text,
        "sections": [{"section": "Abstract", "start": 0, "end": len(text)}],
        "metadata": {
            "arxiv_id": arxiv_id,
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            "coverage": "abstract_only",
        },
    }
    corpus_path = corpus / f"{doc_id}.json"
    corpus_path.write_text(json.dumps(doc), encoding="utf-8")
    manifest = {
        "schema_version": "research-snapshot-v1",
        "status": "complete",
        "corpus_hash": content_hash(corpus),
        "papers": [{"doc_id": doc_id, "arxiv_id": arxiv_id}],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    quote = "reports exact evidence"
    start = text.index(quote)
    span = {
        "doc_id": doc_id,
        "source_revision": arxiv_id,
        "source_version_hash": hashlib.sha256(doc["text"].encode()).hexdigest(),
        "start": start,
        "end": start + len(quote),
        "quote": quote,
    }
    vault = tmp_path / "obsidian"
    vault.mkdir()
    config = ReplayConfig(
        run_id="replay-1",
        question_id="fixture-q1",
        question_split="test_fixture",
        question="What does the paper report?",
        qwen_checkpoint="fixture-qwen",
        nemotron_model="fixture-nemotron",
        index_version="fixture-index-v1",
        tool_protocol_version="research-tools-v1",
        qwen_prompt_sha256="a" * 64,
        drafter_prompt_sha256="b" * 64,
        seed=42,
        hardware="cpu-fixture",
        code_revision="test",
        max_model_rounds=3,
        max_tool_calls=6,
        decoding={"temperature": 0},
    )
    return snapshot, vault, config, span


def test_replay_stages_only_a_verified_unreviewed_draft(inputs, tmp_path):
    snapshot, vault, config, span = inputs
    seen = []

    def drafter(question, bundle):
        seen.append((question, bundle))
        return {
            "schema_version": DRAFT_VERSION,
            "claims": [
                {
                    "text": "The paper reports evidence.",
                    "passage_ids": [bundle["passages"][0]["passage_id"]],
                }
            ],
            "open_questions": [
                {
                    "text": "Would the finding replicate?",
                    "passage_ids": [bundle["passages"][0]["passage_id"]],
                }
            ],
            "proposed_applications": [],
        }

    result = replay_question_draft(
        snapshot=snapshot,
        vault=vault,
        run_root=tmp_path / "runs",
        config=config,
        investigation=ReplayInvestigation(
            candidate_spans=[span],
            returned_doc_ids={span["doc_id"]},
            candidate_findings=[{"text": "Untrusted Qwen proposal", "span_indices": [0]}],
        ),
        drafter=drafter,
    )
    assert result["status"] == "staged_inbox"
    assert result["ledger_status"] == "fixture_controlled_not_live_broker"
    assert len(seen) == 1
    assert seen[0][0] == config.question
    assert "second paragraph" not in json.dumps(seen[0][1])
    note = Path(result["drafts"][0])
    assert note.is_relative_to(vault / "_inbox" / "replay-1")
    assert "review_status: agent_authored_draft" in note.read_text()
    assert "reports exact evidence" in note.read_text()
    assert not (vault / "library").exists()
    assert (tmp_path / "runs" / "replay-1" / "evidence_bundle.json").is_file()


def test_forged_quote_fails_before_drafter_or_vault_write(inputs, tmp_path):
    snapshot, vault, config, span = inputs
    span["quote"] = "invented result"
    called = []

    def drafter(question, bundle):
        called.append(question)
        raise AssertionError("must not draft from forged evidence")

    with pytest.raises(ValueError, match="quote does not match"):
        replay_question_draft(
            snapshot=snapshot,
            vault=vault,
            run_root=tmp_path / "runs",
            config=config,
            investigation=ReplayInvestigation(
                candidate_spans=[span],
                returned_doc_ids={span["doc_id"]},
            ),
            drafter=drafter,
        )
    assert called == []
    assert list(vault.rglob("*")) == []
    events = (tmp_path / "runs" / "replay-1" / "events.jsonl").read_text()
    assert '"state":"invalid_provenance"' in events


def test_uncited_draft_fails_without_staging(inputs, tmp_path):
    snapshot, vault, config, span = inputs

    def drafter(question, bundle):
        return {
            "schema_version": DRAFT_VERSION,
            "claims": [{"text": "Unsupported claim", "passage_ids": []}],
            "open_questions": [],
            "proposed_applications": [],
        }

    with pytest.raises(ValueError, match="passage IDs"):
        replay_question_draft(
            snapshot=snapshot,
            vault=vault,
            run_root=tmp_path / "runs",
            config=config,
            investigation=ReplayInvestigation(
                candidate_spans=[span], returned_doc_ids={span["doc_id"]}
            ),
            drafter=drafter,
        )
    assert list(vault.rglob("*")) == []
    events = (tmp_path / "runs" / "replay-1" / "events.jsonl").read_text()
    assert '"state":"invalid_draft"' in events


def test_ledger_failure_after_staging_reports_visible_draft(inputs, tmp_path, monkeypatch):
    snapshot, vault, config, span = inputs
    original = RunRecord.append_event

    def fail_final_event(self, state, details):
        if state == "staged_inbox":
            raise OSError("simulated disk failure")
        return original(self, state, details)

    monkeypatch.setattr(RunRecord, "append_event", fail_final_event)

    def drafter(question, bundle):
        return {
            "schema_version": DRAFT_VERSION,
            "claims": [
                {
                    "text": "The paper reports evidence.",
                    "passage_ids": [bundle["passages"][0]["passage_id"]],
                }
            ],
            "open_questions": [],
            "proposed_applications": [],
        }

    result = replay_question_draft(
        snapshot=snapshot,
        vault=vault,
        run_root=tmp_path / "runs",
        config=config,
        investigation=ReplayInvestigation(
            candidate_spans=[span], returned_doc_ids={span["doc_id"]}
        ),
        drafter=drafter,
    )
    assert result["status"] == "staged_inbox_ledger_pending"
    assert result["recovery_required"] is True
    assert Path(result["drafts"][0]).is_file()
    events = (tmp_path / "runs" / "replay-1" / "events.jsonl").read_text()
    assert '"state":"failed"' not in events


def test_uncertain_stage_commit_reports_visible_batch(inputs, tmp_path, monkeypatch):
    snapshot, vault, config, span = inputs

    def publish_then_fail(vault_arg, run_id, files):
        staged = stage_drafts(vault_arg, run_id, files)
        raise PublishedButUnconfirmedError(staged.run_dir)

    monkeypatch.setattr("src.harness.pipeline.stage_drafts", publish_then_fail)

    def drafter(question, bundle):
        return {
            "schema_version": DRAFT_VERSION,
            "claims": [
                {
                    "text": "The paper reports evidence.",
                    "passage_ids": [bundle["passages"][0]["passage_id"]],
                }
            ],
            "open_questions": [],
            "proposed_applications": [],
        }

    result = replay_question_draft(
        snapshot=snapshot,
        vault=vault,
        run_root=tmp_path / "runs",
        config=config,
        investigation=ReplayInvestigation(
            candidate_spans=[span], returned_doc_ids={span["doc_id"]}
        ),
        drafter=drafter,
    )
    assert result["status"] == "stage_commit_uncertain"
    assert result["recovery_required"] is True
    assert Path(result["drafts"][0]).is_file()
