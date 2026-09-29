"""CPU-only replay of the fixed evidence-to-draft harness stages.

This is an integration seam for verified, host-recorded investigation results.
The current live Qwen sandbox does not yet provide a host tool-return ledger,
so this function is intentionally named a replay, not a production agent run.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.eval.artifacts import content_hash
from src.harness.draft import render_question_draft
from src.harness.evidence import EvidenceBundle, build_evidence_bundle
from src.harness.run_record import RunRecord
from src.harness.source import load_public_paper_snapshot
from src.harness.staging import PublishedButUnconfirmedError, stage_drafts


@dataclass(frozen=True)
class ReplayConfig:
    run_id: str
    question_id: str
    question_split: str
    question: str
    qwen_checkpoint: str
    nemotron_model: str
    index_version: str
    tool_protocol_version: str
    qwen_prompt_sha256: str
    drafter_prompt_sha256: str
    seed: int
    hardware: str
    code_revision: str
    max_model_rounds: int
    max_tool_calls: int
    decoding: Mapping[str, object]


@dataclass(frozen=True)
class ReplayInvestigation:
    """Candidate spans plus document IDs from fixture-controlled tool results."""

    candidate_spans: Sequence[Mapping[str, object]]
    returned_doc_ids: Collection[str]
    candidate_findings: Sequence[Mapping[str, object]] = ()
    unresolved_gaps: Sequence[str] = ()


def replay_question_draft(
    *,
    snapshot: Path,
    vault: Path,
    run_root: Path,
    config: ReplayConfig,
    investigation: ReplayInvestigation,
    drafter: Callable[[str, EvidenceBundle], Mapping[str, object]],
) -> dict[str, Any]:
    """Verify fixture evidence, check a structured draft, and stage one note.

    The injected drafter receives only the trusted question and verified bundle.
    Model-authored code or raw tool transcripts are never passed to it. The
    ``returned_doc_ids`` input is meaningful only when supplied by a host broker;
    fixture replay checks the rest of the path but is not that broker.
    """
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    vault = Path(vault).expanduser().resolve(strict=True)
    if snapshot.is_relative_to(vault):
        raise ValueError("paper snapshot must be outside the vault")
    if not callable(drafter):
        raise ValueError("drafter must be callable")
    manifest, _ = load_public_paper_snapshot(snapshot)
    inputs = {
        "task_kind": "question_replay",
        "question_id": config.question_id,
        "question_split": config.question_split,
        "question": config.question,
        "corpus_hash": manifest["corpus_hash"],
        "snapshot_path": str(snapshot),
        "snapshot_manifest_sha256": content_hash(snapshot / "manifest.json"),
        "index_version": config.index_version,
        "tool_protocol_version": config.tool_protocol_version,
        "qwen_checkpoint": config.qwen_checkpoint,
        "nemotron_model": config.nemotron_model,
        "qwen_prompt_sha256": config.qwen_prompt_sha256,
        "drafter_prompt_sha256": config.drafter_prompt_sha256,
        "seed": config.seed,
        "hardware": config.hardware,
        "code_revision": config.code_revision,
        "max_model_rounds": config.max_model_rounds,
        "max_tool_calls": config.max_tool_calls,
        "decoding": dict(config.decoding),
        "ledger_status": "fixture_controlled_not_live_broker",
    }
    record = RunRecord(run_root, vault, config.run_id, inputs)
    phase = "recording"
    try:
        record.save_artifact(
            "investigation.json",
            {
                "candidate_spans": list(investigation.candidate_spans),
                "returned_doc_ids": sorted(investigation.returned_doc_ids),
                "candidate_findings": list(investigation.candidate_findings),
                "unresolved_gaps": list(investigation.unresolved_gaps),
            },
        )
        record.append_event("investigating", {"mode": "fixture_replay"})
        phase = "verifying"
        bundle = build_evidence_bundle(
            snapshot=snapshot,
            run_id=config.run_id,
            question=config.question,
            corpus_hash=manifest["corpus_hash"],
            index_version=config.index_version,
            tool_protocol_version=config.tool_protocol_version,
            qwen_checkpoint=config.qwen_checkpoint,
            candidate_spans=investigation.candidate_spans,
            returned_doc_ids=investigation.returned_doc_ids,
            candidate_findings=investigation.candidate_findings,
            unresolved_gaps=investigation.unresolved_gaps,
        )
        record.save_artifact("evidence_bundle.json", bundle)
        record.append_event("evidence_verified", {"passages": len(bundle["passages"])})
        phase = "drafting"
        draft = drafter(config.question, bundle)
        phase = "validating_draft"
        note = render_question_draft(bundle, draft)
        record.save_artifact("structured_draft.json", draft)
        record.append_event("citation_checked", {"claims": len(draft["claims"])})
        phase = "staging"
        try:
            result = stage_drafts(vault, config.run_id, {f"Questions/{config.run_id}.md": note})
        except PublishedButUnconfirmedError as exc:
            ledger_error = None
            try:
                record.append_event(
                    "stage_commit_uncertain",
                    {
                        "staged_run_dir": str(exc.run_dir),
                        "manifest": str(exc.manifest_path),
                    },
                )
            except Exception as log_exc:
                ledger_error = f"{type(log_exc).__name__}: {log_exc}"
            return {
                "status": "stage_commit_uncertain",
                "run_id": config.run_id,
                "run_record": str(record.directory),
                "drafts": [str(exc.run_dir / "Questions" / f"{config.run_id}.md")],
                "semantic_support_status": "not_reviewed",
                "ledger_status": "fixture_controlled_not_live_broker",
                "recovery_required": True,
                "ledger_error": ledger_error,
                "warning": "Draft is visible but inbox directory durability is unconfirmed.",
            }
        stage_event = {"files": sorted(result.files), "hashes": dict(result.hashes)}
        try:
            record.append_event("staged_inbox", stage_event)
        except Exception as exc:
            # The atomic inbox batch is already visible. Reporting an ordinary
            # failure would incorrectly imply that no draft was written.
            return {
                "status": "staged_inbox_ledger_pending",
                "run_id": config.run_id,
                "run_record": str(record.directory),
                "drafts": [str(path) for path in result.files.values()],
                "semantic_support_status": "not_reviewed",
                "ledger_status": "fixture_controlled_not_live_broker",
                "recovery_required": True,
                "ledger_error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "status": "staged_inbox",
            "run_id": config.run_id,
            "run_record": str(record.directory),
            "drafts": [str(path) for path in result.files.values()],
            "semantic_support_status": "not_reviewed",
            "ledger_status": "fixture_controlled_not_live_broker",
        }
    except Exception as exc:
        failure = {
            "verifying": "invalid_provenance",
            "validating_draft": "invalid_draft",
        }.get(phase, "failed")
        record.append_event(failure, {"error_type": type(exc).__name__, "error": str(exc)})
        raise
