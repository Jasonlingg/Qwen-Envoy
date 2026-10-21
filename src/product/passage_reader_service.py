"""Explicit Qwen passage reading with durable single-attempt MCP note IDs."""
from __future__ import annotations

import os
from pathlib import Path

from src.product.ml_experiment_io import LocalFiles, decode, digest, encode
from src.product.ml_experiment_research_notes import ResearchNote
from src.product.packet_verifier import verify_investigation_packet
from src.product.passage_scope import PASSAGES, make_scope
from src.research.agent import load_snapshot


class PassageReaderService:
    def __init__(self, store, library, reader_factory, reports_root: Path, *, data_kind="recorded"):
        if data_kind not in {"recorded", "synthetic"}:
            raise ValueError("unknown passage report data kind")
        self.store, self.library, self.reader_factory = store, library, reader_factory
        self.files = LocalFiles(reports_root)
        self.data_kind = data_kind

    def read(self, experiment_id, expected_revision, note_id, question, arxiv_id, windows):
        # Reuse the note schema for bounded text/ID validation, without inventing evidence.
        ResearchNote(note_id=note_id, question=question, uncertainties=["Not yet inspected"])
        view = self.store.get(experiment_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        reader, budget = self.reader_factory()
        if reader.reader_guidance != PASSAGES or reader.max_steps > 4:
            raise ValueError("passage reader requires passages-v1 and at most four actions")
        arguments = {"experiment_id": experiment_id, "expected_revision": expected_revision,
            "note_id": note_id, "question": question, "arxiv_id": arxiv_id, "windows": windows,
            "model_identity": reader.model_identity, "data_kind": self.data_kind,
            "prompt_sha256": digest(reader.system_prompt.encode())}
        fingerprint = digest(encode(arguments))
        key = digest(encode([experiment_id, note_id]))[:32]
        name = f"experiment_{key}.json"  # Private request index, in a separate selected root.
        with self.files.locked() as directory:
            try:
                os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                if view["revision"] != expected_revision:
                    raise ValueError("Stale expected_revision; resume before delegating")
                if any(n["note_id"] == note_id for n in view["research_notes"]):
                    raise ValueError("note_id is already recorded; no inference was dispatched")
                # Check source, ranges and snapshot before consuming the request identity.
                document = self.library.document(arxiv_id)  # Already open; never fetch here.
                scope = make_scope({document["doc_id"]: document}, windows)
                snapshot = self.library.view([arxiv_id])
                self.files.write(directory, name, encode({"fingerprint": fingerprint,
                    "state": "started", "source_scope": scope}))
            else:
                state = decode(self.files.read(directory, name))
                if state.get("fingerprint") != fingerprint:
                    raise ValueError("note_id is bound to a different passage request")
                if state.get("state") == "completed":
                    return self.files.read_object(directory, state["report_sha256"])
                return {"status": "previous_attempt_incomplete", "experiment_id": experiment_id,
                    "note_id": note_id, "model_calls_retried": 0,
                    "warning": "An earlier attempt started; no automatic inference retry."}
        report = {"schema_version": "qwen-passage-read-v1", "experiment_id": experiment_id,
            "brief_revision": expected_revision, "note_id": note_id, "request_sha256": fingerprint,
            "source_scope": scope, "note_attached": False, "semantic_support": "not_reviewed",
            "data_kind": self.data_kind,
            "warning": "Candidate evidence from selected passages; host semantic review required."}
        try:
            packet = reader.investigate(question, snapshot, passage_windows=windows)
            report["packet"] = packet
            manifest, docs = load_snapshot(snapshot)
            checked = verify_investigation_packet(packet, manifest, docs,
                allowed_prompt_sha256s=frozenset({digest(reader.system_prompt.encode())}),
                expected_reader_mode="assigned", expected_reader_guidance=PASSAGES,
                expected_passage_windows=windows)
            report.update(status=checked["status"], evidence=checked["evidence"],
                          candidate_answer=checked["candidate_answer"], verified=checked)
            refs = [{"kind": "paper_span", "reference_id": f"span_{i}",
                     "arxiv_id": item["arxiv_id"], "start": item["start"], "end": item["end"],
                     "quote": item["quote"]} for i, item in enumerate(checked["evidence"], 1)]
            findings = ([{"claim": checked["candidate_answer"],
                          "reference_ids": [r["reference_id"] for r in refs]}] if refs else [])
            missing = [g["missing"] for g in checked.get("research_gaps", [])]
            note = {"note_id": note_id, "question": question, "findings": findings,
                "references": refs, "origin": "qwen_research", "data_kind": self.data_kind,
                "uncertainties": ["Host semantic review is pending; selected passages only.",
                    *missing][:8] if refs or missing else [
                    "Reader returned no verified evidence. This is not proof of absence."]}
            report["research_note_input"] = note
            try:
                attached = self.store.record_research(experiment_id, expected_revision, note)
                report.update(note_attached=True,
                    research_note_sha256=attached["selected_research_note"]["sha256"])
            except (ValueError, OSError) as exc:
                report["attachment_error"] = str(exc)[:500]
                # Retain checked evidence; revision changes must not trigger another inference.
        except Exception as exc:
            report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        if budget is not None:
            report["budget"] = budget.summary()
        with self.files.locked() as directory:
            report_hash = self.files.object(directory, report)
            self.files.write(directory, name, encode({"fingerprint": fingerprint,
                "state": "completed", "report_sha256": report_hash}))
        return report
