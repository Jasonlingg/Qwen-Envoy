"""Product investigator contract: bounded, sandboxed, provenance-checked handoff."""

from __future__ import annotations

import hashlib
import json

from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.product.memory import freeze_vault
from src.product.qwen_investigator import (
    NEMOTRON_PAPER_SYSTEM_PROMPT,
    PUBLIC_PAPER_SYSTEM_PROMPT,
    QwenInvestigator,
    prompt_for_source_domain,
)
from src.research.agent import load_snapshot


class FakePolicy:
    def __init__(self, actions: list[str]):
        self.actions = iter(actions)
        self.observations: list[str] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, observation: str) -> str:
        self.observations.append(observation)
        return next(self.actions)


class FakeREPL:
    def __init__(self):
        self.started = False
        self.killed = False
        self.executed: list[tuple[str, int]] = []

    def start_session(self) -> None:
        self.started = True

    def execute(self, code: str, timeout: int = 30) -> str:
        self.executed.append((code, timeout))
        return "[{\"doc_id\": \"source\"}]"

    def kill_session(self) -> None:
        self.killed = True


def _snapshot(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Finding.md").write_text(
        "---\nrecord_id: finding-one\nkind: lesson\neffective_date: 2026-09-15\n---\n"
        "# Finding\nOur earlier retrieval test failed because source coverage was incomplete.\n"
    )
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    manifest, docs = load_snapshot(snapshot)
    doc_id, document = next(iter(docs.items()))
    return snapshot, manifest, doc_id, document


def _investigator(actions, repl, *, max_steps=5):
    policy = FakePolicy(actions)
    worker = QwenInvestigator(
        policy_factory=lambda: policy,
        model_identity={"checkpoint": "local-v5", "seed": 42, "temperature": 0},
        max_steps=max_steps,
        repl_factory=lambda corpus: repl,
    )
    return worker, policy


def test_investigator_validates_exact_spans_and_enriches_source(tmp_path):
    snapshot, manifest, doc_id, document = _snapshot(tmp_path)
    quote = "source coverage was incomplete"
    start = document["text"].index(quote)
    evidence = json.dumps([{"doc_id": doc_id, "start": start, "end": start + len(quote)}])
    actions = [
        'hits = search("retrieval source coverage")\nprint(hits)',
        f'paper = read("{doc_id}")\nprint(paper)',
        f'SUBMIT: The source coverage was incomplete. CITATIONS: ["{doc_id}"] '
        f'EVIDENCE: {evidence}',
    ]
    repl = FakeREPL()
    worker, policy = _investigator(actions, repl)

    packet = worker.investigate("Why did retrieval fail?", snapshot)

    assert packet["status"] == "evidence_found"
    assert packet["corpus_hash"] == manifest["corpus_hash"]
    assert packet["retriever_protocol"] == "code-execution"
    assert packet["candidate_claims"][0]["evidence_ids"] == ["E1"]
    assert packet["evidence"][0]["quote"] == quote
    assert packet["evidence"][0]["source_path"] == "Finding.md"
    assert packet["evidence"][0]["record"]["record_id"] == "finding-one"
    assert packet["model_identity"]["checkpoint"] == "local-v5"
    assert len(packet["trajectory"]) == 3
    assert policy.resets == 1
    assert repl.started and repl.killed


def test_investigator_rejects_mismatched_quote_then_accepts_correction(tmp_path):
    snapshot, _, doc_id, document = _snapshot(tmp_path)
    quote = "source coverage was incomplete"
    start = document["text"].index(quote)
    bad = json.dumps([{"doc_id": doc_id, "start": start, "end": start + len(quote),
                       "quote": "invented quote"}])
    good = json.dumps([{"doc_id": doc_id, "start": start, "end": start + len(quote)}])
    actions = [
        'print(search("coverage"))',
        f'print(read("{doc_id}"))',
        f'SUBMIT: Coverage was incomplete. CITATIONS: ["{doc_id}"] EVIDENCE: {bad}',
        f'SUBMIT: Coverage was incomplete. CITATIONS: ["{doc_id}"] EVIDENCE: {good}',
    ]
    worker, policy = _investigator(actions, FakeREPL())

    packet = worker.investigate("What failed?", snapshot)

    assert packet["status"] == "evidence_found"
    assert "quote does not match" in policy.observations[-1]
    assert packet["evidence"][0]["quote"] == quote


def test_investigator_only_reports_no_evidence_after_explicit_investigation(tmp_path):
    snapshot, _, doc_id, _ = _snapshot(tmp_path)
    repl = FakeREPL()
    worker, _ = _investigator([
        'print(search("completed outcome"))',
        f'print(read("{doc_id}"))',
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
    ], repl)

    packet = worker.investigate("What was the measured outcome?", snapshot)

    assert packet["status"] == "no_evidence"
    assert packet["candidate_claims"] == []
    assert packet["evidence"] == []
    assert "does not prove" in packet["warning"]
    assert len(repl.executed) == 2


def test_investigator_does_not_call_uninvestigated_abstention_no_evidence(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)
    repl = FakeREPL()
    worker, _ = _investigator(
        ["SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"] * 2,
        repl, max_steps=2,
    )

    packet = worker.investigate("Unknown outcome?", snapshot)

    assert packet["status"] == "incomplete"
    assert not packet["evidence"]
    assert repl.killed


def test_missing_document_results_do_not_count_as_investigation(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)

    class MissingDocumentREPL(FakeREPL):
        def execute(self, code: str, timeout: int = 30) -> str:
            self.executed.append((code, timeout))
            if 'search("outcome")' in code:
                return "[{'doc_id': 'real-source'}]"
            return "[{'error': \"Document 'invented' not found\"}]"

    worker, policy = _investigator([
        'print(search("outcome"))',
        'print(search_within("invented", "outcome"))',
        'print(search_within("also-invented", "outcome"))',
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
    ], MissingDocumentREPL(), max_steps=4)

    packet = worker.investigate("What was the measured outcome?", snapshot)

    assert packet["status"] == "incomplete"
    assert packet["document_tool_steps"] == 1
    assert packet["inspection_steps"] == 0
    assert "needs two distinct successful" in packet["trajectory"][-1]["observation"]
    assert len(policy.observations) == 4


def test_investigator_requires_discovery_before_id_tool(tmp_path):
    snapshot, _, doc_id, _ = _snapshot(tmp_path)
    repl = FakeREPL()
    worker, _ = _investigator([
        'print(search_within("guessed-id", "outcome"))',
        'print(search("outcome"))',
        f'print(read("{doc_id}"))',
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
    ], repl, max_steps=4)

    packet = worker.investigate("What was the measured outcome?", snapshot)

    assert packet["status"] == "no_evidence"
    assert "first call search" in packet["trajectory"][0]["observation"]
    assert [action for action, _ in repl.executed] == [
        'print(search("outcome"))', f'print(read("{doc_id}"))',
    ]


def test_empty_search_window_does_not_justify_abstention(tmp_path):
    snapshot, _, doc_id, _ = _snapshot(tmp_path)

    class EmptyWindowREPL(FakeREPL):
        def execute(self, code: str, timeout: int = 30) -> str:
            self.executed.append((code, timeout))
            return "[]" if "search_within" in code else f"[{{'doc_id': '{doc_id}'}}]"

    worker, _ = _investigator([
        'print(search("outcome"))',
        f'print(search_within("{doc_id}", "outcome"))',
        "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
    ], EmptyWindowREPL(), max_steps=3)

    packet = worker.investigate("What was the measured outcome?", snapshot)

    assert packet["status"] == "incomplete"
    assert packet["document_tool_steps"] == 2
    assert packet["inspection_steps"] == 0
    assert "inspected source text" in packet["trajectory"][-1]["observation"]


def test_investigator_fails_closed_when_sandbox_image_missing(tmp_path, monkeypatch):
    snapshot, _, _, _ = _snapshot(tmp_path)
    monkeypatch.setattr(
        "src.product.qwen_investigator.PersistentREPL._docker_available",
        lambda image: False,
    )
    called = False

    def policy_factory():
        nonlocal called
        called = True
        raise AssertionError("model must not load without a sandbox")

    worker = QwenInvestigator(policy_factory, {"checkpoint": "local-v5"})
    packet = worker.investigate("Why?", snapshot)

    assert packet["status"] == "unavailable"
    assert packet["trajectory"] == []
    assert not called


def test_public_paper_domain_uses_paper_prompt_without_changing_tool_handoff(tmp_path):
    snapshot, _, doc_id, document = _snapshot(tmp_path)
    quote = "source coverage was incomplete"
    start = document["text"].index(quote)
    policy = FakePolicy([
        'print(search("retrieval source coverage"))',
        f'print(read("{doc_id}"))',
        f'SUBMIT: Source coverage was incomplete. CITATIONS: ["{doc_id}"] '
        f'EVIDENCE: [{{"doc_id":"{doc_id}","start":{start},"end":{start + len(quote)}}}]',
    ])
    worker = QwenInvestigator(
        policy_factory=lambda: policy,
        model_identity={"checkpoint": "test-only"},
        source_domain="public_papers",
        repl_factory=lambda corpus: FakeREPL(),
    )

    packet = worker.investigate("Why did retrieval fail?", snapshot)

    assert packet["status"] == "evidence_found"
    assert packet["source_domain"] == "public_papers"
    assert packet["tool_search_version"] == SEARCH_PROTOCOL_VERSION
    assert packet["tool_preamble_sha256"] == hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest()
    assert packet["question_id"].startswith("paper_")
    assert packet["evidence"][0]["quote"] == quote
    assert policy.observations[0].startswith(PUBLIC_PAPER_SYSTEM_PROMPT)
    assert "personal Obsidian vault" not in policy.observations[0]
    assert "search()" in policy.observations[0]
    assert "read()" in policy.observations[0]
    assert "EVIDENCE:" in policy.observations[0]


def test_unknown_source_domain_is_rejected_before_model_or_sandbox():
    try:
        QwenInvestigator(
            policy_factory=lambda: None,
            model_identity={"checkpoint": "test-only"},
            source_domain="unknown",
        )
    except ValueError as exc:
        assert "unsupported Qwen source domain" in str(exc)
    else:
        raise AssertionError("unknown source domain must be rejected")
    assert prompt_for_source_domain("public_papers") == PUBLIC_PAPER_SYSTEM_PROMPT


def test_nemotron_paper_protocol_submits_with_bounded_reminder(tmp_path):
    snapshot, _, doc_id, document = _snapshot(tmp_path)
    quote = "source coverage was incomplete"
    start = document["text"].index(quote)
    policy = FakePolicy([
        'print(search("retrieval source coverage"))',
        f'print(read("{doc_id}"))',
        f'SUBMIT: Coverage was incomplete. CITATIONS: ["{doc_id}"] '
        f'EVIDENCE: [{{"doc_id":"{doc_id}","start":{start},"end":{start + len(quote)}}}]',
    ])
    worker = QwenInvestigator(
        policy_factory=lambda: policy,
        model_identity={"model_id": "nvidia/Nemotron-3_5-Lightning"},
        source_domain="public_papers",
        protocol_variant="nemotron",
        max_steps=5,
        repl_factory=lambda corpus: FakeREPL(),
    )

    packet = worker.investigate("Why did retrieval fail?", snapshot)

    assert packet["status"] == "evidence_found"
    assert packet["protocol_variant"] == "nemotron"
    assert packet["system_prompt_sha256"] == hashlib.sha256(
        NEMOTRON_PAPER_SYSTEM_PROMPT.encode()
    ).hexdigest()
    assert policy.observations[0].startswith(NEMOTRON_PAPER_SYSTEM_PROMPT)
    assert "There is no\nsubmit() Python function" in policy.observations[0]
    assert "Runner reminder: 3 action(s) remain" in policy.observations[2]
    assert packet["trajectory"][2]["protocol_reminder"] in policy.observations[2]
    assert "Qwen" not in packet["warning"]


def test_nemotron_protocol_rejects_other_source_domains():
    try:
        QwenInvestigator(
            policy_factory=lambda: None,
            model_identity={"model_id": "nemotron"},
            source_domain="personal_vault",
            protocol_variant="nemotron",
        )
    except ValueError as exc:
        assert "only for public papers" in str(exc)
    else:
        raise AssertionError("Nemotron paper protocol must not change the vault")
