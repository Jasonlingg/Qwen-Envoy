"""Chat orchestration over a frozen vault; fake clients make model calls observable."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copytree
from threading import Event

from fastapi.testclient import TestClient

from scripts.personal_memory_web import _configured_key, create_app, make_qwen_investigator
from src.product.chat import ChatService
from src.product.memory import freeze_vault, inspect_evidence
from src.product.qwen_investigator import PUBLIC_PAPER_SYSTEM_PROMPT
from src.research.agent import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)
    return vault, snapshot


class FakeNemotron:
    model = "nvidia/test-nemotron"

    def __init__(self, calls: list, *, investigate: bool = True):
        self.calls = calls
        self.should_investigate = investigate

    def plan(self, question: str, history: list[dict] | None = None) -> dict:
        self.calls.append(("plan", question, history))
        return {"investigate": self.should_investigate,
                "query": "phone drawer next action" if self.should_investigate else ""}

    def answer(self, question: str, evidence_packet: dict,
               history: list[dict] | None = None) -> dict:
        self.calls.append(("answer", question, evidence_packet, history))
        if evidence_packet["evidence"]:
            assert evidence_packet["evidence"][0]["quote"]
            return {
                "answer": "The July correction replaced the June plan [E1].",
                "claims": [{"text": "The July correction replaced the June plan",
                            "evidence_ids": ["E1"]}],
                "limitations": ["A source span is not a causal test."],
                "checks": {"semantic_support": "not_reviewed"},
                "model": self.model, "served_model": self.model,
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            }
        return {"answer": "I do not have source evidence for that yet.",
                "claims": [], "limitations": ["No cited vault passages."],
                "checks": {"semantic_support": "not_reviewed"},
                "model": self.model, "served_model": self.model,
                "usage": {"prompt_tokens": 10, "completion_tokens": 8}}


class FakeInvestigator:
    def __init__(self, calls: list, status: str = "evidence_found"):
        self.calls = calls
        self.status = status

    def investigate(self, question: str, snapshot: Path) -> dict:
        self.calls.append(("investigate", question))
        review = inspect_evidence(snapshot, question)
        return {
            **review,
            "retriever_protocol": "code-execution",
            "status": self.status,
            # The fake worker cites one source, leaving linked notes uncited
            # so the graph/source-link behavior is exercised independently.
            "evidence": review["evidence"][:1] if self.status == "evidence_found" else [],
            "candidate_claims": [],
            "model_identity": {"checkpoint": "test-v5"},
            "trajectory": [{"action": "search"}],
        }


def test_chat_routes_to_qwen_then_nemotron_with_exact_sources(tmp_path):
    vault, snapshot = _snapshot(tmp_path)
    calls: list = []
    service = ChatService(coordinator=FakeNemotron(calls),
                          investigator=FakeInvestigator(calls))
    before = list(vault.rglob("*.md"))

    result = service.reply("What did I change?", snapshot, history=[])

    assert [entry[0] for entry in calls] == ["plan", "investigate", "answer"]
    assert calls[1][1] == "phone drawer next action"
    assert result["answer_status"] == "answered"
    assert result["retrieval"]["mode"] == "qwen_code_execution"
    assert result["retrieval"]["model_identity"]["checkpoint"] == "test-v5"
    assert result["answer_model"] == "nvidia/test-nemotron"
    assert result["answer_served_model"] == "nvidia/test-nemotron"
    assert result["answer_usage"]["completion_tokens"] == 8
    assert result["evidence"] and result["related_notes"]
    cited_doc_ids = {item["doc_id"] for item in result["evidence"]}
    assert all(item["doc_id"] not in cited_doc_ids for item in result["related_notes"])
    assert result["checks"]["semantic_support"] == "not_reviewed"
    assert list(vault.rglob("*.md")) == before
    _, docs = load_snapshot(snapshot)
    for item in result["evidence"]:
        assert docs[item["doc_id"]]["text"][item["start"]:item["end"]] == item["quote"]
    assert "trajectory" not in calls[-1][2]


def test_missing_models_returns_labeled_evidence_only(tmp_path):
    _, snapshot = _snapshot(tmp_path)

    result = ChatService().reply("phone drawer next action", snapshot)

    assert result["answer"] is None
    assert result["answer_status"] == "evidence_only"
    assert result["retrieval"]["mode"] == "lexical_paragraph_baseline"
    assert result["answer_model"] is None
    assert result["evidence"]
    assert "Nemotron" in result["limitations"][0]


def test_nemotron_can_answer_without_calling_vault_worker(tmp_path):
    _, snapshot = _snapshot(tmp_path)
    calls: list = []
    service = ChatService(coordinator=FakeNemotron(calls, investigate=False),
                          investigator=FakeInvestigator(calls))

    result = service.reply("Hello", snapshot)

    assert [entry[0] for entry in calls] == ["plan", "answer"]
    assert result["retrieval"]["mode"] == "not_requested"
    assert result["evidence"] == []
    assert result["answer_status"] == "answered"


def test_unavailable_qwen_uses_labeled_lexical_fallback(tmp_path):
    _, snapshot = _snapshot(tmp_path)
    calls: list = []
    service = ChatService(coordinator=FakeNemotron(calls),
                          investigator=FakeInvestigator(calls, status="unavailable"))

    result = service.reply("What did I change?", snapshot)

    assert result["retrieval"]["mode"] == "lexical_paragraph_baseline"
    assert result["retrieval"]["worker_status"] == "unavailable"
    assert result["answer_status"] == "answered"
    assert [entry[0] for entry in calls] == ["plan", "investigate", "answer"]


def test_qwen_abstention_does_not_hide_lexical_sources(tmp_path):
    _, snapshot = _snapshot(tmp_path)
    calls: list = []
    service = ChatService(coordinator=FakeNemotron(calls),
                          investigator=FakeInvestigator(calls, status="no_evidence"))

    result = service.reply("What did I change?", snapshot)

    assert result["retrieval"]["worker_status"] == "no_evidence"
    assert result["retrieval"]["mode"] == "lexical_paragraph_baseline"
    assert result["evidence"]
    assert any("Qwen abstained" in item for item in result["limitations"])


def test_tampered_qwen_quote_never_reaches_answerer(tmp_path):
    _, snapshot = _snapshot(tmp_path)
    calls: list = []

    class Tampered(FakeInvestigator):
        def investigate(self, question: str, snapshot: Path) -> dict:
            result = super().investigate(question, snapshot)
            result["evidence"][0]["quote"] = "forged passage"
            return result

    service = ChatService(coordinator=FakeNemotron(calls),
                          investigator=Tampered(calls))
    result = service.reply("What did I change?", snapshot)

    assert result["retrieval"]["mode"] == "lexical_paragraph_baseline"
    assert result["retrieval"]["worker_status"] == "invalid_evidence"
    assert "forged passage" not in str(calls[-1][2])


def test_chat_api_source_and_approval_are_separate(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    calls: list = []
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=FakeNemotron(calls), investigator=FakeInvestigator(calls),
    ))
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"question": "What did I change?"})
        assert response.status_code == 200, response.text
        turn = response.json()
        assert turn["review_id"] and turn["session_id"]
        assert client.get("/api/status").json()["documents"] == 9
        evidence = turn["evidence"][0]
        source = client.get(
            f"/api/source/{turn['review_id']}/{evidence['doc_id']}"
        )
        assert source.status_code == 200
        assert evidence["quote"] in source.json()["text"]
        cited_doc_ids = {item["doc_id"] for item in turn["evidence"]}
        related = next(item for item in turn["related_notes"]
                       if item["doc_id"] not in cited_doc_ids)
        linked_source = client.get(
            f"/api/source/{turn['review_id']}/{related['doc_id']}"
        )
        assert linked_source.status_code == 200
        assert linked_source.json()["source_path"] == related["source_path"]
        assert client.post("/api/approve", json={
            "review_id": turn["review_id"], "title": "New lesson",
            "text": "A human-reviewed lesson.", "kind": "lesson",
            "evidence_ids": [evidence["evidence_id"]], "confirm": False,
        }).status_code == 400
        assert client.get("/api/status").json()["documents"] == 9
        next_turn = client.post("/api/chat", json={
            "session_id": turn["session_id"], "question": "And later?",
        })
        assert next_turn.status_code == 200
        assert calls[-3][0] == "plan"
        assert calls[-3][2] == [
            {"role": "user", "content": "What did I change?"},
            {"role": "assistant", "content": turn["answer"]},
        ]
        approved = client.post("/api/approve", json={
            "review_id": turn["review_id"], "title": "New lesson",
            "text": "A human-reviewed lesson.", "kind": "lesson",
            "evidence_ids": [evidence["evidence_id"]], "confirm": True,
        })
        assert approved.status_code == 200, approved.text
        assert client.get("/api/status").json()["documents"] == 10


def test_chat_discards_old_history_after_vault_capture(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    calls: list = []
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=FakeNemotron(calls), investigator=FakeInvestigator(calls),
    ))
    with TestClient(app) as client:
        first = client.post("/api/chat", json={"question": "What changed?"}).json()
        assert client.post("/api/capture", json={
            "title": "New attempt", "kind": "attempt", "text": "I tried a fresh plan.",
        }).status_code == 200
        second = client.post("/api/chat", json={
            "session_id": first["session_id"], "question": "What now?",
        }).json()
    assert second["history_reset"] is True
    assert calls[-3] == ("plan", "What now?", [])


def test_slow_chat_does_not_block_status_or_capture(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    entered, release = Event(), Event()
    calls: list = []

    class SlowNemotron(FakeNemotron):
        def plan(self, question: str, history: list[dict] | None = None) -> dict:
            entered.set()
            assert release.wait(timeout=5)
            return super().plan(question, history)

    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=SlowNemotron(calls, investigate=False),
    ))
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=3) as pool:
        chat_future = pool.submit(client.post, "/api/chat", json={"question": "Hello"})
        assert entered.wait(timeout=2)
        status_future = pool.submit(client.get, "/api/status")
        capture_future = pool.submit(client.post, "/api/capture", json={
            "title": "New note", "kind": "attempt", "text": "A new observation.",
        })
        try:
            assert status_future.result(timeout=2).status_code == 200
            assert capture_future.result(timeout=2).status_code == 200
        finally:
            release.set()
        result = chat_future.result(timeout=5).json()
        assert result["snapshot_stale"] is True
        assert result["history_reset"] is True


def test_qwen_cli_factory_uses_evidence_required_system_prompt():
    worker = make_qwen_investigator(
        "http://127.0.0.1:8000/v1", "qwen-v5", "exact-checkpoint-v5",
    )
    policy = worker.policy_factory()
    assert "EVIDENCE:" in policy.system_prompt
    assert "Obsidian vault snapshot" in policy.system_prompt
    assert worker.model_identity["checkpoint"] == "exact-checkpoint-v5"

    requests = []
    policy._transport = lambda _url, payload, _headers, _timeout: (
        requests.append(payload)
        or {"choices": [{"message": {"content": "print(search('focus'))"}}]}
    )
    assert policy.act("Question: What changed?") == "print(search('focus'))"
    assert requests[0]["model"] == "qwen-v5"
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert requests[0]["temperature"] == 0.0
    assert requests[0]["top_p"] == 1.0
    assert requests[0]["seed"] == 42
    assert requests[0]["max_tokens"] == worker.model_identity["max_tokens"]
    assert worker.model_identity["thinking"] is False


def test_qwen_factory_paper_mode_sends_same_prompt_to_policy_and_investigator():
    worker = make_qwen_investigator(
        "http://127.0.0.1:8000/v1", "qwen-v5", "exact-checkpoint-v5",
        source_domain="public_papers",
    )
    policy = worker.policy_factory()
    requests = []
    policy._transport = lambda _url, payload, _headers, _timeout: (
        requests.append(payload)
        or {"choices": [{"message": {"content": "print(search('retrieval'))"}}]}
    )

    assert worker.source_domain == "public_papers"
    assert worker.system_prompt == PUBLIC_PAPER_SYSTEM_PROMPT
    assert policy.act("Question: What did the paper report?") == "print(search('retrieval'))"
    assert requests[0]["messages"][0] == {
        "role": "system", "content": PUBLIC_PAPER_SYSTEM_PROMPT,
    }
    assert "personal Obsidian vault" not in requests[0]["messages"][0]["content"]


def test_model_key_can_come_from_ignored_env_file_without_overriding_process(tmp_path,
                                                                              monkeypatch):
    secret_file = tmp_path / ".env"
    secret_file.write_text("NEBIUS_API_KEY=file-secret\n")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert _configured_key("NEBIUS_API_KEY", secret_file) == "file-secret"
    monkeypatch.setenv("NEBIUS_API_KEY", "process-secret")
    assert _configured_key("NEBIUS_API_KEY", secret_file) == "process-secret"
