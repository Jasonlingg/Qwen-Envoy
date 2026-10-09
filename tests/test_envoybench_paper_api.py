"""Paper runs preserve real source provenance and never invent a quality score."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from benchmarks.envoybench import paper_api
from src.env.tools import TOOL_PREAMBLE
from src.product.qwen_investigator import (
    NEMOTRON_PAPER_SYSTEM_PROMPT,
    PUBLIC_PAPER_SYSTEM_PROMPT,
    QwenInvestigator,
)
from src.research.web_sources import FetchedPage


def _client(tmp_path, monkeypatch, *, configured=False):
    model_path = None
    if configured:
        model_path = tmp_path / "models.json"
        model_path.write_text(
            json.dumps(
                {
                    "schema_version": "envoybench-models-v1",
                    "models": [
                        {
                            "key": "qwen_v5",
                            "model_id": "test-served-adapter",
                            "revision": "test-revision",
                            "endpoint": "http://127.0.0.1:9999/v1",
                            "api_key_env": "PAPER_TEST_API_KEY",
                            "decoding": {"max_tokens": 256, "temperature": 0, "top_p": 1},
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                        }
                    ],
                }
            )
        )
        monkeypatch.setenv("PAPER_TEST_API_KEY", "test-secret-never-export")
    router = paper_api.create_paper_router(tmp_path / "state", model_path)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), router.paper_service


def _upload(client, text="A 🧠 study. The answer is blue."):
    response = client.post("/api/papers?filename=study.txt", content=text.encode())
    assert response.status_code == 201, response.text
    return response.json()


def _finished(client, identifier):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        response = client.get(f"/api/paper-runs/{identifier}")
        assert response.status_code == 200, response.text
        if response.json()["status"] not in {"queued", "running"}:
            return response.json()
        time.sleep(0.01)
    pytest.fail("Test worker did not finish")


class _Policy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = 0

    def reset(self):
        self.requests = 0

    def act(self, observation):
        self.requests += 1
        return next(self.actions)

    def eval_metadata(self):
        return {
            "token_usage": {
                "request_count": self.requests,
                "usage_complete": False,
                "prompt_tokens": None,
                "completion_tokens": None,
            }
        }


class _TestREPL:
    """Test-only seam; production still requires Docker and never uses exec locally."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.killed = False

    def start_session(self):
        self.namespace = {"__builtins__": __builtins__}
        # Tests execute only hardcoded actions over the production tool preamble.
        import os

        before = os.environ.get("CORPUS_DIR")
        os.environ["CORPUS_DIR"] = str(self.corpus)
        try:
            exec(TOOL_PREAMBLE, self.namespace)
        finally:
            if before is None:
                os.environ.pop("CORPUS_DIR", None)
            else:
                os.environ["CORPUS_DIR"] = before

    def execute(self, code, timeout):
        from contextlib import redirect_stdout

        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, self.namespace)
        return output.getvalue()

    def kill_session(self):
        self.killed = True


def _wire_scripted_run(monkeypatch, service, actions):
    monkeypatch.setattr(paper_api, "preflight_sandbox", lambda: {"image_id": "test-only-image"})
    monkeypatch.setattr(service, "_make_policy", lambda model: _Policy(actions))
    monkeypatch.setattr(
        paper_api,
        "QwenInvestigator",
        lambda **kwargs: QwenInvestigator(
            **kwargs,
            repl_factory=_TestREPL,
        ),
    )


def test_unconfigured_reader_does_not_call_model_and_detects_tampering(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "_make_policy", lambda model: pytest.fail("must not run"))
    paper = _upload(client)
    assert not client.get("/api/papers/config").json()["ready"]
    assert client.get("/api/papers").json()["papers"][0]["id"] == paper["id"]
    assert paper["exposure"] == {"upstream": "unknown", "fine_tuning": "not_audited"}
    assert (
        client.post(
            "/api/paper-runs",
            json={
                "paper_id": paper["id"],
                "question": "What is the answer?",
                "model_key": "qwen_v5",
            },
        ).status_code
        == 503
    )
    raw = next((service.papers / paper["id"] / "raw").iterdir())
    raw.write_text("Changed source")
    assert client.get(f"/api/papers/{paper['id']}").status_code == 409
    assert not list(service.runs.glob("*.json"))


def test_upload_rejects_traversal_oversize_and_cross_origin(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    assert (
        client.post("/api/papers", params={"filename": "../escape.txt"}, content=b"x").status_code
        == 422
    )
    assert (
        client.post(
            "/api/papers?filename=a.txt", content=b"x" * (paper_api.MAX_FILE_BYTES + 1)
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/papers?filename=a.txt", content=b"x", headers={"Origin": "https://other.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/papers?filename=a.txt", content=b"x", headers={"Origin": "http://testserver"}
        ).status_code
        == 201
    )
    assert client.get("/api/papers/not-an-id").status_code == 404


def test_pdf_upload_retains_page_offsets(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): writer._add_object(font),
                }
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 260 Td (A paper says blue.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    raw = io.BytesIO()
    writer.write(raw)
    client, _ = _client(tmp_path, monkeypatch)
    result = client.post("/api/papers?filename=study.pdf", content=raw.getvalue())
    assert result.status_code == 201
    paper = result.json()
    assert "blue" in paper["text"]
    assert paper["sections"][0]["page"] == 1
    assert paper["metadata"]["coverage"] == "pdf_text_no_ocr"
    original = client.get(f"/api/papers/{paper['id']}/raw")
    assert original.status_code == 200
    assert original.headers["content-type"] == "application/pdf"
    assert original.content == raw.getvalue()

    fetched_urls = []

    def fake_fetch(url):
        fetched_urls.append(url)
        return FetchedPage(raw.getvalue(), url, content_type="application/pdf")

    monkeypatch.setattr(paper_api, "fetch_public_pdf", fake_fetch)
    linked = client.post(
        "/api/papers/from-url",
        json={
            "url": "https://arxiv.org/abs/2105.03011v1",
        },
    )
    assert linked.status_code == 201, linked.text
    assert fetched_urls == ["https://arxiv.org/pdf/2105.03011v1"]
    web = linked.json()
    assert web["text"] == paper["text"]
    assert web["metadata"]["source_kind"] == "fetched_pdf"
    assert web["metadata"]["requested_url"] == "https://arxiv.org/abs/2105.03011v1"
    assert web["source_url"] == "https://arxiv.org/pdf/2105.03011v1"
    assert web["raw_sha256"] == paper["raw_sha256"]
    assert client.get(f"/api/papers/{web['id']}").json() == web
    assert client.get(f"/api/papers/{web['id']}/raw").content == raw.getvalue()
    again = client.post(
        "/api/papers/from-url",
        json={
            "url": "https://arxiv.org/abs/2105.03011v1",
        },
    )
    assert again.status_code == 201
    assert again.json()["id"] == web["id"]


def test_web_pdf_input_rejects_private_urls_bad_content_and_bad_origins(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    assert paper_api._pdf_url("https://arxiv.org/pdf/2609.34428v1.pdf")[1] == (
        "https://arxiv.org/pdf/2609.34428v1"
    )
    for url in ("http://example.org/paper.pdf", "file:///etc/passwd"):
        assert client.post("/api/papers/from-url", json={"url": url}).status_code == 422
    assert (
        client.post("/api/papers/from-url", json={"url": "https://127.0.0.1/x.pdf"}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/papers/from-url",
            json={
                "url": "https://arxiv.org/abs/2105.03011",
            },
            headers={"Origin": "https://other.example"},
        ).status_code
        == 403
    )
    monkeypatch.setattr(
        paper_api,
        "fetch_public_pdf",
        lambda url: FetchedPage(
            b"<html>Not a PDF</html>",
            url,
            content_type="text/html",
        ),
    )
    response = client.post(
        "/api/papers/from-url",
        json={
            "url": "https://example.org/download",
        },
    )
    assert response.status_code == 422
    assert not list(service.papers.glob("paper_*"))


def test_full_investigator_flow_saves_checked_evidence_and_no_quality_score(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    paper = _upload(client)
    doc_id = paper["doc_id"]
    start = paper["text"].index("blue")
    spans = json.dumps([{"doc_id": doc_id, "start": start, "end": start + 4}])
    _wire_scripted_run(
        monkeypatch,
        service,
        [
            'print(search("answer blue"))',
            f'print(read("{doc_id}"))',
            f'SUBMIT: Blue. CITATIONS: ["{doc_id}"] EVIDENCE: {spans}',
        ],
    )
    config = client.get("/api/papers/config").json()
    assert config["ready"]
    assert "test-secret" not in json.dumps(config)
    response = client.post(
        "/api/paper-runs",
        json={
            "paper_id": paper["id"],
            "question": "What is the answer?",
            "model_key": "qwen_v5",
        },
    )
    assert response.status_code == 202
    run = _finished(client, response.json()["id"])
    assert run["status"] == "completed", run
    assert run["answer"] == "Blue."
    assert run["evidence"][0]["quote"] == "blue"
    assert run["evidence"][0]["start"] == start
    assert len(run["trajectory"]) == 3
    assert run["review_status"] == "unreviewed"
    assert "score" not in run
    assert run["duration_seconds"] > 0
    assert run["usage"]["prompt_tokens"] is None
    export = client.get(f"/api/paper-runs/{run['id']}/export")
    assert export.json() == run
    assert "attachment" in export.headers["content-disposition"]
    assert "test-secret" not in export.text and "127.0.0.1:9999" not in export.text
    assert client.get("/api/paper-runs").json()["runs"][0]["id"] == run["id"]
    raw = next((service.papers / paper["id"] / "raw").iterdir())
    raw.write_text("edited")
    assert client.get(f"/api/paper-runs/{run['id']}/export").status_code == 409


def test_live_paper_prompt_is_selected_by_model_id_without_changing_qwen(tmp_path, monkeypatch):
    _, service = _client(tmp_path, monkeypatch, configured=True)
    qwen = service.models["qwen_v5"]
    variant, prompt = paper_api._paper_protocol(qwen)
    assert variant == "default"
    assert prompt == PUBLIC_PAPER_SYSTEM_PROMPT
    assert service._make_policy(qwen).system_prompt == PUBLIC_PAPER_SYSTEM_PROMPT

    nemotron = {**qwen, "safe": {**qwen["safe"], "model_id": "nvidia/Nemotron-3_5-Lightning"}}
    variant, prompt = paper_api._paper_protocol(nemotron)
    assert variant == "nemotron"
    assert prompt == NEMOTRON_PAPER_SYSTEM_PROMPT
    assert service._make_policy(nemotron).system_prompt == NEMOTRON_PAPER_SYSTEM_PROMPT


def test_progress_trace_keeps_optional_provider_telemetry_separate_from_action():
    class TelemetryPolicy(_Policy):
        def act(self, observation):
            self.last_reasoning = "provider-reported reasoning"
            self.last_logprob_diagnostics = {"scope": "sampled tokens, not confidence"}
            return super().act(observation)

    record = {"trajectory": []}
    progress = paper_api._ProgressPolicy(
        TelemetryPolicy(['print(search("paper"))']), record, lambda _: None, time.monotonic()
    )
    assert progress.act("Question: paper?") == 'print(search("paper"))'
    step = progress.trace[0]
    assert step["action"] == 'print(search("paper"))'
    assert step["provider_reasoning"] == "provider-reported reasoning"
    assert step["sampled_logprobs"]["scope"] == "sampled tokens, not confidence"
    assert step["observation"] is None


def test_reasoning_only_provider_response_fails_closed_with_safe_metadata():
    policy = paper_api.OpenAICompatiblePolicy(
        endpoint="https://provider.invalid/v1",
        model="nemotron-test",
        transport=lambda *_: {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": None, "reasoning": "secret internal text"},
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        },
    )
    record = {"trajectory": []}
    progress = paper_api._ProgressPolicy(policy, record, lambda _: None, time.monotonic())
    with pytest.raises(ValueError, match="did not contain text content"):
        progress.act("Question")
    assert len(progress.trace) == 1
    failed = progress.trace[0]
    assert failed["step"] == 1 and failed["action"] is None
    assert failed["model_error_category"] == "missing_visible_action"
    assert failed["response_metadata"] == {
        "finish_reason": "length",
        "visible_content_present": False,
        "reasoning_field_present": True,
    }
    assert "secret internal text" not in json.dumps(record)
    assert record["usage"]["failed_request_count"] == 1


def test_ambiguous_provider_text_is_saved_unexecuted_and_redacted(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    paper = _upload(client)
    rejected = "print('unsafe')<think>test-secret-never-export</think>"
    monkeypatch.setattr(paper_api, "preflight_sandbox", lambda: {"image_id": "test-only-image"})
    monkeypatch.setattr(service, "_make_policy", lambda model: paper_api.OpenAICompatiblePolicy(
        endpoint="https://provider.invalid/v1", model="nemotron-test",
        transport=lambda *_: {"choices": [{
            "finish_reason": "stop", "message": {"content": rejected},
        }]},
    ))

    class NoExecutionREPL(_TestREPL):
        def execute(self, code, timeout):
            pytest.fail("rejected model text must never reach the sandbox")

    monkeypatch.setattr(
        paper_api, "QwenInvestigator",
        lambda **kwargs: QwenInvestigator(**kwargs, repl_factory=NoExecutionREPL),
    )
    response = client.post("/api/paper-runs", json={
        "paper_id": paper["id"], "question": "What is blue?", "model_key": "qwen_v5",
    })
    run = _finished(client, response.json()["id"])
    assert run["status"] == "failed" and run["outcome"] == "error"
    failed = run["trajectory"][0]
    assert failed["action"] is None
    assert failed["model_error_category"] == "ambiguous_thinking"
    assert failed["rejected_model_content_unexecuted"] == rejected.replace(
        "test-secret-never-export", "[REDACTED]"
    )
    assert "test-secret-never-export" not in json.dumps(run)


def test_failed_response_content_capture_is_bounded():
    raw = "x" * 9_000 + "<think>bad</think>"
    policy = paper_api.OpenAICompatiblePolicy(
        endpoint="https://provider.invalid/v1", model="nemotron-test",
        transport=lambda *_: {"choices": [{"message": {"content": raw}}]},
    )
    progress = paper_api._ProgressPolicy(policy, {"trajectory": []}, lambda _: None,
                                         time.monotonic())
    with pytest.raises(ValueError, match="ambiguous thinking"):
        progress.act("Question")
    failed = progress.trace[0]
    assert len(failed["rejected_model_content_unexecuted"]) == 8_000
    assert failed["rejected_content_truncated"] is True


def test_scripted_nemotron_paper_run_uses_selected_protocol(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    service.models["qwen_v5"]["safe"]["model_id"] = "NVIDIA-Nemotron-3-Nano-30B-A3B"
    paper = _upload(client)
    doc_id = paper["doc_id"]
    start = paper["text"].index("blue")
    _wire_scripted_run(monkeypatch, service, [
        'print(search("answer blue"))',
        f'print(read("{doc_id}"))',
        f'SUBMIT: Blue. CITATIONS: ["{doc_id}"] '
        f'EVIDENCE: [{{"doc_id":"{doc_id}","start":{start},"end":{start + 4}}}]',
    ])
    response = client.post("/api/paper-runs", json={
        "paper_id": paper["id"], "question": "What is the answer?", "model_key": "qwen_v5",
    })
    assert response.status_code == 202
    run = _finished(client, response.json()["id"])
    assert run["status"] == "completed", run
    assert run["protocol_variant"] == "nemotron"
    assert run["answer"] == "Blue."
    assert run["evidence"][0]["quote"] == "blue"
    assert run["system_prompt_sha256"] == hashlib.sha256(
        NEMOTRON_PAPER_SYSTEM_PROMPT.encode()
    ).hexdigest()


def test_failed_model_turn_is_preserved_in_saved_trajectory(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    paper = _upload(client)
    doc_id = paper["doc_id"]

    class FailsOnThird(_Policy):
        def act(self, observation):
            if self.requests >= 2:
                self.last_response_metadata = {
                    "finish_reason": "length",
                    "visible_content_present": False,
                    "reasoning_field_present": True,
                }
                raise ValueError("chat-completions response did not contain text content")
            return super().act(observation)

    monkeypatch.setattr(paper_api, "preflight_sandbox", lambda: {"image_id": "test-only-image"})
    monkeypatch.setattr(service, "_make_policy", lambda model: FailsOnThird([
        'print(search("blue"))', f'print(read("{doc_id}"))',
    ]))
    monkeypatch.setattr(
        paper_api, "QwenInvestigator",
        lambda **kwargs: QwenInvestigator(**kwargs, repl_factory=_TestREPL),
    )
    response = client.post("/api/paper-runs", json={
        "paper_id": paper["id"], "question": "What is blue?", "model_key": "qwen_v5",
    })
    run = _finished(client, response.json()["id"])
    assert run["status"] == "failed"
    assert run["outcome"] == "error"
    assert len(run["trajectory"]) == 3
    assert run["trajectory"][-1]["model_error_category"] == "missing_visible_action"
    assert run["trajectory"][-1]["response_metadata"]["finish_reason"] == "length"


def test_duplicate_run_rejected_and_failed_sandbox_never_calls_model(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    paper = _upload(client)
    entered, release = threading.Event(), threading.Event()

    def unavailable():
        entered.set()
        release.wait(timeout=3)
        raise RuntimeError("http://127.0.0.1:9999/v1 test-secret-never-export")

    monkeypatch.setattr(paper_api, "preflight_sandbox", unavailable)
    monkeypatch.setattr(service, "_make_policy", lambda model: pytest.fail("must not call"))
    body = {"paper_id": paper["id"], "question": "Answer?", "model_key": "qwen_v5"}
    first = client.post("/api/paper-runs", json=body)
    assert entered.wait(timeout=2)
    try:
        assert client.post("/api/paper-runs", json=body).status_code == 409
    finally:
        release.set()
    run = _finished(client, first.json()["id"])
    assert run["status"] == "failed"
    assert run["answer"] is None and run["evidence"] == []
    assert "test-secret" not in json.dumps(run)
    assert "127.0.0.1:9999" not in json.dumps(run)


def test_incomplete_run_fails_and_restart_keeps_history(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch, configured=True)
    paper = _upload(client)
    _wire_scripted_run(monkeypatch, service, ['print(search("blue"))'] * paper_api.MAX_STEPS)
    first = client.post(
        "/api/paper-runs",
        json={
            "paper_id": paper["id"],
            "question": "Answer?",
            "model_key": "qwen_v5",
        },
    ).json()
    run = _finished(client, first["id"])
    assert run["status"] == "failed" and run["outcome"] == "incomplete"
    assert "Step limit" in run["error"]
    run.update(status="running")
    service._save(run)
    reopened = paper_api._PaperService(service.state_dir, None)
    recovered = reopened.run(run["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["answer"] is None


def test_reject_invalid_evidence_and_web_endpoint_override(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    assert (
        client.post(
            "/api/paper-runs",
            json={
                "paper_id": "x",
                "question": "x",
                "model_key": "x",
                "endpoint": "https://other.example",
            },
        ).status_code
        == 422
    )
    with pytest.raises(ValueError, match="invalid"):
        paper_api._checked_evidence(
            [{"doc_id": "d", "start": 0, "end": 4, "quote": "fake"}], {"d": {"text": "blue"}}
        )


def test_studio_routes_and_blind_session_isolation(tmp_path):
    from benchmarks.envoybench.demo import create_app

    client = TestClient(create_app(paper_state_dir=tmp_path / "normal"))
    assert client.get("/papers").status_code == 200
    assert client.get("/model").status_code == 200
    assert (
        client.get("/api/model-card").json()["training"]["Displayed checkpoint"] == "checkpoint-50"
    )
    blind = TestClient(
        create_app(blind_review_dir=Path("unused"), paper_state_dir=tmp_path / "blind")
    )
    for path in ("/papers", "/model", "/api/model-card", "/api/papers", "/api/paper-runs"):
        assert blind.get(path).status_code == 404
