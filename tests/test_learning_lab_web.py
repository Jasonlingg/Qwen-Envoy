"""Exercise a web source, dated revision, and Obsidian review boundary together."""

import io
from pathlib import Path
from shutil import copytree

import pytest
from fastapi.testclient import TestClient

from scripts.personal_memory_web import create_app
from src.product.chat import ChatService
from src.research.agent import load_snapshot
from src.research.file_sources import MAX_FILE_BYTES
from src.research.web_sources import FetchedPage, build_web_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_web_source_to_thread_and_reviewed_vault_recall(tmp_path, monkeypatch):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    state = tmp_path / "state"
    html = (
        b"<html><head><title>Agent retrieval study</title></head><body><article>"
        b"<h1>Agent retrieval study</h1>"
        b"<p>In our small library, BM25 found the required papers before the agent ran.</p>"
        b"<p>A larger corpus may change the result; this page does not establish that it will.</p>"
        b"</article></body></html>"
    )

    def offline_build(url, output):
        return build_web_snapshot(
            url, output,
            downloader=lambda _: FetchedPage(body=html, final_url=url),
        )

    monkeypatch.setattr("src.product.learning_lab_api.build_web_snapshot", offline_build)

    with TestClient(create_app(vault, state)) as client:
        before = client.get("/api/status").json()["documents"]
        created = client.post("/api/lab/threads", json={
            "question": "Does iterative search beat BM25 for my research library?",
            "current_view": "I expected iterative search to find better evidence.",
            "effective_date": "2026-09-15",
        })
        assert created.status_code == 200, created.text
        thread_id = created.json()["thread_id"]
        assert len(client.get("/api/lab/threads").json()["threads"]) == 1

        imported = client.post(f"/api/lab/threads/{thread_id}/sources", json={
            "url": "https://example.org/retrieval-study",
        })
        assert imported.status_code == 200, imported.text
        source = imported.json()
        assert source["promotion_status"] == "draft_in_inbox"
        assert source["staged_path"].startswith("_inbox/source-")
        assert (vault / source["staged_path"]).is_file()
        assert client.get("/api/status").json()["documents"] == before
        passage = next(item for item in source["passages"] if "BM25" in item["quote"])
        own_results = client.get(
            f"/api/lab/threads/{thread_id}/observations",
            params={"query": "one-pass retrieval iterative worker pilot"},
        )
        assert own_results.status_code == 200, own_results.text
        observation = next(item for item in own_results.json()["evidence"]
                           if "one-pass retrieval gave" in item["quote"])
        observation_hash = own_results.json()["corpus_hash"]

        proposal = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "Our small-corpus result favors BM25; a larger test remains open.",
            "reason": "The new source describes the same small-library limitation.",
            "effective_date": "2026-09-29",
            "evidence": [{"source_id": source["source_id"],
                          "start": passage["start"], "end": passage["end"]}],
            "observations": [{"doc_id": observation["doc_id"],
                              "start": observation["start"],
                              "end": observation["end"],
                              "corpus_hash": observation_hash}],
        })
        assert proposal.status_code == 200, proposal.text
        staged = proposal.json()["staged_path"]
        assert (vault / staged).is_file()
        assert "Exact spans establish provenance" in (vault / staged).read_text()
        assert "vault_observation" in (vault / staged).read_text()
        proposal_id = proposal.json()["proposal"]["proposal_id"]

        thread = client.get(f"/api/lab/threads/{thread_id}").json()
        assert thread["current_approved_view"]["view"] == (
            "I expected iterative search to find better evidence."
        )
        assert len(thread["pending_proposals"]) == 1
        assert thread["pending_proposals"][0]["evidence_receipts"][0]["quote"] == passage["quote"]
        assert (thread["pending_proposals"][0]["observation_receipts"][0]["quote"]
                == observation["quote"])
        draft_path = vault / staged
        original_draft = draft_path.read_bytes()
        draft_path.write_bytes(original_draft + b"\nExtra text after staging.\n")
        changed_draft = client.post(
            f"/api/lab/threads/{thread_id}/proposals/{proposal_id}/accept",
            json={"confirmed": True},
        )
        assert changed_draft.status_code == 400
        assert "changed" in changed_draft.json()["detail"]
        draft_path.write_bytes(original_draft)
        rejected = client.post(
            f"/api/lab/threads/{thread_id}/proposals/{proposal_id}/accept",
            json={"confirmed": False},
        )
        assert rejected.status_code == 400
        accepted = client.post(
            f"/api/lab/threads/{thread_id}/proposals/{proposal_id}/accept",
            json={"confirmed": True},
        )
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["vault_promotion_status"] == "pending_manual_promotion"
        assert len(accepted.json()["thread"]["timeline"]) == 2
        timeline = client.get(f"/api/lab/threads/{thread_id}").json()["timeline"]
        assert timeline[1]["observation_receipts"][0]["status"] == "exact_span_verified"

        # Moving a draft without changing its review marker must not make it
        # searchable as reviewed knowledge.
        library = vault / "library"
        library.mkdir()
        moved = library / "agent-retrieval-study.md"
        (vault / source["staged_path"]).rename(moved)
        unreviewed = client.post("/api/lab/refresh-vault")
        assert unreviewed.status_code == 200, unreviewed.text
        assert unreviewed.json()["documents"] == before

        # The user inspects the original and edits the review field in Obsidian.
        moved.write_text(
            moved.read_text().replace("review_status: agent_authored_draft",
                                      "review_status: user_reviewed", 1),
            encoding="utf-8",
        )
        refreshed = client.post("/api/lab/refresh-vault")
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["documents"] == before + 1
        recalled = client.post("/api/chat", json={
            "question": "BM25 required papers small library",
        })
        assert recalled.status_code == 200, recalled.text
        assert any("BM25 found the required papers" in item["quote"]
                   for item in recalled.json()["evidence"])
        stale = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "A second view", "reason": "Recheck the old passage.",
            "effective_date": "2026-09-30",
            "evidence": [{"source_id": source["source_id"],
                          "start": passage["start"], "end": passage["end"]}],
            "observations": [{"doc_id": observation["doc_id"],
                              "start": observation["start"], "end": observation["end"],
                              "corpus_hash": observation_hash}],
        })
        assert stale.status_code == 400
        assert "stale" in stale.json()["detail"]
        assert client.get(f"/api/lab/threads/{thread_id}").json()["pending_proposals"] == []


def test_learning_lab_rejects_unfrozen_evidence_span(tmp_path, monkeypatch):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    html = (b"<html><head><title>Source</title></head><body><article>"
            b"<p>One bounded source passage.</p></article></body></html>")

    def offline_build(url, output):
        return build_web_snapshot(
            url, output,
            downloader=lambda _: FetchedPage(body=html, final_url=url),
        )

    monkeypatch.setattr("src.product.learning_lab_api.build_web_snapshot", offline_build)
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        thread_id = client.post("/api/lab/threads", json={
            "question": "What did the source show?", "current_view": "I am unsure.",
            "effective_date": "2026-09-29",
        }).json()["thread_id"]
        source = client.post(f"/api/lab/threads/{thread_id}/sources", json={
            "url": "https://example.org/source",
        }).json()
        source_id = source["source_id"]
        response = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "It showed one passage.", "reason": "Review it.",
            "effective_date": "2026-09-29",
            "evidence": [{"source_id": source_id, "start": 0, "end": 99_999}],
        })
        assert response.status_code == 400
        assert client.get(f"/api/lab/threads/{thread_id}").json()["pending_proposals"] == []
        passage = source["passages"][0]
        monkeypatch.setattr(
            "src.product.learning_lab_api.stage_drafts",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("stage failed")),
        )
        failed_stage = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "A provisional view.", "reason": "Check the passage.",
            "effective_date": "2026-09-29",
            "evidence": [{"source_id": source_id,
                          "start": passage["start"], "end": passage["end"]}],
        })
        assert failed_stage.status_code == 400
        assert client.get(f"/api/lab/threads/{thread_id}").json()["pending_proposals"] == []


def test_optional_nemotron_suggestion_sees_checked_source_and_never_saves(tmp_path, monkeypatch):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    html = (b"<html><head><title>Paper note</title></head><body><article>"
            b"<p>The new source reports a limited retrieval result.</p>"
            b"</article></body></html>")

    def offline_build(url, output):
        return build_web_snapshot(
            url, output,
            downloader=lambda _: FetchedPage(body=html, final_url=url),
        )

    class FakeNemotron:
        def answer(self, question, packet, history):
            assert "saved question" in question
            assert history == []
            assert [item["evidence_id"] for item in packet["evidence"]] == ["E1", "E2"]
            assert packet["evidence"][0]["source_kind"] == "user_statement"
            assert "limited retrieval result" in packet["evidence"][1]["quote"]
            return {
                "answer": "My previous view [E1] needs a narrower test [E2].",
                "limitations": ["The source is not our own experiment."],
                "model": "nvidia/test-nemotron",
            }

    monkeypatch.setattr("src.product.learning_lab_api.build_web_snapshot", offline_build)
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=FakeNemotron(),
    ))
    with TestClient(app) as client:
        thread_id = client.post("/api/lab/threads", json={
            "question": "Does retrieval work?", "current_view": "I think it does.",
            "effective_date": "2026-09-20",
        }).json()["thread_id"]
        source = client.post(f"/api/lab/threads/{thread_id}/sources", json={
            "url": "https://example.org/paper-note",
        }).json()
        passage = source["passages"][0]
        response = client.post(f"/api/lab/threads/{thread_id}/suggest", json={
            "evidence": [{"source_id": source["source_id"],
                          "start": passage["start"], "end": passage["end"]}],
        })
        assert response.status_code == 200, response.text
        assert response.json()["semantic_support"] == "not_reviewed"
        assert "narrower test" in response.json()["suggested_view"]
        assert client.get("/api/status").json()["model_calls"] == 1
        assert client.get(f"/api/lab/threads/{thread_id}").json()["pending_proposals"] == []


def test_local_lab_rejects_foreign_host_and_cross_origin_write(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        assert client.get("/api/lab/threads", headers={"host": "evil.example"}).status_code == 400
        assert client.post(
            "/api/lab/threads/thread_" + "0" * 32 + "/investigate-vault", json={},
        ).status_code == 503
        body = {
            "question": "Which evidence changed my view?",
            "current_view": "I am unsure.",
            "effective_date": "2026-09-29",
        }
        rejected = client.post("/api/lab/threads", json=body,
                               headers={"origin": "https://evil.example"})
        assert rejected.status_code == 403
        assert client.get("/api/lab/threads").json()["threads"] == []
        accepted = client.post("/api/lab/threads", json=body,
                               headers={"origin": "http://testserver",
                                        "sec-fetch-site": "same-origin"})
        assert accepted.status_code == 200


def test_optional_qwen_investigation_returns_checked_note_spans_and_run_log(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    state = tmp_path / "state"

    class FakeQwen:
        def investigate(self, question, snapshot):
            assert "retrieval" in question
            manifest, docs = load_snapshot(snapshot)
            doc = next(doc for doc in docs.values()
                       if "one-pass retrieval gave" in doc["text"])
            quote = "one-pass retrieval gave more complete supported responses"
            start = doc["text"].index(quote)
            return {
                "status": "evidence_found",
                "corpus_hash": manifest["corpus_hash"],
                "evidence": [{"evidence_id": "E1", "doc_id": doc["doc_id"],
                              "start": start, "end": start + len(quote), "quote": quote}],
                "model_requests_attempted": 3,
                "model_identity": {"checkpoint": "qwen-test-checkpoint"},
                "trajectory": [{"step": 1, "action": "search('retrieval')"}],
                "warning": "Review source support.",
            }

    app = create_app(vault, state, chat_service=ChatService(investigator=FakeQwen()))
    with TestClient(app) as client:
        assert client.get("/api/status").json()["qwen_configured"] is True
        thread_id = client.post("/api/lab/threads", json={
            "question": "Does iterative retrieval help?",
            "current_view": "I thought it always would.",
            "effective_date": "2026-09-01",
        }).json()["thread_id"]
        response = client.post(f"/api/lab/threads/{thread_id}/investigate-vault", json={})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "evidence_found"
        assert result["evidence"][0]["quote"] == (
            "one-pass retrieval gave more complete supported responses"
        )
        assert result["evidence"][0]["corpus_hash"] == result["corpus_hash"]
        assert result["model_identity"]["checkpoint"] == "qwen-test-checkpoint"
        assert result["model_requests_attempted"] == 3
        assert client.get("/api/status").json()["model_calls"] == 3
        assert (state / result["trajectory_path"]).is_file()
        assert client.get(f"/api/lab/threads/{thread_id}").json()["pending_proposals"] == []


def test_qwen_investigation_rejects_invented_quote(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")

    class FakeQwen:
        def investigate(self, question, snapshot):
            manifest, docs = load_snapshot(snapshot)
            doc = next(iter(docs.values()))
            return {
                "status": "evidence_found", "corpus_hash": manifest["corpus_hash"],
                "evidence": [{"evidence_id": "E1", "doc_id": doc["doc_id"],
                              "start": 0, "end": 12, "quote": "not a real quote"}],
                "model_requests_attempted": 1,
            }

    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        investigator=FakeQwen(),
    ))
    with TestClient(app) as client:
        thread_id = client.post("/api/lab/threads", json={
            "question": "What is the evidence?", "current_view": "Unsure.",
            "effective_date": "2026-09-29",
        }).json()["thread_id"]
        response = client.post(f"/api/lab/threads/{thread_id}/investigate-vault", json={})
        assert response.status_code == 400
        assert "exact snapshot span" in response.json()["detail"]
        assert client.get("/api/status").json()["model_calls"] == 1


def test_attached_file_becomes_frozen_evidence_without_entering_reviewed_vault(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    state = tmp_path / "state"
    body = (
        b"Pilot result: one-pass retrieval found more supported answers than "
        b"the iterative worker on this small selected sample.\n"
    )
    with TestClient(create_app(vault, state)) as client:
        before = client.get("/api/status").json()["documents"]
        thread_id = client.post("/api/lab/threads", json={
            "question": "Did iterative retrieval help in the pilot?",
            "current_view": "I expected an iterative worker to help.",
            "effective_date": "2026-09-01",
        }).json()["thread_id"]
        uploaded = client.post(
            f"/api/lab/threads/{thread_id}/files",
            params={"filename": "pilot-results.txt"},
            content=body,
            headers={"content-type": "application/octet-stream"},
        )
        assert uploaded.status_code == 200, uploaded.text
        source = uploaded.json()
        assert source["source_kind"] == "uploaded_file"
        assert source["filename"] == "pilot-results.txt"
        assert source["source_url"] is None
        assert source["download_url"]
        assert (vault / source["staged_path"]).is_file()
        assert client.get("/api/status").json()["documents"] == before
        download = client.get(source["download_url"])
        assert download.status_code == 200
        assert download.content == body
        assert "attachment" in download.headers["content-disposition"]

        passage = source["passages"][0]
        proposal = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "The small pilot favors one-pass retrieval.",
            "reason": "The attached pilot result challenges my earlier expectation.",
            "effective_date": "2026-09-29",
            "evidence": [{"source_id": source["source_id"],
                          "start": passage["start"], "end": passage["end"],
                          "relation": "challenges"}],
        })
        assert proposal.status_code == 200, proposal.text
        receipt = proposal.json()["source_receipts"][0]
        assert receipt["reference"].startswith(f"file:{source['source_id']}:")
        assert receipt["filename"] == "pilot-results.txt"
        assert receipt["quote"] == passage["quote"]
        assert receipt["relation"] == "challenges"
        staged = (vault / proposal.json()["staged_path"]).read_text()
        assert "Attached file: pilot-results.txt" in staged
        assert "Your assessment: challenges" in staged
        thread = client.get(f"/api/lab/threads/{thread_id}").json()
        assert thread["pending_proposals"][0]["evidence_receipts"][0]["status"] == (
            "exact_span_verified"
        )
        assert thread["pending_proposals"][0]["evidence_receipts"][0]["relation"] == (
            "challenges"
        )
        proposal_id = proposal.json()["proposal"]["proposal_id"]
        accepted = client.post(
            f"/api/lab/threads/{thread_id}/proposals/{proposal_id}/accept",
            json={"confirmed": True},
        )
        assert accepted.status_code == 200, accepted.text
        revised = client.get(f"/api/lab/threads/{thread_id}").json()
        assert revised["timeline"][1]["evidence_receipts"][0]["relation"] == "challenges"
        raw_path = next((state / "learning_lab" / "sources" / thread_id
                         / source["source_id"] / "raw").glob("*.txt"))
        raw_path.write_bytes(b"A different pilot result.")
        changed = client.get(f"/api/lab/threads/{thread_id}")
        assert changed.status_code == 200, changed.text
        assert changed.json()["sources"][0]["promotion_status"] == (
            "source_unavailable_or_changed"
        )
        assert changed.json()["timeline"][1]["evidence_receipts"][0]["status"] == (
            "source_unavailable_or_changed"
        )


def test_file_upload_rejects_oversize_unsafe_name_and_broken_pdf(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        thread_id = client.post("/api/lab/threads", json={
            "question": "What did the report show?", "current_view": "Unknown.",
            "effective_date": "2026-09-29",
        }).json()["thread_id"]
        route = f"/api/lab/threads/{thread_id}/files"
        too_large = client.post(route, params={"filename": "report.txt"},
                                content=b"x" * (MAX_FILE_BYTES + 1))
        assert too_large.status_code == 413
        unsafe = client.post(route, params={"filename": "../report.txt"},
                             content=b"A valid text file.")
        assert unsafe.status_code == 400
        bad_pdf = client.post(route, params={"filename": "report.pdf"},
                              content=b"not a PDF")
        assert bad_pdf.status_code == 400
        assert client.get(f"/api/lab/threads/{thread_id}").json()["sources"] == []


def test_empty_vault_can_start_with_an_attachment_and_promote_it(tmp_path):
    vault = tmp_path / "empty-vault"
    vault.mkdir()
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        assert client.get("/api/status").json()["documents"] == 0
        assert client.get("/lab").status_code == 200
        assert client.post("/api/chat", json={"question": "What do I know?"}).status_code == 409
        thread_id = client.post("/api/lab/threads", json={
            "question": "Does a cheap baseline work?",
            "current_view": "I have not tested it yet.",
            "effective_date": "2026-09-29",
        }).json()["thread_id"]
        imported = client.post(
            f"/api/lab/threads/{thread_id}/files",
            params={"filename": "first-result.txt"},
            content=b"One-pass retrieval found the needed evidence in the first pilot.",
        )
        assert imported.status_code == 200, imported.text
        assert client.get("/api/status").json()["documents"] == 0
        staged = vault / imported.json()["staged_path"]
        library = vault / "library"
        library.mkdir()
        reviewed = library / "first-result.md"
        reviewed.write_text(
            staged.read_text().replace("review_status: agent_authored_draft",
                                       "review_status: user_reviewed", 1),
            encoding="utf-8",
        )
        refreshed = client.post("/api/lab/refresh-vault")
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["documents"] == 1
        answer = client.post("/api/chat", json={"question": "one-pass retrieval pilot"})
        assert answer.status_code == 200, answer.text


def test_pdf_attachment_receipts_point_to_original_page(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (One-pass search won the pilot.) Tj ET")
    page[NameObject("/Contents")] = stream
    raw = io.BytesIO()
    writer.write(raw)

    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        thread_id = client.post("/api/lab/threads", json={
            "question": "Did one-pass search win the pilot?", "current_view": "I doubt it.",
            "effective_date": "2026-09-29",
        }).json()["thread_id"]
        source_response = client.post(
            f"/api/lab/threads/{thread_id}/files", params={"filename": "pilot.pdf"},
            content=raw.getvalue(),
        )
        assert source_response.status_code == 200, source_response.text
        source = source_response.json()
        passage = source["passages"][0]
        assert passage["pages"] == [1]
        proposal = client.post(f"/api/lab/threads/{thread_id}/proposals", json={
            "revised_view": "One-pass search won this pilot.",
            "reason": "The attached pilot reports that outcome.",
            "effective_date": "2026-09-29",
            "evidence": [{"source_id": source["source_id"],
                          "start": passage["start"], "end": passage["end"],
                          "relation": "challenges"}],
        })
        assert proposal.status_code == 200, proposal.text
        assert proposal.json()["source_receipts"][0]["pages"] == [1]
        assert "PDF pages: 1" in (vault / proposal.json()["staged_path"]).read_text()
