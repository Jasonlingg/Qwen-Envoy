"""A new source can prompt a review without silently rewriting an old decision."""

import pytest
from fastapi.testclient import TestClient

from scripts.personal_memory_web import create_app
from src.product.chat import ChatService
from src.product.graph import build_graph
from src.product.memory import freeze_vault

PRIOR_RECORD_ID = "retrieval-decision-2026-09-01"


def _vault_with_prior_decision(tmp_path):
    vault = tmp_path / "vault"
    prior = vault / "Learning" / "Research" / "Retrieval decision.md"
    prior.parent.mkdir(parents=True)
    prior.write_text(
        "---\n"
        f"record_id: {PRIOR_RECORD_ID}\n"
        "kind: decision\n"
        "effective_date: 2026-09-01\n"
        "status: current\n"
        "---\n\n"
        "# Retrieval decision\n\n"
        "For the research-question prototype, I decided to route every question "
        "through the iterative Qwen worker. I expect the worker to yield more "
        "source-supported answers than the lexical search baseline. This is a "
        "prospective expectation; it has not been measured on paired questions.\n",
        encoding="utf-8",
    )
    return vault, prior


def _capture_result(client, *, link_prior=True):
    prior_reference = ("[[Retrieval decision]]" if link_prior
                       else "the earlier retrieval decision")
    response = client.post("/api/capture", json={
        "title": "Paired retrieval run",
        "kind": "result",
        "effective_date": "2026-09-27",
        "text": (
            "Paired retrieval evaluation result, September 27. On the same 12 "
            "held-out research questions and frozen source snapshot, lexical "
            "search yielded 10 source-supported answers and the iterative "
            "Qwen worker yielded 7. This is an observed local result, not a "
            f"general verdict. It challenges {prior_reference} and the earlier "
            "expectation that the iterative worker would beat lexical search. "
            "Latency and cost were not recorded."
        ),
    })
    assert response.status_code == 200, response.text
    return response.json()["saved"]


def _impact(client, source_path):
    response = client.post("/api/impact", json={"source_path": source_path})
    assert response.status_code == 200, response.text
    return response.json()


class FakeDraftCoordinator:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def answer(self, question, evidence_packet, history=None):
        self.calls.append((question, evidence_packet, history))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_measured_result_prompts_review_with_both_exact_sources(tmp_path):
    vault, prior_file = _vault_with_prior_decision(tmp_path)
    prior_bytes = prior_file.read_bytes()
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        source_path = _capture_result(client)
        review = _impact(client, source_path)

        assert review["status"] == "possible_impact"
        assert review["review_id"]
        assert review["source"]["source_path"] == source_path
        assert review["prior"]["record_id"] == PRIOR_RECORD_ID
        assert review["reason"]
        assert review["proposal"]["kind"] == "correction"
        assert review["proposal"]["supersedes"] == review["prior"]["doc_id"]
        evidence_by_id = {item["evidence_id"]: item for item in review["evidence"]}
        assert evidence_by_id
        assert set(review["proposal"]["evidence_ids"]) <= set(evidence_by_id)
        assert {review["source"]["doc_id"], review["prior"]["doc_id"]} <= {
            evidence_by_id[item_id]["doc_id"]
            for item_id in review["proposal"]["evidence_ids"]
        }
        assert [(entry["role"], entry["doc_id"]) for entry in review["timeline"]] == [
            ("prior", review["prior"]["doc_id"]),
            ("source", review["source"]["doc_id"]),
        ]

        for item in review["evidence"]:
            opened = client.get(
                f"/api/source/{review['review_id']}/{item['doc_id']}"
            )
            assert opened.status_code == 200, opened.text
            assert opened.json()["text"][item["start"]:item["end"]] == item["quote"]

        # Investigation alone has no write authority.
        assert client.get("/api/status").json()["documents"] == 2
        assert prior_file.read_bytes() == prior_bytes


def test_plan_is_not_presented_as_a_completed_result(tmp_path):
    vault, _ = _vault_with_prior_decision(tmp_path)
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        response = client.post("/api/capture", json={
            "title": "Planned retrieval comparison",
            "kind": "source",
            "effective_date": "2026-09-27",
            "text": (
                "I plan to compare the iterative Qwen worker with lexical search "
                "on paired research questions next week. This would test "
                "[[Retrieval decision]]. No evaluation has been run and there "
                "are no observed answer-quality results yet."
            ),
        })
        assert response.status_code == 200, response.text
        review = _impact(client, response.json()["saved"])

        assert review["status"] in {"possible_impact", "no_candidate"}
        assert review["proposal"] is None
        assert client.get("/api/status").json()["documents"] == 2


def test_unrelated_result_has_no_prior_candidate_or_revision(tmp_path):
    vault, _ = _vault_with_prior_decision(tmp_path)
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        response = client.post("/api/capture", json={
            "title": "Bread temperature result",
            "kind": "result",
            "effective_date": "2026-09-27",
            "text": "A kitchen thermometer measured the bread center at 94 C after baking.",
        })
        assert response.status_code == 200, response.text
        review = _impact(client, response.json()["saved"])

        assert review["status"] == "no_candidate"
        assert review["prior"] is None
        assert review["proposal"] is None
        assert review["reason"]
        assert client.get("/api/status").json()["documents"] == 2


def test_result_without_wikilink_finds_the_same_topic_prior(tmp_path):
    vault, _ = _vault_with_prior_decision(tmp_path)
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        source_path = _capture_result(client, link_prior=False)
        review = _impact(client, source_path)

        assert review["status"] == "possible_impact"
        assert review["prior"]["record_id"] == PRIOR_RECORD_ID
        assert review["proposal"]["kind"] == "correction"
        assert {item["doc_id"] for item in review["evidence"]} == {
            review["source"]["doc_id"], review["prior"]["doc_id"],
        }


def test_approval_appends_version_and_stale_review_cannot_write(tmp_path):
    vault, prior_file = _vault_with_prior_decision(tmp_path)
    prior_bytes = prior_file.read_bytes()
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        source_path = _capture_result(client)
        review = _impact(client, source_path)
        proposal = review["proposal"]
        assert proposal is not None
        approval = {
            "review_id": review["review_id"],
            "title": "Revised retrieval decision",
            "text": (
                "I reviewed the paired local result: lexical search had 10 "
                "source-supported answers and the iterative Qwen worker had 7 "
                "on these 12 questions. I will use lexical search as the default "
                "until a held-out test justifies the worker's added cost."
            ),
            "kind": proposal["kind"],
            "supersedes": proposal["supersedes"],
            "evidence_ids": proposal["evidence_ids"],
            "confirm": False,
        }
        denied = client.post("/api/approve", json=approval)
        assert denied.status_code == 400
        assert client.get("/api/status").json()["documents"] == 2

        approval["confirm"] = True
        saved = client.post("/api/approve", json=approval)
        assert saved.status_code == 200, saved.text
        saved_file = vault / saved.json()["saved"]
        assert saved_file.is_file()
        assert prior_file.read_bytes() == prior_bytes
        assert f"supersedes_doc_id: {review['prior']['doc_id']}" in saved_file.read_text()
        assert client.get("/api/status").json()["documents"] == 3

        snapshot = tmp_path / "after-approval"
        freeze_vault(vault, snapshot)
        graph = build_graph(snapshot)
        by_path = {node["source_path"]: doc_id for doc_id, node in graph["nodes"].items()}
        assert prior_file.relative_to(vault).as_posix() in by_path
        assert saved.json()["saved"] in by_path
        assert {"source": by_path[saved.json()["saved"]],
                "target": by_path[prior_file.relative_to(vault).as_posix()],
                "type": "supersedes"} in graph["edges"]

        # A review made before any later capture is bound to its old snapshot.
        second_review = _impact(client, source_path)
        later = client.post("/api/capture", json={
            "title": "New observation", "kind": "attempt",
            "text": "I recorded another research session after reviewing the result.",
        })
        assert later.status_code == 200, later.text
        stale = client.post("/api/approve", json={
            **approval,
            "review_id": second_review["review_id"],
        })
        assert stale.status_code == 409
        assert client.get("/api/status").json()["documents"] == 4


def test_nemotron_draft_is_opt_in_even_when_coordinator_is_configured(tmp_path):
    vault, _ = _vault_with_prior_decision(tmp_path)
    coordinator = FakeDraftCoordinator({
        "answer": "The earlier decision was prospective [E2]. The paired run differed [E1].",
        "model": "nvidia/test-nemotron", "limitations": ["One local run."],
    })
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=coordinator,
    ))
    with TestClient(app) as client:
        source_path = _capture_result(client)
        review = _impact(client, source_path)

        assert review["status"] == "possible_impact"
        assert review["proposal"]["origin"] == "local_review_scaffold"
        assert review["model_requests_attempted"] == 0
        assert coordinator.calls == []
        assert client.get("/api/status").json()["model_calls"] == 0
        assert client.get("/api/status").json()["documents"] == 2


def test_requested_nemotron_draft_uses_both_sources_and_needs_approval(tmp_path):
    vault, _ = _vault_with_prior_decision(tmp_path)
    draft = (
        "The earlier decision expected the iterative worker to win [E2]. "
        "The paired local run found 7 supported worker answers versus 10 "
        "from lexical search [E1]. This suggests revisiting the decision, "
        "but it is one local comparison [E1][E2]."
    )
    coordinator = FakeDraftCoordinator({
        "answer": draft, "model": "nvidia/test-nemotron",
        "limitations": ["Latency and cost remain unmeasured."],
    })
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=coordinator,
    ))
    with TestClient(app) as client:
        source_path = _capture_result(client)
        response = client.post("/api/impact", json={
            "source_path": source_path, "draft_with_model": True,
        })
        assert response.status_code == 200, response.text
        review = response.json()

        assert len(coordinator.calls) == 1
        question, packet, history = coordinator.calls[0]
        assert "Draft a possible correction" in question
        assert history == []
        assert packet["corpus_hash"] == review["corpus_hash"]
        assert [item["evidence_id"] for item in packet["evidence"]] == ["E1", "E2"]
        assert {item["doc_id"] for item in packet["evidence"]} == {
            review["source"]["doc_id"], review["prior"]["doc_id"],
        }
        assert packet["evidence"] == review["evidence"]
        assert review["proposal"]["text"] == draft
        assert review["proposal"]["origin"] == "nemotron_draft_for_user_review"
        assert review["proposal"]["model"] == "nvidia/test-nemotron"
        assert review["model_requests_attempted"] == 1
        assert client.get("/api/status").json()["model_calls"] == 1
        assert client.get("/api/status").json()["documents"] == 2

        denied = client.post("/api/approve", json={
            "review_id": review["review_id"],
            "title": "Revised retrieval decision",
            "text": draft,
            "kind": "correction",
            "supersedes": review["prior"]["doc_id"],
            "evidence_ids": ["E1", "E2"],
            "confirm": False,
        })
        assert denied.status_code == 400
        assert client.get("/api/status").json()["documents"] == 2

        missing_prior = client.post("/api/approve", json={
            "review_id": review["review_id"],
            "title": "Revised retrieval decision",
            "text": draft,
            "kind": "correction",
            "supersedes": review["prior"]["doc_id"],
            "evidence_ids": ["E1"],
            "confirm": True,
        })
        assert missing_prior.status_code == 400
        assert "both notes" in missing_prior.json()["detail"]
        assert client.get("/api/status").json()["documents"] == 2


@pytest.mark.parametrize("model_output", [
    pytest.param({
        "answer": "Only the new result is cited [E1].",
        "model": "nvidia/test-nemotron", "limitations": [],
    }, id="missing-prior-citation"),
    pytest.param({
        "answer": None, "model": "nvidia/test-nemotron", "limitations": [],
    }, id="malformed-answer"),
    pytest.param(RuntimeError("provider unavailable"), id="provider-failure"),
])
def test_bad_or_failed_nemotron_draft_keeps_local_review(tmp_path, model_output):
    vault, _ = _vault_with_prior_decision(tmp_path)
    coordinator = FakeDraftCoordinator(model_output)
    app = create_app(vault, tmp_path / "state", chat_service=ChatService(
        coordinator=coordinator,
    ))
    with TestClient(app) as client:
        source_path = _capture_result(client)
        response = client.post("/api/impact", json={
            "source_path": source_path, "draft_with_model": True,
        })
        assert response.status_code == 200, response.text
        review = response.json()

        assert review["status"] == "possible_impact"
        assert review["proposal"]["origin"] == "local_review_scaffold"
        assert review["proposal"]["requires_edit"] is True
        assert "Nemotron draft unavailable" in review["draft_error"]
        assert review["model_requests_attempted"] == 1
        assert len(coordinator.calls) == 1
        assert client.get("/api/status").json()["model_calls"] == 1
        assert client.get("/api/status").json()["documents"] == 2
