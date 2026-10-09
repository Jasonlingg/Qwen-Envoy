"""Real-report-derived memory flow; approval here is a test-only simulation.

This exercises the product workflow over an isolated copy of the public QASPER
report. It neither approves a memory in the user's vault nor runs Qwen.
"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp")
from mcp import Client  # noqa: E402

from scripts.build_qwen_experiment_vault import DEFAULT_REPORT, build_vault  # noqa: E402
from scripts.personal_memory_mcp import create_server  # noqa: E402
from scripts.personal_memory_web import create_app  # noqa: E402
from src.product.memory import freeze_vault  # noqa: E402
from src.research.agent import load_snapshot  # noqa: E402


def test_real_qwen_report_flows_through_impact_chat_approval_and_mcp(tmp_path: Path):
    report = json.loads(DEFAULT_REPORT.read_text(encoding="utf-8"))
    vault = tmp_path / "vault"
    notes = build_vault(vault, report_path=DEFAULT_REPORT)
    original_notes = {role: path.read_bytes() for role, path in notes.items()}
    reported_passes = {
        arm: report["results"][arm]["semantic"]["pass"]
        for arm in ("starting_sft", "epoch_1", "epoch_2")
    }

    with TestClient(create_app(vault, tmp_path / "web-state")) as web:
        status = web.get("/api/status").json()
        assert status["documents"] == 3
        assert status["recent_results"] == [{
            "source_path": notes["result"].name,
            "title": "Qwen QASPER continuation measured result",
            "effective_date": report["date"],
            "captured_at": None,
            "review_status": "source_derived_unreviewed",
        }]
        assert status["nemotron_configured"] is False
        assert status["qwen_configured"] is False
        assert status["model_requests_attempted"] == 0

        impact_response = web.post(
            "/api/impact", json={"source_path": notes["result"].name}
        )
        assert impact_response.status_code == 200, impact_response.text
        impact = impact_response.json()
        assert impact["status"] == "possible_impact"
        assert impact["snapshot_stale"] is False
        assert impact["source"]["source_path"] == notes["result"].name
        assert impact["prior"]["source_path"] == notes["prior"].name
        assert impact["prior"]["record"]["status"] == "retrospective_reconstruction"
        assert impact["proposal"]["kind"] == "correction"
        assert impact["proposal"]["requires_edit"] is True
        assert impact["proposal"]["supersedes"] == impact["prior"]["doc_id"]
        assert [(entry["role"], entry["doc_id"]) for entry in impact["timeline"]] == [
            ("prior", impact["prior"]["doc_id"]),
            ("source", impact["source"]["doc_id"]),
        ]
        evidence_by_id = {item["evidence_id"]: item for item in impact["evidence"]}
        assert set(impact["proposal"]["evidence_ids"]) == set(evidence_by_id)
        assert {item["source_path"] for item in evidence_by_id.values()} == {
            notes["prior"].name, notes["result"].name,
        }
        for item in impact["evidence"]:
            opened = web.get(
                f"/api/source/{impact['review_id']}/{item['doc_id']}"
            )
            assert opened.status_code == 200, opened.text
            text = opened.json()["text"]
            assert text[item["start"]:item["end"]] == item["quote"]
            assert text.encode("utf-8") == (vault / item["source_path"]).read_bytes()

        first_chat_response = web.post("/api/chat", json={
            "question": (
                "What changed from the Qwen continuation strategy "
                "after the QASPER measured result?"
            ),
        })
        assert first_chat_response.status_code == 200, first_chat_response.text
        first_chat = first_chat_response.json()
        assert first_chat["answer"] is None
        assert first_chat["answer_status"] == "evidence_only"
        assert first_chat["retrieval"]["mode"] == "lexical_paragraph_baseline"
        assert first_chat["retrieval"]["worker_status"] == "not_configured"
        assert first_chat["model_requests_attempted"] == 0
        assert {notes["prior"].name, notes["result"].name} <= {
            item["source_path"] for item in first_chat["evidence"]
        }

        # This HTTP approval is exercised only against the temporary test vault.
        # A caller must confirm and provide a conclusion beyond the scaffold.
        test_only_conclusion = (
            "Test-only simulated review of the public QASPER report: I would not "
            "promote either continuation checkpoint. The starting SFT checkpoint "
            f"passed {reported_passes['starting_sft']}/40 questions, epoch 1 passed "
            f"{reported_passes['epoch_1']}/40, and epoch 2 passed "
            f"{reported_passes['epoch_2']}/40 under the reported model-assisted "
            "semantic review. This does not establish whether base Qwen or the "
            "personal-vault Qwen worker improved."
        )
        approval = {
            "review_id": impact["review_id"],
            "title": "Test-only Qwen continuation review",
            "text": test_only_conclusion,
            "kind": "correction",
            "supersedes": impact["prior"]["doc_id"],
            "evidence_ids": impact["proposal"]["evidence_ids"],
            "confirm": False,
        }
        denied = web.post("/api/approve", json=approval)
        assert denied.status_code == 400
        assert web.get("/api/status").json()["documents"] == 3
        approval["confirm"] = True
        approved_response = web.post("/api/approve", json=approval)
        assert approved_response.status_code == 200, approved_response.text
        correction_path = approved_response.json()["saved"]
        correction = vault / correction_path
        correction_text = correction.read_text(encoding="utf-8")
        assert test_only_conclusion in correction_text
        assert f"supersedes_doc_id: {impact['prior']['doc_id']}" in correction_text
        assert "review_status: user_approved" in correction_text
        assert all(path.read_bytes() == original_notes[role]
                   for role, path in notes.items())
        assert web.get("/api/status").json()["documents"] == 4

        later_chat_response = web.post("/api/chat", json={
            "question": "Why did the test-only review reject Qwen continuation checkpoints?",
        })
        assert later_chat_response.status_code == 200, later_chat_response.text
        later_chat = later_chat_response.json()
        assert later_chat["answer"] is None
        assert later_chat["answer_status"] == "evidence_only"
        assert correction_path in {
            item["source_path"] for item in later_chat["evidence"]
        }
        assert web.get("/api/status").json()["model_requests_attempted"] == 0

        # Freeze again after the test-only approval and use the actual MCP
        # protocol client, not a direct call to the retrieval helper.
        snapshot = tmp_path / "mcp-snapshot"
        freeze_vault(vault, snapshot)
        manifest, docs = load_snapshot(snapshot)

        async def check_read_only_mcp() -> None:
            async with Client(create_server(snapshot)) as mcp:
                for query, expected_path in (
                    ("identity-blind semantic review 15/40 10/40 4/40",
                     notes["result"].name),
                    ("Test-only simulated review Qwen continuation checkpoints",
                     correction_path),
                ):
                    found = await mcp.call_tool("search_memory", {
                        "query": query, "top_k": 5,
                    })
                    assert not found.is_error
                    review = found.structured_content
                    assert review["corpus_hash"] == manifest["corpus_hash"]
                    item = next(item for item in review["evidence"]
                                if item["source_path"] == expected_path)
                    assert docs[item["doc_id"]]["text"][
                        item["start"]:item["end"]
                    ] == item["quote"]
                    full_source = await mcp.call_tool("get_memory_source", {
                        "review_id": review["review_id"],
                        "doc_id": item["doc_id"],
                        "start": 0,
                        "length": 8_000,
                    })
                    assert not full_source.is_error
                    source = full_source.structured_content
                    assert source["source_path"] == expected_path
                    assert source["corpus_hash"] == manifest["corpus_hash"]
                    assert source["text"].encode("utf-8") == (
                        vault / expected_path
                    ).read_bytes()

        asyncio.run(check_read_only_mcp())
        assert web.get("/api/status").json()["model_requests_attempted"] == 0
