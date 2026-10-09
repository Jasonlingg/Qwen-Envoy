"""HTTP exercise of the localhost product loop on a copied synthetic vault."""

from pathlib import Path
from shutil import copytree

from fastapi.testclient import TestClient

from scripts.personal_memory_web import create_app

ROOT = Path(__file__).resolve().parents[1]


def test_web_capture_source_approval_and_later_recall(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    app = create_app(vault, tmp_path / "state")
    with TestClient(app) as client:
        home = client.get("/", follow_redirects=False)
        assert home.status_code == 307
        assert home.headers["location"] == "/lab"
        page = client.get("/chat")
        assert page.status_code == 200
        assert "no Qwen or Nemotron calls" in page.text
        status = client.get("/api/status").json()
        assert status["documents"] == 9
        assert status["retriever"] == "lexical_paragraph_baseline"
        assert status["model_calls"] == 0

        captured = client.post("/api/capture", json={
            "title": "One more focus session", "kind": "attempt",
            "effective_date": "2026-09-25",
            "text": "I wrote a clear next action before studying and muted notifications; "
                    "starting felt easier in one session.",
        })
        assert captured.status_code == 200
        assert client.get("/api/status").json()["documents"] == 10

        response = client.get("/api/search", params={
            "query": "correction starting friction phone next action",
        })
        assert response.status_code == 200
        review = response.json()
        assert review["status"] == "evidence_found"
        assert review["retriever"] == "lexical_paragraph_baseline"
        assert review["evidence"][0]["record"]["record_id"] == (
            "life-focus-correction-2026-07-04"
        )
        source = client.get(
            f"/api/source/{review['review_id']}/{review['evidence'][0]['doc_id']}"
        )
        assert source.status_code == 200
        assert "Correction: starting friction" in source.json()["text"]

        approval = {
            "review_id": review["review_id"], "title": "Revised focus lesson",
            "text": "The June phone drawer idea was incomplete. My September session combined "
                    "a next action with muted notifications; one session is not a causal test.",
            "kind": "correction", "effective_date": "2026-09-25",
            "supersedes": "life-focus-plan-2026-06-10",
            "evidence_ids": ["E1", "E2", "E4"], "confirm": False,
        }
        assert client.post("/api/approve", json=approval).status_code == 400
        assert client.get("/api/status").json()["documents"] == 10
        approval["confirm"] = True
        saved = client.post("/api/approve", json=approval)
        assert saved.status_code == 200, saved.text
        assert saved.json()["saved"].startswith("Learning Memory/")
        assert client.get("/api/status").json()["documents"] == 11

        recalled = client.get("/api/search", params={
            "query": "June phone drawer incomplete September muted notifications",
        }).json()
        assert any(item["record"].get("review_status") == "user_approved"
                   for item in recalled["evidence"])


def test_web_no_evidence_does_not_create_approved_memory(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        review = client.get("/api/search", params={"query": "xylophone quasars"}).json()
        assert review["status"] == "no_evidence"
        assert review["evidence"] == []
        response = client.post("/api/approve", json={
            "review_id": review["review_id"], "title": "Unsupported lesson",
            "text": "No source", "kind": "lesson", "evidence_ids": [], "confirm": True,
        })
        assert response.status_code == 400
        assert client.get("/api/status").json()["documents"] == 9


def test_web_rejects_approval_after_vault_changes(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        review = client.get("/api/search", params={"query": "phone drawer"}).json()
        assert review["evidence"]
        captured = client.post("/api/capture", json={
            "title": "New observation", "kind": "attempt",
            "text": "I tried a new focus routine today.",
        })
        assert captured.status_code == 200
        approval = client.post("/api/approve", json={
            "review_id": review["review_id"], "title": "Old review",
            "text": "This review predates the new observation.",
            "kind": "lesson", "evidence_ids": ["E1"], "confirm": True,
        })
        assert approval.status_code == 409
        assert client.get("/api/status").json()["documents"] == 10


def test_status_lists_five_recent_results_from_imported_and_captured_notes(tmp_path):
    vault = tmp_path / "vault"
    research = vault / "Research"
    research.mkdir(parents=True)
    for day in range(1, 7):
        (research / f"qwen-{day:02d}.md").write_text(
            "---\n"
            "kind: result\n"
            f"effective_date: 2026-09-{day:02d}\n"
            "---\n\n"
            f"# Visible Qwen result {day:02d}\n\n"
            f"Qwen experiment result recorded on September {day}.\n",
            encoding="utf-8",
        )
    (research / "captured-without-effective-date.md").write_text(
        "---\nkind: result\ncaptured_at: 2026-09-07T12:00:00+00:00\n---\n\n"
        "# Result with capture date only\n\nThis result has no effective date.\n",
        encoding="utf-8",
    )
    (research / "newer-decision.md").write_text(
        "---\nkind: decision\neffective_date: 2026-09-08\n---\n\n"
        "# Newer decision\n\nThis is not a result.\n",
        encoding="utf-8",
    )

    with TestClient(create_app(vault, tmp_path / "state")) as client:
        initial = client.get("/api/status")
        assert initial.status_code == 200, initial.text
        recent = initial.json()["recent_results"]
        assert len(recent) == 5
        assert [item["source_path"] for item in recent] == [
            "Research/captured-without-effective-date.md",
            *[f"Research/qwen-{day:02d}.md" for day in range(6, 2, -1)],
        ]
        assert recent[0]["title"] == "Result with capture date only"
        assert recent[1]["title"] == "Visible Qwen result 06"
        assert recent[1]["effective_date"] == "2026-09-06"
        assert all(item["source_path"] != "Research/newer-decision.md" for item in recent)

        captured = client.post("/api/capture", json={
            "title": "New Qwen result",
            "kind": "result",
            "effective_date": "2026-09-09",
            "text": "A new measured Qwen result for this research thread.",
        })
        assert captured.status_code == 200, captured.text
        saved = captured.json()["saved"]
        updated = client.get("/api/status")
        assert updated.status_code == 200, updated.text
        recent = updated.json()["recent_results"]
        assert len(recent) == 5
        assert [item["source_path"] for item in recent] == [
            saved,
            "Research/captured-without-effective-date.md",
            *[f"Research/qwen-{day:02d}.md" for day in range(6, 3, -1)],
        ]
        assert recent[0]["source_path"] == saved
        assert recent[0]["title"] == "New Qwen result"
        assert recent[0]["effective_date"] == "2026-09-09"
        assert all(item["source_path"] != "Research/newer-decision.md" for item in recent)
