"""The browser demo exposes the chat and source-review contract in offline mode."""

from html.parser import HTMLParser
from pathlib import Path
from shutil import copytree

from fastapi.testclient import TestClient

from scripts.personal_memory_web import create_app

ROOT = Path(__file__).resolve().parents[1]


class _PageControls(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.labels: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        if tag == "label" and attributes.get("for"):
            self.labels.append(attributes["for"])


def test_chat_page_and_offline_response_keep_review_separate(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    with TestClient(create_app(vault, tmp_path / "state")) as client:
        page = client.get("/chat")
        assert page.status_code == 200
        controls = _PageControls()
        controls.feed(page.text)
        assert len(controls.ids) == len(set(controls.ids))
        assert {"chat-form", "question", "conversation", "source-dialog",
                "revision-form", "revision-confirm"} <= set(controls.ids)
        assert "question" in controls.labels
        assert "Offline mode makes no Qwen or Nemotron calls" in page.text
        assert 'jsonPost("/api/chat"' in page.text

        turn_response = client.post("/api/chat", json={
            "question": "How did my phone-drawer focus plan change?",
        })
        assert turn_response.status_code == 200
        turn = turn_response.json()
        assert turn["answer_status"] == "evidence_only"
        assert turn["answer"] is None
        assert turn["retrieval"]["mode"] == "lexical_paragraph_baseline"
        assert turn["evidence"]
        first = turn["evidence"][0]
        source = client.get(
            f"/api/source/{turn['review_id']}/{first['doc_id']}"
        )
        assert source.status_code == 200
        assert first["quote"] in source.json()["text"]
        assert client.get("/api/status").json()["model_calls"] == 0
        assert client.get("/api/status").json()["documents"] == 9
