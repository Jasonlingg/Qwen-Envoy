"""The curated Transformer map stays source-linked and network independent in CI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from scripts import launch_transformer_map_demo as demo
from src.product import learning_lab_api
from src.research.file_sources import build_file_snapshot
from src.research.web_sources import FetchedPage, build_web_snapshot

# Excerpts preserving the paper's relevant wording exercise the actual HTML
# snapshot/parser/API path without depending on arXiv availability in tests.
TRANSFORMER_HTML = """<!doctype html><html><head><title>Attention Is All You Need</title></head>
<body><article>
<p>Here, the encoder maps an input sequence to a sequence of continuous
representations. Given them, the decoder then generates an output sequence of
symbols one
element at a time. At each step the model is auto-regressive, consuming the
previously generated symbols as additional input when generating the next.</p>
<p>Each layer has two sub-layers. The first is a multi-head self-attention
mechanism, and the second is a position-wise fully connected feed-forward
network.</p>
<p>In addition to the two sub-layers in each encoder layer, the decoder
inserts a third sub-layer, which performs multi-head attention over the output
of the encoder stack. We also modify the self-attention sub-layer in the decoder
stack to prevent positions from attending to subsequent positions. This masking,
combined with fact that the output embeddings are offset by one position.</p>
<p>In "encoder-decoder attention" layers, the queries come from the previous
decoder layer, and the memory keys and values come from the output of the encoder.
This allows every position in the decoder to attend over all positions in the
input sequence.</p>
<p>In addition to attention sub-layers, each of the layers in our encoder and
decoder contains a fully connected feed-forward network, which is applied to
each position separately and identically.</p>
<p>Similarly to other sequence transduction models, we use learned embeddings
to convert the input tokens and output tokens to vectors. We also use the usual
learned linear transformation and softmax function to convert the decoder output
to predicted next-token probabilities.</p>
<p>To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks.</p>
</article></body></html>"""

GPT_TEXT = (
    "In our experiments, we use a multi-layer Transformer decoder for the "
    "language model, which is a variant of the transformer. This model applies "
    "a multi-headed self-attention operation over the input context tokens "
    "followed by position-wise feedforward layers to produce an output "
    "distribution over target tokens:"
)


def _offline_arxiv(monkeypatch):
    def freeze(url, output):
        assert url == demo.TRANSFORMER_URL
        return build_web_snapshot(
            url,
            output,
            downloader=lambda requested: FetchedPage(
                body=TRANSFORMER_HTML.encode("utf-8"), final_url=requested
            ),
        )

    monkeypatch.setattr(learning_lab_api, "build_web_snapshot", freeze)


def test_original_architecture_has_distinct_verified_paper_spans(tmp_path, monkeypatch):
    _offline_arxiv(monkeypatch)
    app, thread_id = demo.prepare_demo(tmp_path, include_gpt=False)
    with TestClient(app) as client:
        url = f"/api/lab/threads/{thread_id}/map"
        thread = client.get(f"/api/lab/threads/{thread_id}").json()
        map_state = client.get(url).json()
        nodes = map_state["revision"]["nodes"]
        edges = map_state["revision"]["edges"]

        assert "original 2017" in thread["question"]
        assert len(thread["sources"]) == 1
        assert thread["sources"][0]["source_url"] == demo.TRANSFORMER_URL
        assert len(nodes) == 12 and len(edges) == 12
        assert len({n["id"] for n in nodes}) == len(nodes)
        assert len({e["id"] for e in edges}) == len(edges)
        assert all(edge["basis"] == "documented" for edge in edges)
        receipts = [receipt for edge in edges for receipt in edge["receipts"]]
        assert len(receipts) == len({r["reference"] for r in receipts})
        assert all(r["status"] == "exact_span_verified" for r in receipts)
        assert all(r["source_url"] == demo.TRANSFORMER_URL for r in receipts)
        assert all(0 < len(r["quote"]) <= 1_200 for r in receipts)

        cross = next(e for e in edges if e["id"] == "memory_to_cross")
        assert cross["source"] == "encoder_states"
        assert cross["target"] == "cross_attention"
        assert "keys and values" in cross["receipts"][0]["quote"]
        shifted = next(e for e in edges if e["id"] == "prior_to_embed")
        assert any("offset by one position" in r["quote"] for r in shifted["receipts"])
        output = next(e for e in edges if e["id"] == "softmax_to_token")
        assert "probabilities" in output["explanation"]
        assert client.get(f"{url}/proposal").status_code == 404
        assert client.get("/api/status").json()["model_calls"] == 0


def test_gpt_variant_is_a_separate_reviewable_comparison(tmp_path, monkeypatch):
    _offline_arxiv(monkeypatch)

    # The app's PDF parser has its own tests. This test isolates proposal
    # behavior by substituting text at the file-snapshot boundary.
    original_find_spec = demo.importlib.util.find_spec
    monkeypatch.setattr(
        demo.importlib.util,
        "find_spec",
        lambda name: object() if name == "pypdf" else original_find_spec(name),
    )
    monkeypatch.setattr(demo, "_download_gpt_pdf", lambda: b"%PDF-test fixture")

    def freeze_gpt(filename, body, output):
        assert filename == "Radford_et_al_2018_GPT.pdf"
        assert body == b"%PDF-test fixture"
        return build_file_snapshot("Radford_et_al_2018_GPT.txt", GPT_TEXT.encode(), output)

    monkeypatch.setattr(learning_lab_api, "build_file_snapshot", freeze_gpt)
    app, thread_id = demo.prepare_demo(tmp_path)
    with TestClient(app) as client:
        map_url = f"/api/lab/threads/{thread_id}/map"
        current = client.get(map_url).json()
        response = client.get(f"{map_url}/proposal")
        assert response.status_code == 200, response.text
        proposal = response.json()
        assert proposal["authorship"] == "curated_demo"
        assert proposal["base_revision_id"] == current["revision"]["id"]
        assert len(client.get(f"/api/lab/threads/{thread_id}").json()["sources"]) == 2
        proposed = proposal["proposed_map"]
        assert len(proposed["nodes"]) == 14
        assert not any(
            edge["source"] == "gpt_variant"
            and edge["target"] in {"masked_self", "cross_attention", "decoder_ffn"}
            for edge in proposed["edges"]
        )
        gpt = next(e for e in proposed["edges"] if e["id"] == "gpt_to_exercise")
        assert gpt["target"] == "trace_exercise"
        assert gpt["receipts"][0]["status"] == "exact_span_verified"
        assert "variant of the transformer" in gpt["receipts"][0]["quote"]
        exercise = next(e for e in proposed["edges"] if e["id"] == "exercise_to_cross")
        assert exercise["type"] == "tests"
        assert exercise["evidence"] == []
        assert len(current["history"]) == 1

        accepted = client.put(
            map_url,
            json={
                "nodes": proposed["nodes"],
                "edges": proposed["edges"],
                "summary": proposal["summary"],
            },
        )
        assert accepted.status_code == 200, accepted.text
        assert len(accepted.json()["history"]) == 2
        assert client.get(f"{map_url}/proposal").status_code == 404
        assert client.get("/api/status").json()["model_calls"] == 0
