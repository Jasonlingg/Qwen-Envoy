"""Launch a disposable, manually curated map of the original Transformer.

At launch, freeze the 2017 paper's arXiv HTML through the learning-thread API.
If pypdf and the 2018 OpenAI paper are available, attach that original PDF and
stage a reviewable GPT comparison. No model is called. The temporary vault and
its frozen source snapshots disappear when the server exits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import tempfile
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import uvicorn
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.personal_memory_web import create_app  # noqa: E402

TRANSFORMER_URL = "https://arxiv.org/html/1706.03762"
GPT_PDF_URL = (
    "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
)
QUESTION = "How does the original 2017 Transformer encoder–decoder produce the next output token?"


def _request(client: TestClient, method: str, url: str, **kwargs) -> dict:
    response = client.request(method, url, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {url} failed: {response.status_code} {response.text}")
    return response.json()


def _source_text(state_dir: Path, thread_id: str, source_id: str) -> str:
    folder = state_dir / "learning_lab" / "sources" / thread_id / source_id / "corpus"
    documents = list(folder.glob("*.json"))
    if len(documents) != 1:
        raise RuntimeError("Expected one frozen document for the Transformer map source")
    return json.loads(documents[0].read_text(encoding="utf-8"))["text"]


def _evidence(source_id: str, text: str, first: str, last: str) -> dict:
    """Locate a short exact span in the imported snapshot, never in live text."""
    if text.count(first) != 1:
        raise ValueError(f"Source wording changed or became ambiguous: {first!r}")
    start = text.index(first)
    end = text.find(last, start)
    if end < 0:
        raise ValueError(f"Source wording changed: {last!r}")
    end += len(last)
    if not 0 < end - start <= 1_200:
        raise ValueError("Source evidence span must be between 1 and 1,200 characters")
    return {"source_id": source_id, "start": start, "end": end, "relation": "supports"}


def _node(id: str, label: str, description: str, x: int, y: int, type: str = "process") -> dict:
    return {"id": id, "type": type, "label": label, "description": description, "x": x, "y": y}


def _edge(
    id: str,
    source: str,
    target: str,
    label: str,
    explanation: str,
    evidence: list[dict],
    *,
    type: str = "flows_to",
    basis: str = "documented",
) -> dict:
    return {
        "id": id,
        "source": source,
        "target": target,
        "type": type,
        "label": label,
        "explanation": explanation,
        "basis": basis,
        "evidence": evidence,
    }


def _original_map(source_id: str, text: str) -> tuple[list[dict], list[dict]]:
    """Show one prediction in the 2017 sequence-to-sequence architecture."""
    # Map validation requires a distinct frozen span for every arrow. Short,
    # sometimes overlapping excerpts keep each receipt easy to inspect.
    spans = {
        "input_embed": _evidence(
            source_id,
            text,
            "Similarly to other sequence transduction models, we use learned embeddings",
            "the input tokens and output tokens to vectors",
        ),
        "input_position": _evidence(
            source_id,
            text,
            'To this end, we add "positional encodings"',
            "the encoder and decoder stacks.",
        ),
        "encoder_order": _evidence(
            source_id,
            text,
            "Each layer has two sub-layers. The first is a multi-head self-attention mechanism",
            "fully connected feed-forward network.",
        ),
        "representations": _evidence(
            source_id,
            text,
            "Here, the encoder maps an input sequence",
            "a sequence of continuous representations",
        ),
        "prior_embed": _evidence(
            source_id,
            text,
            "At each step the model is auto-regressive",
            "previously generated symbols as additional input",
        ),
        "output_shift": _evidence(
            source_id,
            text,
            "This masking, combined with fact",
            "output embeddings are offset by one position",
        ),
        "masked": _evidence(
            source_id,
            text,
            "We also modify the self-attention sub-layer in the decoder stack",
            "prevent positions from attending to subsequent positions.",
        ),
        "decoder_order": _evidence(
            source_id,
            text,
            "In addition to the two sub-layers in each encoder layer",
            "output of the encoder stack.",
        ),
        "cross": _evidence(
            source_id,
            text,
            'In "encoder-decoder attention" layers',
            "all positions in the input sequence.",
        ),
        "decoder_ffn": _evidence(
            source_id,
            text,
            "In addition to attention sub-layers, each of the layers in our encoder and decoder",
            "applied to each position separately and identically.",
        ),
        "output_project": _evidence(
            source_id,
            text,
            "We also use the usual learned linear transformation",
            "predicted next-token probabilities.",
        ),
        "output_token": _evidence(
            source_id,
            text,
            "softmax function to convert the decoder output",
            "predicted next-token probabilities.",
        ),
        "decoder_generates": _evidence(
            source_id,
            text,
            "the decoder then generates an output sequence",
            "symbols one element at a time.",
        ),
        "repeat": _evidence(
            source_id,
            text,
            "consuming the previously generated symbols",
            "when generating the next.",
        ),
    }
    nodes = [
        _node(
            "source_tokens",
            "Source tokens",
            "The input sequence, such as the sentence to translate.",
            10,
            15,
            "component",
        ),
        _node(
            "source_embed",
            "Input embeddings + position",
            "Learned token embeddings plus positional encodings enter the encoder stack.",
            240,
            15,
            "component",
        ),
        _node(
            "encoder_self",
            "Encoder self-attention",
            "First sublayer of each of six encoder layers; positions can attend "
            "across the input. Residual and layer norm surround it.",
            470,
            15,
        ),
        _node(
            "encoder_ffn",
            "Encoder feed-forward",
            "Second sublayer of each encoder layer; the same network acts "
            "separately at each position.",
            700,
            15,
        ),
        _node(
            "encoder_states",
            "Encoded representations",
            "A vector at each source position; these become memory for the decoder.",
            700,
            180,
            "stock",
        ),
        _node(
            "prior_outputs",
            "Previous output tokens",
            "Known output symbols from earlier steps; the decoder predicts one more.",
            10,
            345,
            "component",
        ),
        _node(
            "target_embed",
            "Output embeddings + position",
            "Shifted output embeddings plus positional encodings enter the decoder stack.",
            240,
            345,
            "component",
        ),
        _node(
            "masked_self",
            "Masked self-attention",
            "First decoder sublayer; a causal mask prevents each position "
            "from seeing later output positions.",
            470,
            345,
        ),
        _node(
            "cross_attention",
            "Encoder–decoder attention",
            "Decoder queries attend to keys and values from the encoder output.",
            700,
            345,
        ),
        _node(
            "decoder_ffn",
            "Decoder feed-forward",
            "Third decoder sublayer, applied independently at each position.",
            700,
            510,
        ),
        _node(
            "linear_softmax",
            "Linear + softmax",
            "Converts the decoder output into next-token probabilities.",
            240,
            510,
        ),
        _node(
            "next_token",
            "Next output token",
            "A symbol is generated from the predicted probabilities; it "
            "becomes prior context for the following step.",
            10,
            510,
            "outcome",
        ),
    ]
    edges = [
        _edge(
            "input_to_embed",
            "source_tokens",
            "source_embed",
            "embed inputs",
            "The paper converts input symbols into learned vectors.",
            [spans["input_embed"]],
        ),
        _edge(
            "embed_to_encoder",
            "source_embed",
            "encoder_self",
            "add positions",
            "Positional encodings are added to input embeddings before the encoder stack.",
            [spans["input_position"]],
        ),
        _edge(
            "encoder_self_to_ffn",
            "encoder_self",
            "encoder_ffn",
            "then feed-forward",
            "Each encoder layer has multi-head self-attention followed by a "
            "position-wise feed-forward sublayer.",
            [spans["encoder_order"]],
        ),
        _edge(
            "ffn_to_states",
            "encoder_ffn",
            "encoder_states",
            "encode source",
            "The encoder maps source symbols to a sequence of continuous representations.",
            [spans["representations"]],
        ),
        _edge(
            "prior_to_embed",
            "prior_outputs",
            "target_embed",
            "embed prior outputs",
            "Previously generated output symbols become decoder input; "
            "output embeddings are shifted one position.",
            [spans["prior_embed"], spans["output_shift"]],
        ),
        _edge(
            "embed_to_mask",
            "target_embed",
            "masked_self",
            "mask future",
            "The decoder masks self-attention so a position cannot see later outputs.",
            [spans["masked"]],
        ),
        _edge(
            "mask_to_cross",
            "masked_self",
            "cross_attention",
            "attend to source",
            "The decoder inserts an attention sublayer over encoder output "
            "after masked self-attention.",
            [spans["decoder_order"]],
        ),
        _edge(
            "memory_to_cross",
            "encoder_states",
            "cross_attention",
            "keys + values",
            "Cross-attention takes its keys and values from the encoder output.",
            [spans["cross"]],
        ),
        _edge(
            "cross_to_ffn",
            "cross_attention",
            "decoder_ffn",
            "then feed-forward",
            "Each decoder layer also contains a position-wise feed-forward network.",
            [spans["decoder_ffn"]],
        ),
        _edge(
            "ffn_to_softmax",
            "decoder_ffn",
            "linear_softmax",
            "project to vocabulary",
            "The decoder output is converted to predicted next-token "
            "probabilities by a learned linear transform and softmax.",
            [spans["output_project"]],
        ),
        _edge(
            "softmax_to_token",
            "linear_softmax",
            "next_token",
            "generate token",
            "Softmax yields next-token probabilities, from which the decoder "
            "generates an output symbol. The paper does not specify a "
            "particular decoding rule here.",
            [spans["output_token"], spans["decoder_generates"]],
        ),
        _edge(
            "token_to_prior",
            "next_token",
            "prior_outputs",
            "repeat next step",
            "Autoregressive generation consumes previously generated symbols "
            "to predict the next one.",
            [spans["repeat"]],
        ),
    ]
    return nodes, edges


def _download_gpt_pdf() -> bytes:
    """Fetch only the published OpenAI PDF, bounded by the attachment limit."""
    request = Request(GPT_PDF_URL, headers={"User-Agent": "Envoy-research/0.1"})
    with urlopen(request, timeout=15) as response:
        if response.geturl() != GPT_PDF_URL:
            raise ValueError("GPT PDF redirected away from the pinned OpenAI URL")
        raw = response.read(5_000_001)
    if not raw.startswith(b"%PDF-") or len(raw) > 5_000_000:
        raise ValueError("OpenAI source is not a PDF within the 5 MB limit")
    return raw


def _stage_gpt_comparison(
    client: TestClient,
    state_dir: Path,
    thread_id: str,
    base_revision_id: str,
    initial_nodes: list[dict],
    initial_edges: list[dict],
) -> None:
    if importlib.util.find_spec("pypdf") is None:
        raise RuntimeError("pypdf is unavailable; install the viewer extra for the GPT comparison")
    raw = _download_gpt_pdf()
    source = _request(
        client,
        "POST",
        f"/api/lab/threads/{thread_id}/files",
        params={"filename": "Radford_et_al_2018_GPT.pdf"},
        content=raw,
        headers={"content-type": "application/pdf"},
    )
    gpt_text = _source_text(state_dir, thread_id, source["source_id"])
    gpt_span = _evidence(
        source["source_id"],
        gpt_text,
        "we use a multi-layer Transformer decoder",
        "over target tokens:",
    )
    proposed_nodes = [dict(node) for node in initial_nodes] + [
        _node(
            "gpt_variant",
            "GPT (2018) decoder variant",
            "The 2018 paper uses a Transformer decoder over context tokens "
            "for language modeling. Compare its path with the original "
            "2017 model.",
            470,
            180,
            "component",
        ),
        _node(
            "trace_exercise",
            "Trace one next token",
            "Follow one prediction in each architecture. Which "
            "encoder-memory link is absent from the GPT language model?",
            240,
            180,
            "process",
        ),
    ]
    proposed_edges = [dict(edge) for edge in initial_edges] + [
        _edge(
            "gpt_to_exercise",
            "gpt_variant",
            "trace_exercise",
            "compare this variant",
            "The GPT paper describes a multi-layer Transformer decoder with "
            "multi-head self-attention and feed-forward layers; use this "
            "source while tracing its language-model path.",
            [gpt_span],
            type="tests",
        ),
        _edge(
            "exercise_to_cross",
            "trace_exercise",
            "cross_attention",
            "which link disappears?",
            "Compare the original decoder's encoder-memory connection with "
            "the later GPT language-model path. The absent link is an "
            "architectural comparison, not a contradiction of the 2017 paper.",
            [],
            type="tests",
            basis="hypothesis",
        ),
    ]
    proposal = {
        "thread_id": thread_id,
        "authorship": "curated_demo",
        "title": "Compare the 2017 model with a later decoder-only variant",
        "rationale": (
            "The original 2017 architecture has an encoder and a decoder with "
            "cross-attention. The 2018 GPT paper describes a Transformer decoder "
            "for language modeling. Trace a token through both and inspect the "
            "source passages before accepting this comparison. This is a curated "
            "learning exercise, not a Qwen-generated result or a contradiction."
        ),
        "summary": "Add a sourced GPT decoder variant and a next-token tracing exercise.",
        "base_revision_id": base_revision_id,
        "proposed_map": {"nodes": proposed_nodes, "edges": proposed_edges},
    }
    folder = state_dir / "learning_lab" / "map_proposals"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{thread_id}.json").write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def seed_demo(client: TestClient, state_dir: Path, *, include_gpt: bool = True) -> str:
    thread = _request(
        client,
        "POST",
        "/api/lab/threads",
        json={
            "question": QUESTION,
            "current_view": (
                "The original paper describes source tokens flowing through an encoder, "
                "while the decoder uses prior outputs and encoder memory to predict the "
                "next token. This working map is manually curated from the papers."
            ),
            "effective_date": date.today().isoformat(),
        },
    )
    thread_id = thread["thread_id"]
    source = _request(
        client, "POST", f"/api/lab/threads/{thread_id}/sources", json={"url": TRANSFORMER_URL}
    )
    paper_text = _source_text(state_dir, thread_id, source["source_id"])
    nodes, edges = _original_map(source["source_id"], paper_text)
    saved = _request(
        client,
        "PUT",
        f"/api/lab/threads/{thread_id}/map",
        json={
            "summary": (
                "Curated flow of the original 2017 encoder–decoder model, "
                "linked to frozen paper excerpts."
            ),
            "nodes": nodes,
            "edges": edges,
        },
    )
    if include_gpt:
        try:
            _stage_gpt_comparison(
                client, state_dir, thread_id, saved["revision"]["id"], nodes, edges
            )
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            print(f"GPT comparison unavailable: {exc}", file=sys.stderr)
    return thread_id


def prepare_demo(workspace: Path, *, include_gpt: bool = True):
    """Build an isolated local app and return it with the seeded thread ID."""
    vault, state = workspace / "vault", workspace / "state"
    vault.mkdir(parents=True)
    state.mkdir(parents=True)
    app = create_app(vault, state)
    with TestClient(app) as client:
        thread_id = seed_demo(client, state, include_gpt=include_gpt)
    return app, thread_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    def stop_on_term(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        with tempfile.TemporaryDirectory(prefix="envoy-transformer-map-demo-") as temp:
            app, thread_id = prepare_demo(Path(temp))
            with TestClient(app) as client:
                thread = _request(client, "GET", f"/api/lab/threads/{thread_id}")
                proposal = client.get(f"/api/lab/threads/{thread_id}/map/proposal")
            print(
                json.dumps(
                    {
                        "url": f"http://127.0.0.1:{args.port}/lab/map?thread={thread_id}",
                        "question": QUESTION,
                        "mode": "curated_source_linked_demo",
                        "sources": [
                            {"title": item["title"], "snapshot_hash": item["source_snapshot_hash"]}
                            for item in thread["sources"]
                        ],
                        "gpt_comparison_available": proposal.status_code == 200,
                        "model_calls": 0,
                        "temporary": True,
                    }
                ),
                flush=True,
            )
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
