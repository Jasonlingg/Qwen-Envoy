"""Offline tests for the no-inference preparation and bounded comparator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.envoybench import retrieval_baseline as baseline
from benchmarks.envoybench import run as runner
from benchmarks.envoybench.score import prepare_provisional_bundle
from src.eval.artifacts import configuration_hash, content_hash


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class FakeCounter:
    """Deliberately simple fixture; never reported as production token usage."""

    def __init__(self, revision: str, kwargs: dict) -> None:
        self.metadata = {"test_fixture": True, "revision": revision, "kwargs": kwargs}

    def __call__(self, messages: list[dict]) -> int:
        return 20 + sum(len(message["content"]) // 4 for message in messages)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    dataset = tmp_path / "data"
    source = tmp_path / "agent-run"
    corpus = dataset / "dev/corpus"
    text = "Blue is the measured color. " * 20
    write(corpus / "paper_a.json", {"doc_id": "paper_a", "title": "Color study", "text": text})
    question = {
        "id": "q1", "split": "pilot_evaluation",
        "question": 'Use the known paper "Color study" (doc_id: "paper_a") '
                    'to answer: What color was measured?',
        "answer": "gold-reference-never-in-prompt", "expected_citations": ["paper_a"],
        "expected_answerability": "sufficient", "required_doc_ids": ["paper_a"],
        "minimum_distinct_sources": 1, "grader_notes": ["gold-evidence-never-in-prompt"],
        "source_paper_id": "paper_a",
    }
    benchmark = {"schema_version": "research-benchmark-v1", "benchmark_id": "fixture",
                 "corpus_hash": content_hash(corpus), "reserved_doc_ids": ["paper_a"],
                 "questions": [question]}
    benchmark_path = dataset / "dev/benchmark.json"
    write(benchmark_path, benchmark)
    benchmark_file_hash = runner._file_sha256(benchmark_path)
    write(dataset / "dev/manifest.json", {
        "schema_version": "research-snapshot-v1", "corpus_hash": benchmark["corpus_hash"],
        "split_status": "previously_used_development", "benchmark_sha256": benchmark_file_hash,
        "selected_question_ids": ["q1"], "selected_target_paper_ids": ["paper_a"],
        "papers": [{"doc_id": "paper_a"}],
    })
    splits = dataset / "splits.json"
    write(splits, {"schema_version": "envoybench-splits-v1",
                   "splits": {"dev": {"question_ids": ["q1"]}}})
    write(dataset / "manifest.json", {
        "schema_version": "envoybench-data-manifest-v1",
        "frozen_splits_sha256": runner._file_sha256(splits),
        "splits": {"dev": {
            "benchmark": "dev/benchmark.json", "corpus": "dev/corpus",
            "manifest": "dev/manifest.json", "status": "previously_used_development",
            "benchmark_sha256": benchmark_file_hash, "corpus_hash": benchmark["corpus_hash"],
            "question_count": 1,
        }},
    })
    models_path = tmp_path / "models.json"
    models = [{
        "key": "base", "model_id": baseline.TOKENIZER_ID, "revision": "base-revision",
        "endpoint_env": "BASELINE_TEST_ENDPOINT", "api_key_env": "BASELINE_TEST_KEY",
        "decoding": {"max_tokens": 64, "temperature": 0, "top_p": 1},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }, {
        "key": "v5", "model_id": "envoy", "revision": "adapter-revision",
        "base_model_id": baseline.TOKENIZER_ID, "base_revision": "base-revision",
        "adapter_id": "fixture/v5", "adapter_revision": "adapter-revision",
        "endpoint_env": "BASELINE_TEST_ENDPOINT", "api_key_env": "BASELINE_TEST_KEY",
        "decoding": {"max_tokens": 64, "temperature": 0, "top_p": 1},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }]
    write(models_path, {"schema_version": runner.MODEL_SCHEMA, "models": models})
    monkeypatch.setenv("BASELINE_TEST_ENDPOINT", "http://localhost:8000/v1")
    monkeypatch.setenv("BASELINE_TEST_KEY", "test-secret-not-in-artifacts")
    manifest = {
        "schema_version": runner.RUN_SCHEMA, "run_id": "agent-run", "status": "complete",
        "comparison_id": "agent-comparison", "benchmark_id": benchmark["benchmark_id"],
        "benchmark_hash": configuration_hash(benchmark), "corpus_hash": benchmark["corpus_hash"],
        "question_ids": ["q1"], "full_split": True, "subset_smoke": False, "split": "dev",
        "split_status": "previously_used_development", "seed": 42,
        "models": [item["safe"] for item in runner.load_models(models_path)],
    }
    write(source / "manifest.json", manifest)
    return {"dataset": dataset, "source": source, "models": models_path,
            "benchmark": benchmark, "question": question, "corpus": corpus,
            "document": {"doc_id": "paper_a", "text": text}, "root": tmp_path}


def prepared(setup: dict) -> Path:
    path = setup["root"] / "prepared"
    baseline.prepare(setup["dataset"], setup["source"], path, context_window=2048,
                     counter_factory=FakeCounter)
    return path


def test_deterministic_rank_has_exact_offsets_and_no_reference_dependency(setup: dict) -> None:
    ranked = baseline.rank_passages("irrelevant filler " * 20 + "blue measured color " * 10,
                                    "measured color", chunk_chars=128, overlap_chars=32)
    assert "measured color" in ranked[0]["text"]
    assert ranked == baseline.rank_passages(
        "irrelevant filler " * 20 + "blue measured color " * 10,
        "measured color", chunk_chars=128, overlap_chars=32,
    )
    counter = FakeCounter("revision", {})
    kwargs = {"context_window": 2048, "output_reserve": 64}
    original = baseline.make_packet(setup["question"], setup["document"], counter, **kwargs)
    corrupted_gold = {**setup["question"], "answer": "Red", "grader_notes": ["opposite"],
                      "expected_answerability": "insufficient", "required_doc_ids": ["other"]}
    assert original == baseline.make_packet(corrupted_gold, setup["document"], counter, **kwargs)
    assert "gold-reference" not in json.dumps(original)
    for passage in original["passages"]:
        assert passage["text"] == setup["document"]["text"][passage["start"]:passage["end"]]


def test_budget_counts_complete_prompt_and_reserves_output(setup: dict) -> None:
    question, document = setup["question"], setup["document"]
    counter = FakeCounter("revision", {})
    packet = baseline.make_packet(question, document, counter, context_window=650,
                                  output_reserve=128, safety_margin=32, chunk_chars=128,
                                  overlap_chars=32)
    assert packet["prompt_tokens_local"] == counter(packet["messages"])
    assert packet["prompt_tokens_local"] + 128 + 32 <= 650
    assert len(packet["passages"]) < len(baseline.rank_passages(
        document["text"], "What color was measured?", chunk_chars=128, overlap_chars=32,
    ))
    with pytest.raises(ValueError, match="instructions alone"):
        baseline.make_packet(question, document, counter, context_window=128, output_reserve=64)


def test_prepare_needs_no_endpoint_and_is_explicitly_not_run(setup: dict,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BASELINE_TEST_ENDPOINT")
    monkeypatch.delenv("BASELINE_TEST_KEY")
    monkeypatch.setattr(baseline, "_post_json", lambda *args: pytest.fail("network request"))
    path = prepared(setup)
    plan = json.loads((path / "plan.json").read_text())
    assert plan["status"] == "prepared_not_run"
    assert plan["requests_planned"] == 2
    assert plan["requests_executed"] == 0
    assert plan["provider_token_usage"] is None
    assert not (path / "results.json").exists()
    combined = (path / "plan.json").read_text() + (path / "packets.json").read_text()
    assert "gold-reference-never-in-prompt" not in combined
    assert "gold-evidence-never-in-prompt" not in combined
    assert "test-secret-not-in-artifacts" not in combined


def test_execute_preserves_settings_usage_and_produces_reviewable_results(setup: dict) -> None:
    path = prepared(setup)
    requests = []

    def transport(url, payload, headers, timeout):
        requests.append(payload)
        assert payload["max_tokens"] == 64
        assert payload["temperature"] == 0
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "seed" not in payload
        return {"choices": [{"message": {"content": 'SUBMIT: Blue CITATIONS: ["paper_a"] '
                                         'EVIDENCE: [{"doc_id":"paper_a","start":0,"end":26}]'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 24, "total_tokens": 924}}

    output = setup["root"] / "executed"
    manifest = baseline.execute(path, setup["dataset"], setup["source"], setup["models"], output,
                                context_window=2048, counter_factory=FakeCounter,
                                transport=transport)
    rows = json.loads((output / "results.json").read_text())
    assert len(requests) == 2
    assert manifest["architecture"] == baseline.VERSION
    assert manifest["source_run_id"] == "agent-run"
    assert {row["model_key"] for row in rows} == {"base_retrieval", "v5_retrieval"}
    assert rows[0]["provider_token_usage"]["prompt_tokens"] == 900
    assert rows[0]["prompt_tokens_local"] != 900
    assert all(row["duration_seconds"] > 0 for row in rows)
    assert all(row["evidence_diagnostics"] == [] for row in rows)
    review, _, _ = prepare_provisional_bundle(setup["benchmark"], manifest, rows, setup["corpus"])
    assert len(review["rows"]) == 2


@pytest.mark.parametrize("mutation", ["packet", "policy", "context"])
def test_tamper_or_mismatch_refused_before_inference(setup: dict, mutation: str) -> None:
    path = prepared(setup)
    context = 2048
    if mutation == "packet":
        packet_path = path / "packets.json"
        packets = json.loads(packet_path.read_text())
        packets["packets"][0]["messages"][1]["content"] += " injected oracle answer"
        write(packet_path, packets)
    elif mutation == "policy":
        models = json.loads(setup["models"].read_text())
        models["models"][0]["decoding"]["temperature"] = 1
        write(setup["models"], models)
    else:
        context = 4096
    with pytest.raises(ValueError, match="changed|differs"):
        baseline.execute(path, setup["dataset"], setup["source"], setup["models"],
                         setup["root"] / "executed", context_window=context,
                         counter_factory=FakeCounter,
                         transport=lambda *args: pytest.fail("mismatched inference happened"))


def test_failed_request_records_elapsed_time_and_redacts_secret(setup: dict) -> None:
    path = prepared(setup)

    def failed(*args):
        raise RuntimeError("test-secret-not-in-artifacts failure")

    output = setup["root"] / "failed"
    baseline.execute(path, setup["dataset"], setup["source"], setup["models"], output,
                     context_window=2048, counter_factory=FakeCounter, transport=failed)
    rows = json.loads((output / "results.json").read_text())
    assert all(row["status"] == "error" for row in rows)
    assert all(row["duration_seconds"] > 0 for row in rows)
    assert all(row["provider_token_usage"] is None for row in rows)
    assert "test-secret-not-in-artifacts" not in (output / "results.json").read_text()


def test_evidence_not_seen_in_packet_is_reported_without_inventing_support(setup: dict) -> None:
    counter = FakeCounter("revision", {})
    packet = baseline.make_packet(setup["question"], setup["document"], counter,
                                  context_window=2048, output_reserve=64,
                                  chunk_chars=128, overlap_chars=0, max_passages=1)
    packet = copy.deepcopy(packet)
    packet["passages"][0].update({"start": 100, "end": 200})
    row, issues = baseline._parse_response(
        'SUBMIT: Blue CITATIONS: ["paper_a"] '
        'EVIDENCE: [{"doc_id":"paper_a","start":0,"end":26}]',
        packet, {"paper_a": setup["document"]},
    )
    assert row["predicted_answer"] == "Blue"
    assert issues == ["source span was not visible in the retrieved packet"]


def test_cli_requires_explicit_prepare_or_execute_mode() -> None:
    with pytest.raises(SystemExit) as raised:
        baseline.main(["--source-run", "unused", "--output", "unused", "--context-window", "8192"])
    assert raised.value.code == 2


def test_explicit_context_metadata_cannot_be_silently_changed(setup: dict) -> None:
    path = setup["source"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["models"][0]["serving_context_window_tokens"] = 4096
    write(path, manifest)
    with pytest.raises(ValueError, match="source model context window differs"):
        prepared(setup)
