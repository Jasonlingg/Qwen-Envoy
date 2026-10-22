"""Terminal gaps remove a control turn, never source inspection or semantic review."""
import asyncio
import copy
import hashlib
import os
from types import SimpleNamespace

import pytest
from mcp import Client

from scripts.experiment_mode_mcp import create_server
from src.product.assigned_reader import build_assigned_reader
from src.product.packet_verifier import verify_investigation_packet
from src.product.passage_reader_service import PassageReaderService
from src.product.passage_scope import PASSAGES, PASSAGES_V2
from src.product.qwen_investigator import QwenInvestigator, effective_system_prompt
from src.product.reader_gaps import parse_gap_submission
from src.research.agent import load_snapshot
from tests.test_passage_reader import ID, ScopedREPL, fixture
from tests.test_qwen_investigator import FakePolicy

GAP = 'submit_gap(missing="Target p95 not established here", follow_up_query="target p95 hardware")'
ABSTAIN = 'submit(answer="Unanswerable", citations=[], evidence=[])'


def setup(tmp_path, action_builder=None, *, variant="default", docker=False):
    library, store, key, _, _, windows, first, submit, _ = fixture(tmp_path)
    start, end = windows[0]["start"], windows[0]["end"]
    second = f'print(passage("{ID}", {start + 1}, {end - start - 1}))'
    actions = (action_builder(first, second, submit) if action_builder
               else [first, second, GAP])
    policy = FakePolicy(actions)
    repl = ScopedREPL(library.document("2609.12345v2")["text"])
    image = os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE", "rlm-sandbox")
    reader = QwenInvestigator(lambda: policy, {"checkpoint": "scripted"}, max_steps=4,
        image=image, source_domain="public_papers", protocol_variant=variant,
        harness_version="v2", reader_mode="assigned", reader_guidance=PASSAGES_V2,
        repl_factory=None if docker else lambda _: repl)
    return library, store, key, windows, reader, policy, repl


def verify(reader, library, windows, packet):
    manifest, docs = load_snapshot(library.view(["2609.12345v2"]))
    return verify_investigation_packet(packet, manifest, docs,
        allowed_prompt_sha256s=frozenset({hashlib.sha256(reader.system_prompt.encode()).hexdigest()}),
        expected_reader_mode="assigned", expected_reader_guidance=PASSAGES_V2,
        expected_passage_windows=windows)


def test_two_reads_and_gap_finish_in_three_actions(tmp_path):
    library, _, _, windows, reader, _, repl = setup(tmp_path)
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    checked = verify(reader, library, windows, packet)
    assert packet["model_requests_attempted"] == 3
    assert packet["document_tool_steps"] == packet["inspection_steps"] == 2
    assert len(repl.executed) == 2  # Control call was not sent to Python.
    assert checked["status"] == "no_evidence" and checked["evidence"] == []
    assert checked["candidate_answer"] is None and checked["semantic_support"] == "not_reviewed"
    assert checked["research_gaps"][0]["absence_proven"] is False
    assert checked["research_gaps"][0]["review_status"] == "unreviewed_model_report"
    assert checked["research_gaps"][0]["reported_at_step"] == 3


@pytest.mark.parametrize("variant", ["default", "nemotron"])
def test_recorded_ordering_failure_recovers_on_last_turn(tmp_path, variant):
    library, _, _, windows, reader, policy, repl = setup(tmp_path,
        lambda first, second, _: [first, second, ABSTAIN, GAP], variant=variant)
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "no_evidence" and packet["model_requests_attempted"] == 4
    assert "submit_gap" in policy.observations[-1]
    assert "SUBMIT: Unanswerable" not in reader.system_prompt
    if variant == "nemotron":
        assert all("use SUBMIT: Unanswerable" not in p for p in policy.observations)
    assert len(repl.executed) == 2
    assert verify(reader, library, windows, packet)["research_gaps"][0]["reported_at_step"] == 4


@pytest.mark.parametrize("before_reads", [0, 1])
def test_premature_gap_does_not_record_or_replace_inspection(tmp_path, before_reads):
    def actions(first, second, _):
        reads = [first, second]
        return reads[:before_reads] + [GAP] + reads[before_reads:] + [GAP]
    library, _, _, windows, reader, _, repl = setup(tmp_path, actions)
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["trajectory"][before_reads]["observation"].startswith("Gap report rejected:")
    assert len(packet["research_gaps"]) == 1 and packet["research_gaps"][0]["reported_at_step"] == 4
    assert len(repl.executed) == 2
    assert verify(reader, library, windows, packet)["status"] == "no_evidence"


def test_duplicate_or_failed_reads_cannot_unlock_gap(tmp_path):
    library, _, _, windows, reader, _, repl = setup(tmp_path,
        lambda first, second, _: [first, first, GAP, GAP])
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "incomplete" and packet["research_gaps"] == []
    assert packet["inspection_steps"] == 1 and len(repl.executed) == 1
    library, _, _, windows, reader, _, repl = setup(tmp_path / "failed",
        lambda first, second, _: [first, second, GAP, GAP])
    original = repl.execute
    calls = 0
    def fail_first(code, **kwargs):
        nonlocal calls
        calls += 1
        return "RuntimeError: read failed" if calls == 1 else original(code, **kwargs)
    repl.execute = fail_first
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "incomplete" and packet["research_gaps"] == []
    assert packet["inspection_steps"] == 1


def test_separate_report_gets_one_action_recovery_without_storing_a_duplicate(tmp_path):
    library, _, _, windows, reader, policy, repl = setup(tmp_path,
        lambda first, second, _: [first, second, GAP.replace("submit_gap", "report_gap"), GAP])
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "no_evidence" and len(repl.executed) == 2
    assert "report the gap and finish in one action" in policy.observations[-1]
    assert len(verify(reader, library, windows, packet)["research_gaps"]) == 1


@pytest.mark.parametrize("action", [
    'submit_gap(missing=value, follow_up_query="q")',
    'submit_gap(missing=__import__("os").getcwd(), follow_up_query="q")',
    'submit_gap("m", "q"); print("side effect")',
    'print(submit_gap("m", "q"))',
    'if False:\n    submit_gap("m", "q")',
    'submit_gap(missing="m")',
    'submit_gap("m", "q", "extra")',
    'submit_gap("m", missing="duplicate", follow_up_query="q")',
    'submit_gap(**{"missing":"m", "follow_up_query":"q"})',
    'submit_gap(missing=True, follow_up_query="q")',
    'submit_gap(missing=" ", follow_up_query="q")',
    f'submit_gap(missing={"x" * 601!r}, follow_up_query="q")',
    f'submit_gap(missing="m", follow_up_query={"q" * 301!r})',
])
def test_terminal_gap_is_bounded_standalone_literal_data(action):
    with pytest.raises(ValueError):
        parse_gap_submission(action)


def test_unsafe_control_never_executes_and_later_evidence_survives(tmp_path):
    unsafe = 'submit_gap(missing=__import__("os").system("false"), follow_up_query="q")'
    library, _, _, windows, reader, _, repl = setup(tmp_path,
        lambda first, second, submit: [first, unsafe, submit])
    packet = reader.investigate("Memory?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    assert packet["status"] == "evidence_found" and len(repl.executed) == 1
    assert packet["research_gaps"] == []
    packet["research_gaps"] = None
    checked = verify(reader, library, windows, packet)
    assert checked["evidence"] and checked["gap_report_error"]


@pytest.mark.parametrize("tamper", ["missing_gap", "not_terminal", "legacy_gap", "duplicate_reads",
                                     "inspection_count", "document_count", "step", "after_finish"])
def test_verifier_rejects_forged_or_incomplete_terminal_gap(tmp_path, tamper):
    library, _, _, windows, reader, _, _ = setup(tmp_path)
    packet = reader.investigate("Target p95?", library.view(["2609.12345v2"]),
                                passage_windows=windows)
    packet = copy.deepcopy(packet)
    if tamper == "missing_gap":
        packet["research_gaps"] = []
    elif tamper == "not_terminal":
        packet["trajectory"][-1].pop("gap_submission")
    elif tamper == "legacy_gap":
        packet["trajectory"][-1]["action"] = GAP.replace("submit_gap", "report_gap")
    elif tamper == "duplicate_reads":
        packet["trajectory"][1] = copy.deepcopy(packet["trajectory"][0])
        packet["inspected_passage_windows"][1] = packet["inspected_passage_windows"][0]
    elif tamper == "inspection_count":
        packet["inspection_steps"] = 1
    elif tamper == "document_count":
        packet["document_tool_steps"] = 1
    elif tamper == "step":
        packet["research_gaps"][0]["reported_at_step"] = 2
    else:
        packet["trajectory"].append({"step": 4, "action": "print('after')"})
    with pytest.raises(ValueError):
        verify(reader, library, windows, packet)


@pytest.mark.parametrize("kind", ["qwen", "nemotron"])
def test_new_effective_prompt_matches_policy_and_legacy_prompt_is_pinned(monkeypatch, kind):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-only")
    reader = build_assigned_reader(kind, qwen_endpoint="http://scripted", qwen_model="fake",
        qwen_checkpoint="fake", reader_guidance=PASSAGES_V2)
    assert reader.policy_factory().system_prompt == reader.system_prompt
    legacy = effective_system_prompt("public_papers", "default", "v2",
                                    reader_mode="assigned", reader_guidance=PASSAGES)
    assert hashlib.sha256(legacy.encode()).hexdigest() == (
        "df1b6d39f21ead366471a4f90ab7566c59bb2061251c845d3aa5fb3774667641")
    assert reader.system_prompt != legacy


def test_optional_cli_selects_v2_without_model_requests(tmp_path, monkeypatch):
    from scripts import experiment_mode_mcp as cli
    from src.env.repl import PersistentREPL
    from src.policies import bounded_http
    from src.product import experiment_tokenizer

    monkeypatch.setattr(PersistentREPL, "_docker_available", lambda _: True)
    monkeypatch.setattr(bounded_http, "validate_endpoint", lambda _: None)
    monkeypatch.setattr(experiment_tokenizer, "load_tokenizer", lambda *_: object())
    def no_model_requests(*args, **kwargs):
        raise AssertionError("CLI setup must not dispatch inference")
    monkeypatch.setattr(bounded_http, "bounded_transport", no_model_requests)
    configured = []
    def server(*args, passage_reader, **kwargs):
        configured.append(passage_reader)
        return SimpleNamespace(run=lambda **_: None)
    monkeypatch.setattr(cli, "create_server", server)
    monkeypatch.setattr("sys.argv", ["experiment_mode_mcp.py",
        "--store-root", str(tmp_path / "records"), "--library-root", str(tmp_path / "library"),
        "--qwen-endpoint", "https://example.com/v1", "--qwen-model", "fake",
        "--qwen-checkpoint", "scripted"])
    cli.main()
    reader, budget = configured[0].reader_factory()
    assert reader.reader_guidance == PASSAGES_V2 and reader.max_steps == 4
    assert reader.policy_factory().system_prompt == reader.system_prompt
    assert budget.remaining == 4


@pytest.mark.parametrize("docker", [False, True])
def test_mcp_gap_saved_with_next_query_and_retry_does_not_redispatch(tmp_path, docker):
    if docker and not os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE"):
        pytest.skip("set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE for real Docker")
    library, store, key, windows, reader, policy, _ = setup(tmp_path,
        lambda first, second, _: [first, second, ABSTAIN, GAP], docker=docker)
    service = PassageReaderService(store, library, lambda: (reader, None), tmp_path / "reports",
                                   data_kind="synthetic")
    arguments = {"experiment_id": key, "expected_revision": 1, "note_id": "gap",
                 "question": "Target p95?", "arxiv_id": "2609.12345v2", "windows": windows}
    async def run():
        async with Client(create_server(store, library, None, passage_reader=service)) as client:
            response = await client.call_tool("read_experiment_passages", arguments)
            assert not response.is_error
            report = response.structured_content
            assert report["status"] == "no_evidence" and report["note_attached"]
            assert report["source_scope"]["version"] == PASSAGES_V2
            assert report["packet"]["model_requests_attempted"] == 4
            if docker:
                assert report["packet"]["execution"] == "docker_only"
            again = await client.call_tool("read_experiment_passages", arguments)
            assert again.structured_content == report and len(policy.observations) == 4
            resumed = await client.call_tool("resume_experiment", {"experiment_id": key})
            assert resumed.structured_content["latest_research_note_id"] == "gap"
        restarted = PassageReaderService(store, library, lambda: (reader, None),
                                         tmp_path / "reports", data_kind="synthetic")
        assert restarted.read(**arguments) == report and len(policy.observations) == 4
    asyncio.run(run())
    uncertainties = store.get(key)["selected_research_note"]["note"]["uncertainties"]
    assert "Target p95 not established here" in uncertainties
    assert any("target p95 hardware" in u and "not executed" in u for u in uncertainties)
