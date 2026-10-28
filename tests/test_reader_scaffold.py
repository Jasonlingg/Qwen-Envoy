"""Offline experiment separation, pairing, leakage and launch-boundary checks."""
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import reader_scaffold as ablation
from src.eval.passage_pilot_replay import PROMPT, PROMPT_SHA
from src.training.prime_run import write


def test_selection_is_training_only_and_same_as_predeclared_seeded_order():
    _, cases, _, _, _ = ablation.selection()
    _, roles, _, _, ids, _ = ablation.load_training_bundle(ablation.BUNDLE,
                                                        ablation.TRAINING_REVISION)
    ablation.random.Random(42).shuffle(ids)
    assert [c["case_id"] for c in cases] == ids[:6]
    assert all(c["role"] == "train" and c["source_split"] == "train" for c in cases)
    assert not {c["doc_id"] for c in cases} & {
        c["doc_id"] for role in ("reward_validation", "model_evaluation_locked")
        for c in roles[role]}


def test_schedule_pairs_seeds_without_changing_model_or_oversampling_questions():
    _, cases, _, _, _ = ablation.selection()
    rows = list(ablation.schedule(cases))
    assert len(rows) == 48
    for offset in range(0, len(rows), 2):
        left, right = rows[offset:offset + 2]
        assert left[:2] == right[:2] and left[3] == right[3]
        assert {left[2], right[2]} == {"control", "checklist"}
    assert [r[2] for r in rows[:4]] == ["control", "checklist", "checklist", "control"]


def test_prompt_and_export_preserve_control_without_exporting_reference_answers(tmp_path):
    plan = ablation.prepare(tmp_path / "prepared")
    assert ablation.PROMPTS["control"] == PROMPT
    assert ablation.sha(PROMPT) == PROMPT_SHA
    assert ablation.PROMPTS["checklist"] == PROMPT + ablation.CHECKLIST
    assert plan["optimizer_updates"] == 0 and plan["training_eligible"] is False
    assert plan["limits"]["requests"] == 48 * 12
    assert plan["limits"]["reserved_tokens"] == 48 * 12 * 8192
    assert all(set(t) == {"case_id", "question_id", "question", "doc_id", "scope", "snapshot"}
               for t in plan["tasks"])
    exported = json.dumps(plan)
    assert '"accepted_answers"' not in exported and '"supporting_spans"' not in exported
    assert ablation.validate(tmp_path / "prepared")[0] == plan


@pytest.mark.parametrize("mutation", ["budget", "tokens", "seed", "question", "weights",
                                     "prompt_hash", "helper", "schedule", "prompt_file"])
def test_drift_requires_fresh_preparation(tmp_path, mutation):
    output = tmp_path / "prepared"
    plan = copy.deepcopy(ablation.prepare(output))
    if mutation == "budget":
        plan["budget_proposal"]["maximum_owned_seconds"] += 1
    elif mutation == "tokens":
        plan["limits"]["reserved_tokens"] += 1
    elif mutation == "seed":
        plan["schedule"][0]["seed"] += 1
    elif mutation == "question":
        plan["tasks"][0]["question"] += " gold answer"
    elif mutation == "weights":
        plan["adapter_sha256"] = "changed"
    elif mutation == "prompt_hash":
        plan["prompt_sha256"]["checklist"] = "changed"
    elif mutation == "helper":
        plan["dependencies_sha256"]["src/training/reader_data.py"] = "changed"
    elif mutation == "schedule":
        plan["schedule"].reverse()
    else:
        (output / "checklist.txt").write_text("changed")
    write(output / "plan.json", plan)
    with pytest.raises(ValueError, match="changed"):
        ablation.validate(output)


def test_unapproved_live_run_fails_before_importing_native_stack_or_contacting_engine(tmp_path):
    output = tmp_path / "prepared"
    ablation.prepare(output)
    approval = tmp_path / "approval.json"
    write(approval, {"approved": False})
    with pytest.raises(ValueError, match="approval"):
        ablation.run(output, "http://127.0.0.1:1", None, None, approval)
    assert not (output / "started.json").exists()


def test_initial_prompt_matches_real_policy_prefix_including_duplicated_system_text():
    _, cases, _, _, _ = ablation.selection()
    for prompt in ablation.PROMPTS.values():
        messages = ablation.initial_messages(cases[0], prompt)
        assert messages[0] == {"role": "system", "content": prompt}
        assert messages[1]["content"].startswith(f"{prompt}\nQuestion: {cases[0]['question']}\n")
        assert "Controller passage ranges (metadata only): " in messages[1]["content"]


def test_diagnostics_distinguish_solved_groups_from_useful_variance_and_partial_groups():
    rows = []
    for cid, rewards in (("solved", [1, 1, 1, 1]), ("mixed", [0, 1, 0, 1]),
                         ("failed", [0, 0, 0, 0]), ("partial", [0, 1])):
        rows.extend({"arm": "checklist", "case_id": cid, "reward": {"reward": r},
                     "usage": [{"prompt_tokens": 100}]} for r in rewards)
    result = ablation.diagnostics(rows)["checklist"]
    assert result["complete_groups"] == 3 and result["mixed_groups"] == 1
    assert result["uniform_one_groups"] == result["uniform_zero_groups"] == 1
    assert result["reserved_tokens_for_recorded_healthy_episodes"] == 14 * 1124


@pytest.mark.parametrize("fail", [False, True])
def test_real_native_renderer_uses_budget_proxy_and_does_not_retry_provider_failure(tmp_path, fail):
    pytest.importorskip("renderers")
    from renderers import create_renderer
    from renderers.configs import DefaultRendererConfig
    from transformers import AutoTokenizer

    if not ablation.TOKENIZER.is_dir():
        pytest.skip("requires locally pinned tokenizer; no downloads")
    tokenizer = AutoTokenizer.from_pretrained(ablation.TOKENIZER, local_files_only=True)
    renderer = create_renderer(tokenizer, DefaultRendererConfig(enable_thinking=False))
    completion = tokenizer.encode("print('ok')", add_special_tokens=False)
    calls = []

    class Engine(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"token_ids": completion, "finish_reason": "stop",
                "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": -0.1}
                                         for t in completion]}}]}).encode()
            self.send_response(500 if fail else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    write(tmp_path / "started.json", {"deadline": time.time() + 30})
    plan = {"limits": {"requests": 1, "reserved_tokens": 8192}}
    try:
        with ablation.proxy(tmp_path, plan, {},
                            engine=f"http://127.0.0.1:{server.server_port}") as url:
            transport = ablation.NativeTransport(url, renderer, 42)
            payload = {"messages": [{"role": "user", "content": "Write an action"}]}
            if fail:
                with pytest.raises(Exception):
                    transport("", payload, {}, 30)
            else:
                response = transport("", payload, {}, 30)
                assert response["choices"][0]["message"]["content"] == "print('ok')"
                assert response["usage"]["completion_tokens"] == len(completion)
        ledger = ablation.load(tmp_path / "dispatches.json")
        assert len(calls) == len(ledger["requests"]) == 1
        assert calls[0]["sampling_params"]["seed"] == 42
        assert calls[0]["sampling_params"]["temperature"] == 1.0
        assert ledger["reserved_tokens"] == len(calls[0]["token_ids"]) + 1024
        assert ledger["closed"] is fail
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
