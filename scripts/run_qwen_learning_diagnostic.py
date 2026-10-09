"""Supervised, bounded evidence-use and two-episode SFT memorization diagnostic."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train_qasper_grpo import (
    BASE_MODEL, BASE_REVISION, PROMPT, collect, file_hash, generation_kwargs,
)
from scripts.train_sft import _expand_per_action, _tokenize_per_action
from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.qasper_reward import score_submission
from src.policies.code_execution import clean_action


def control_cases(dev_benchmark, dev_episodes, train_benchmark, train_episodes):
    """Matched trailing passage/instruction, with and without previous history.

    These are oracle-evidence interventions, not autonomous retrieval scores.
    No gold answer string or evaluator answerability field enters the prompts.
    """
    cases = []
    terminal_instruction = "\n\nDiagnostic instruction: use the provided evidence to return your final SUBMIT line now."
    for data, episodes, row_index in [(dev_benchmark, dev_episodes, 0),
                                      (train_benchmark, train_episodes, 7)]:
        row = episodes[row_index]
        q = next(q for q in data["questions"] if q["id"] == row["question_id"])
        span = q["answer_annotations"][0]["evidence"][0]
        action = f'print(passage({json.dumps(span["doc_id"])}, start={span["start"]}, length={span["end"]-span["start"]}))'
        observation = str({k: span[k] for k in ("doc_id", "start", "end", "text")})
        prefix = [{"role": "system", "content": PROMPT},
                  {"role": "user", "content": f'Question: {q["question"]}\n'}]
        history = []
        for step in row["trajectory"][:-1]:
            history.extend([{"role": "assistant", "content": step["raw_action"]},
                            {"role": "user", "content": step["observation"]}])
        suffix = [{"role": "assistant", "content": action},
                  {"role": "user", "content": observation + terminal_instruction}]
        for condition, preceding in [("short_evidence", []), ("history_plus_same_evidence", history)]:
            cases.append({"question": q, "condition": condition,
                          "messages": prefix + preceding + suffix,
                          "oracle_evidence": True, "source_split": data["source_split"]})
    return cases


def batch_tensor(row, device):
    return {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in row.items()}


def main():
    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--dev-benchmark", type=Path, required=True)
    p.add_argument("--dev-corpus", type=Path, required=True)
    p.add_argument("--dev-episodes", type=Path, required=True)
    p.add_argument("--probe-episodes", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--max-seconds", type=int, default=5400)
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()
    if not 1 <= args.updates <= 60 or not 1 <= args.max_seconds <= 6000:
        p.error("This diagnostic is limited to 60 updates and 6000 seconds")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no accidental CPU training")
    start = time.monotonic()
    deadline = start + args.max_seconds
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Diagnostic wall-time limit reached")
    def stop(*_):
        raise TimeoutError("Diagnostic interrupted")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    train_data = json.loads((args.data / "benchmark.json").read_text())
    dev_data = json.loads(args.dev_benchmark.read_text())
    rows = [json.loads(s) for s in (args.data / "train.jsonl").read_text().splitlines()]
    assert train_data["source_split"] == "train"
    assert all(r["source_split"] == "train" for r in rows)
    assert {r["question_id"] for r in rows} == {q["id"] for q in train_data["questions"]}
    assert not ({q["target_doc_ids"][0] for q in train_data["questions"]} &
                {q["target_doc_ids"][0] for q in dev_data["questions"]})
    assert content_hash(args.corpus) == train_data["corpus_hash"]
    assert content_hash(args.dev_corpus) == dev_data["corpus_hash"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = _expand_per_action(rows)
    tokenized, token_stats = _tokenize_per_action(tokenizer, examples, 4096, "tiny-train")
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "starting_adapter_sha256": file_hash(args.checkpoint / "adapter_model.safetensors"),
        "training_sha256": file_hash(args.data / "train.jsonl"),
        "benchmark_sha256": file_hash(args.data / "benchmark.json"),
        "corpus_hash": train_data["corpus_hash"], "dev_corpus_hash": dev_data["corpus_hash"],
        "script_sha256": file_hash(__file__),
        "dependencies_sha256": {f: file_hash(f) for f in ["scripts/train_sft.py",
              "scripts/train_qasper_grpo.py", "src/env/document_env.py", "src/env/tools.py",
              "src/eval/qasper_reward.py"]},
        "question_ids": [r["question_id"] for r in rows], "tokenization": token_stats,
        "system_prompt": PROMPT, "seed": 42, "hardware": torch.cuda.get_device_name(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "lora": json.loads((args.checkpoint / "adapter_config.json").read_text()),
        "quantization": "NF4 double quantization, BF16 compute",
        "generation": generation_kwargs(0, 512, tokenizer.eos_token_id)["generation_config"].to_dict(),
        "use_model_defaults": False,
        "scope": "oracle evidence and training-set memorization; no generalization claim",
        "training": {"equal_action_weight": True, "dropout": 0, "weight_decay": 0,
                     "gradient_checkpointing": True, "max_updates": args.updates, "lr": args.lr},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(args.checkpoint), is_trainable=True)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    trainable = {n: param for n, param in model.named_parameters() if param.requires_grad}
    if not trainable or any("lora_" not in n for n in trainable):
        raise RuntimeError("Unexpected trainable parameter set")
    before = {n: param.detach().cpu().clone() for n, param in trainable.items()}
    device = next(model.parameters()).device
    corpus = Corpus(str(args.corpus)); corpus.load(build_index=False)
    dev_corpus = Corpus(str(args.dev_corpus)); dev_corpus.load(build_index=False)
    docs = {**corpus._documents, **dev_corpus._documents}
    log = (args.out / "events.jsonl").open("w", buffering=1)
    def event(kind, **value):
        record = {"kind": kind, "elapsed_seconds": time.monotonic()-start, **value}
        log.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in {"messages", "predictions", "trajectory"}}), flush=True)

    def generate(messages):
        check_time()
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                   enable_thinking=False, return_tensors="pt").to(device)
        if ids.shape[1] > 8192:
            return {"raw": "", "action": "", "input_tokens": ids.shape[1], "finish": "context_limit"}
        model.eval()
        with torch.no_grad():
            output = model.generate(ids, attention_mask=torch.ones_like(ids),
                 **generation_kwargs(0, 512, tokenizer.eos_token_id))
        tokens = output[0, ids.shape[1]:]
        raw = tokenizer.decode(tokens, skip_special_tokens=True)
        return {"raw": raw, "action": clean_action(raw), "input_tokens": ids.shape[1],
                "output_tokens": len(tokens), "finish": "token_limit" if len(tokens)>=512 and
                    tokens[-1].item()!=tokenizer.eos_token_id else "generated"}

    def fixed(step):
        model.eval()
        losses, accuracies, predictions = [], [], []
        for i, (row, example) in enumerate(zip(tokenized, examples)):
            check_time()
            batch = batch_tensor(row, device)
            with torch.no_grad():
                output = model(**batch, use_cache=False)
                labels = batch["labels"][:, 1:]
                mask = labels != -100
                accuracy = (output.logits[:, :-1][mask].argmax(-1) == labels[mask]).float().mean().item()
                losses.append(output.loss.item()); accuracies.append(accuracy)
            del output, batch
            pred = generate(example["prompt"])
            target = example["completion"][0]["content"]
            pred.update(index=i, target=target, exact_action_match=pred["action"].strip()==target.strip())
            if target.startswith("SUBMIT:"):
                qi = i // 3
                pred["score"] = score_submission(pred["action"], train_data["questions"][qi], docs, investigated=True)
            predictions.append(pred)
        result = {"update": step, "mean_action_loss": sum(losses)/len(losses),
                  "mean_token_accuracy": sum(accuracies)/len(accuracies),
                  "exact_actions": sum(p["exact_action_match"] for p in predictions),
                  "action_count": len(predictions), "predictions": predictions}
        (args.out / f"fixed-{step:03}.json").write_text(json.dumps(result, indent=2) + "\n")
        event("fixed_history", **result)
        return result

    def episodes(stage):
        env = DocumentExplorationEnv(corpus, train_data["questions"], max_steps=10,
            use_docker=False, corpus_path=str(args.corpus), include_preamble=False)
        cfg = SimpleNamespace(mode="eval", max_steps=10, max_context=8192, temperature=0,
                              max_action_tokens=512)
        try:
            with (args.out / f"episodes-{stage}.jsonl").open("w") as handle:
                for i in range(len(train_data["questions"])):
                    check_time()
                    _, row = collect(model, tokenizer, env, i, cfg, deadline)
                    handle.write(json.dumps(row) + "\n"); handle.flush()
                    event("autonomous_episode", stage=stage, **row)
        finally:
            env.close()

    step = 0
    try:
        event("loaded", trainable_parameters=sum(p.numel() for p in trainable.values()),
              adapter_status=str(model.get_model_status()), **token_stats)
        cases = control_cases(dev_data, [json.loads(s) for s in args.dev_episodes.read_text().splitlines()],
                              train_data, [json.loads(s) for s in args.probe_episodes.read_text().splitlines()])
        (args.out / "control-inputs.json").write_text(json.dumps(cases, indent=2) + "\n")
        for label in ("base", "starting_sft"):
            with model.disable_adapter() if label == "base" else nullcontext():
                for case in cases:
                    pred = generate(case["messages"])
                    event("evidence_control", model=label, question_id=case["question"]["id"],
                          condition=case["condition"], **pred,
                          score=score_submission(pred["action"], case["question"], docs, investigated=True))
        fixed(0)
        episodes("before")
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=args.lr, weight_decay=0)
        stop_reason = "update_limit"
        for step in range(1, args.updates + 1):
            check_time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for row in tokenized:
                check_time()
                output = model(**batch_tensor(row, device), use_cache=False)
                if not torch.isfinite(output.loss):
                    raise RuntimeError("Nonfinite training loss")
                losses.append(output.loss.item())
                (output.loss / len(tokenized)).backward()
                del output
            grad = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
            if float(grad) <= 0:
                raise RuntimeError("Zero gradient")
            optimizer.step()
            event("optimizer_update", update=step, mean_action_loss=sum(losses)/len(losses),
                  gradient_norm=float(grad), max_gpu_bytes=torch.cuda.max_memory_allocated())
            if step in {5, 10, 20, 40, 60, args.updates}:
                check = fixed(step)
                model.save_pretrained(args.out / f"checkpoint-{step}")
                if check["exact_actions"] == len(examples) and check["mean_action_loss"] < 0.05:
                    stop_reason = "memorization_gate_passed"
                    break
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        changed = sum(not torch.equal(before[n], param.detach().cpu()) for n, param in trainable.items())
        if not changed:
            raise RuntimeError("Adapter weights did not change")
        final = args.out / "final"
        model.save_pretrained(final); tokenizer.save_pretrained(final)
        saved_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
        del trainable, before
        base = model.unload()
        model = PeftModel.from_pretrained(base, str(final), is_trainable=False)
        reloaded_state = get_peft_model_state_dict(model)
        reload_equal = all(torch.equal(value, reloaded_state[key].cpu()) for key, value in saved_state.items())
        if not reload_equal:
            raise RuntimeError("Saved/reloaded adapter weights differ")
        event("reload_verified", changed_tensors=changed, adapter_tensors=len(saved_state),
              identical_tensors=reload_equal, final_adapter_sha256=file_hash(final / "adapter_model.safetensors"))
        episodes("after-reload")
        event("completed", updates=step, stop_reason=stop_reason)
        (args.out / "completion.json").write_text(json.dumps({"status": "completed", "updates": step,
                   "stop_reason": stop_reason, "reload_equal": reload_equal,
                   "duration_seconds": time.monotonic()-start}, indent=2) + "\n")
    except BaseException as exc:
        event("failed", updates=step, error=repr(exc))
        if step:
            model.save_pretrained(args.out / "interrupted")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
