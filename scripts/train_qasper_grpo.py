"""Bounded Qwen3 code-execution GRPO experiment; annotation-only supervision.

The legacy MuSiQue entry point and reward remain available. This entry point
shares the tested PPO update but records full episodes, a task-specific reward,
Qwen3 non-thinking prefixes, actual hardware, and immutable input identities.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.qasper_reward import REWARD_VERSION, score_submission
from src.policies.code_execution import QASPER_SYSTEM_PROMPT as PROMPT, clean_action, used_document_tool
from scripts import train_grpo_custom as grpo

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"



def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapter_fingerprint(model, adapter_name):
    """Hash one PEFT adapter's parameters without serializing the full model."""
    digest = hashlib.sha256()
    matched = 0
    marker = f".{adapter_name}."
    for name, parameter in sorted(model.named_parameters()):
        if marker not in name:
            continue
        matched += parameter.numel()
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    if not matched:
        raise ValueError(f"No parameters found for adapter {adapter_name!r}")
    return digest.hexdigest(), matched


def generation_kwargs(temperature, max_action_tokens, eos_token_id):
    settings = dict(max_new_tokens=max_action_tokens, do_sample=temperature>0,
                    use_cache=True, pad_token_id=eos_token_id,
                    eos_token_id=eos_token_id, top_k=0, top_p=1.0,
                    repetition_penalty=1.0)
    if temperature>0:
        settings["temperature"] = temperature
    # In Transformers 4.50+, globally default-valued settings can otherwise be
    # replaced by Qwen's saved sampling defaults, including do_sample=False.
    return {"generation_config": GenerationConfig(**settings), "use_model_defaults": False}




def collect(model, tokenizer, env, question_index, args, deadline):
    question = env.questions[question_index]
    initial = env.reset(question_idx=question_index)
    messages = [{"role":"system", "content":PROMPT}, {"role":"user", "content":initial}]
    steps, trace, investigated = [], [], False
    terminal, finish = "", "step_limit"
    started = time.monotonic()
    model.eval()
    for turn in range(args.max_steps):
        if time.monotonic() >= deadline:
            finish = "time_limit"
            break
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                 add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0]
        if ids.numel() > args.max_context:
            finish = "context_limit"
            break
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(ids.unsqueeze(0), attention_mask=torch.ones_like(ids).unsqueeze(0),
                                       **generation_kwargs(args.temperature, args.max_action_tokens,
                                                           tokenizer.eos_token_id))
        action_ids = generated[0, len(ids):].detach().cpu()
        if not action_ids.numel():
            finish = "empty_generation"
            break
        if args.mode == "train":
            old_lp = grpo._compute_token_log_probs(model, ids, action_ids, args.temperature)
            steps.append((ids.cpu(), action_ids, old_lp))
        raw = tokenizer.decode(action_ids, skip_special_tokens=True)
        action = clean_action(raw)
        observation, _, done, _ = env.step(action)
        trace.append({"action":action, "raw_action":raw, "observation":observation,
                      "done":done, "input_tokens":len(ids), "output_tokens":len(action_ids),
                      "hit_token_limit":len(action_ids)>=args.max_action_tokens and
                                        int(action_ids[-1]) != tokenizer.eos_token_id})
        if not action.upper().startswith("SUBMIT:"):
            investigated = investigated or used_document_tool(action, observation)
        if done:
            if action.upper().startswith("SUBMIT:"):
                terminal, finish = action, "submitted"
            break
        messages.extend([{"role":"assistant", "content":raw}, {"role":"user", "content":observation}])
    scoring = score_submission(terminal, question, env.corpus._documents, investigated=investigated)
    row = dict(question_id=question["id"], question=question["question"],
               expected_answerability=question["expected_answerability"],
               trajectory=trace, finish=finish, duration_seconds=time.monotonic()-started,
               **scoring)
    env.close()
    return steps, row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["eval", "probe", "train"], required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-action-tokens", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--advantage-normalization", choices=["std", "centered"], default="std")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.temperature = (0.0 if args.mode == "eval" else 1.0) if args.temperature is None else args.temperature
    if args.mode == "train" and (args.checkpoint is None or args.temperature<=0 or args.group_size<2):
        p.error("Training requires an SFT checkpoint, positive temperature, and group size >=2")
    if not 0 < args.max_seconds <= 12*3600:
        p.error("Each invocation must have a wall-time limit of at most 12 hours")
    if args.max_context < 1024 or min(args.max_steps,args.max_action_tokens,args.batch_size,args.updates)<1:
        p.error("Invalid token, step, batch or update limit")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    data = json.loads(args.benchmark.read_text())
    if args.mode == "train" and data["source_split"] != "train":
        p.error("Only official training questions may supply updates")
    if content_hash(args.corpus) != data["corpus_hash"]:
        p.error("Corpus differs from the frozen benchmark")
    questions = data["questions"][:args.limit or None]
    if not questions or any("answer_annotations" not in q for q in questions):
        p.error("Annotation-rich benchmark required")
    if args.checkpoint:
        config = json.loads((args.checkpoint/"adapter_config.json").read_text())
        if not config.get("base_model_name_or_path", "").endswith("Qwen3-8B"):
            p.error("Adapter does not identify Qwen3-8B as its base")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+args.max_seconds
    manifest = {"args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "base_model":BASE_MODEL, "base_revision":BASE_REVISION,
                "checkpoint_sha256": file_hash(args.checkpoint/"adapter_model.safetensors") if args.checkpoint else None,
                "benchmark_sha256":file_hash(args.benchmark), "corpus_hash":data["corpus_hash"],
                "question_ids":[q["id"] for q in questions], "system_prompt":PROMPT,
                "reward_version":REWARD_VERSION, "loss":"per-token PPO, fixed episode normalization",
                "advantage_normalization":args.advantage_normalization if args.mode=="train" else None,
                "ppo_epochs":1 if args.mode=="train" else None,
                "kl_beta":grpo.KL_BETA if args.mode=="train" else None,
                "kl_reference":"frozen starting SFT" if args.mode=="train" else None,
                "gradient_checkpointing":args.mode=="train", "thinking":False, "quantization":"NF4 double-quant",
                "generation_config":generation_kwargs(args.temperature,args.max_action_tokens,None)["generation_config"].to_dict(),
                "use_model_defaults":False,
                "hardware":torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                "source_sha256":{f:file_hash(f) for f in [__file__, "scripts/train_grpo_custom.py",
                                 "src/eval/qasper_reward.py", "src/env/document_env.py", "src/env/tools.py"]}}
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.out/"environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; no accidental CPU training")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    manifest["generation_config"] = generation_kwargs(args.temperature,args.max_action_tokens,
        tokenizer.eos_token_id)["generation_config"].to_dict()
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                quantization_config=grpo._bnb_config(), torch_dtype=torch.bfloat16,
                device_map={"":"cuda:0"}, attn_implementation="sdpa")
    if args.checkpoint:
        from peft import PeftModel, prepare_model_for_kbit_training
        if args.mode == "train":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                        gradient_checkpointing_kwargs={"use_reentrant":False})
        model = PeftModel.from_pretrained(model, str(args.checkpoint), is_trainable=args.mode=="train")
        if args.mode == "train":
            model.load_adapter(str(args.checkpoint), adapter_name=grpo.REF_ADAPTER, is_trainable=False)
            model.set_adapter(grpo.POLICY_ADAPTER)
            model.config.use_cache = False
            if getattr(model.config,"attention_dropout",0) != 0:
                raise ValueError("Nonzero attention dropout invalidates old-policy ratios")
    model.eval()
    grpo.MAX_CTX_TOKENS = args.max_context
    grpo.NORM_TOKENS = args.max_steps*args.max_action_tokens
    grpo.PPO_EPOCHS = 1
    corpus = Corpus(corpus_path=str(args.corpus))
    corpus.load(build_index=False)
    env = DocumentExplorationEnv(corpus,questions,max_steps=args.max_steps,use_docker=False,
                                 corpus_path=str(args.corpus),include_preamble=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable,lr=args.lr,weight_decay=0) if args.mode=="train" else None
    policy_start = reference_start = None
    if args.mode == "train":
        policy_start = adapter_fingerprint(model, grpo.POLICY_ADAPTER)
        reference_start = adapter_fingerprint(model, grpo.REF_ADAPTER)
        manifest["adapter_parameters"] = {
            "policy": policy_start[1], "reference": reference_start[1]
        }
        manifest["initial_adapter_sha256"] = {
            "policy": policy_start[0], "reference": reference_start[0]
        }
        (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    records, metrics, uniform_streak = [], [], 0
    stopped = "completed"
    def request_stop(*_):
        nonlocal deadline
        deadline = time.monotonic()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        with (args.out/"episodes.jsonl").open("w") as episodes, (args.out/"metrics.jsonl").open("w") as log:
            iterations = args.updates if args.mode=="train" else len(questions)
            for step in range(iterations):
                if time.monotonic()>=deadline:
                    stopped="time_limit"
                    break
                indices = random.sample(range(len(questions)), min(args.batch_size,len(questions))) if args.mode=="train" else [step]
                batch_data, batch_rewards = [], []
                tick = time.monotonic()
                for qi in indices:
                    group_data, rewards = [], []
                    repeats = 1 if args.mode=="eval" else args.group_size
                    for sample in range(repeats):
                        sd,row = collect(model,tokenizer,env,qi,args,deadline)
                        row.update(update=step,sample=sample)
                        episodes.write(json.dumps(row)+"\n"); episodes.flush()
                        records.append({k:v for k,v in row.items() if k!="trajectory"})
                        group_data.append(sd); rewards.append(row["reward"])
                        print(json.dumps({k:row[k] for k in ["update","sample","question_id","reward","reason","finish","duration_seconds"]}),flush=True)
                        if time.monotonic()>=deadline:
                            break
                    batch_data.append(group_data); batch_rewards.append(rewards)
                    if time.monotonic()>=deadline:
                        break
                if args.mode=="train":
                    if time.monotonic()>=deadline or len(batch_data)!=len(indices) or any(len(g)!=args.group_size or any(not sd for sd in g) for g in batch_data):
                        stopped="incomplete_group_at_deadline"
                        break
                    loss,kl,uniform,stats = grpo._grpo_update(model,optimizer,batch_data,batch_rewards,
                            step_num=step,temperature=args.temperature,kl_ref="sft",use_gradient_checkpointing=True,
                            advantage_normalization=args.advantage_normalization)
                    if not all(math.isfinite(v) for stat in stats for v in stat.values()):
                        raise RuntimeError("Nonfinite update diagnostics")
                    metric=dict(update=step+1,loss=loss,kl=kl,uniform_groups=uniform,
                                groups=len(indices),reward=sum(map(sum,batch_rewards))/(len(indices)*args.group_size),
                                seconds=time.monotonic()-tick,stats=stats,
                                peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
                    policy_now = adapter_fingerprint(model, grpo.POLICY_ADAPTER)
                    reference_now = adapter_fingerprint(model, grpo.REF_ADAPTER)
                    metric.update(
                        policy_adapter_sha256=policy_now[0],
                        policy_changed_from_start=policy_now[0] != policy_start[0],
                        reference_adapter_sha256=reference_now[0],
                        reference_unchanged=reference_now[0] == reference_start[0],
                    )
                    if not metric["reference_unchanged"]:
                        raise RuntimeError("Frozen SFT reference adapter changed during training")
                    metrics.append(metric); log.write(json.dumps(metric)+"\n"); log.flush()
                    print(json.dumps(metric),flush=True)
                    uniform_streak=uniform_streak+1 if uniform==len(indices) else 0
                    if (step+1)%args.save_every==0:
                        target=args.out/f"checkpoint-{step+1}"
                        model.save_pretrained(target,selected_adapters=[grpo.POLICY_ADAPTER]); tokenizer.save_pretrained(target)
                    if uniform_streak>=5:
                        stopped="five_updates_without_advantage_signal"
                        break
    except BaseException:
        stopped = "failed"
        raise
    finally:
        env.close()
        if args.mode=="train":
            model.set_adapter(grpo.POLICY_ADAPTER)
            model.save_pretrained(args.out/"final",selected_adapters=[grpo.POLICY_ADAPTER])
            tokenizer.save_pretrained(args.out/"final")
        summary=dict(status=stopped,updates=len(metrics),episodes=len(records),
                     elapsed_seconds=time.monotonic()-started,
                     mean_reward=sum(r["reward"] for r in records)/max(1,len(records)),
                     reasons=dict(Counter(r["reason"] for r in records)),
                     finishes=dict(Counter(r["finish"] for r in records)))
        (args.out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
