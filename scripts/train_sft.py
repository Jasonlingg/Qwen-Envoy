"""QLoRA SFT on code-execution trajectories.

Requires the [training] extra:
  pip install -e ".[training]"

Check the published v5 split without loading a model:
  python scripts/train_sft.py check-data --data data/sft/qasper-v5/train.jsonl \
      --val-data data/sft/qasper-v5/val.jsonl \
      --corpus-manifest data/sft/qasper-v5/manifest.json

Training requires a GPU, the [training] extra, and a new output directory.
The frozen September 2026 training record is in release/qwen-v5-training/;
this current script is an inspectable recipe, not its byte-identical source.
"""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from src.eval.hashing import sha256 as _sha256
from src.policies.code_execution import SYSTEM_PROMPT

console = Console()
app = typer.Typer()

BASE_MODEL = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

# r=4, not 16: v5 has 106 short training-action targets and 4,812 supervised
# tokens. This is a small dataset; rank 4 limits adapter capacity, but it does
# not by itself prevent memorisation or establish that this rank is optimal.
# alpha tracks r to hold the alpha/r update scale at 2 — leaving alpha at 32
# while dropping r would quadruple the effective step size on top of the LR.
LORA_CONFIG = {
    "r": 4,
    "lora_alpha": 8,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

_ASSIGNED_DOC = re.compile(r'doc_id:\s*"([^"]+)"')


def _load_conversations(path: Path, split: str) -> list[dict]:
    """Read complete alternating agent turns before any model/GPU allocation."""
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{split} line {line_number}: invalid JSON: {exc.msg}") from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
            raise ValueError(f"{split} line {line_number}: need system, user, assistant turns")
        for index, message in enumerate(messages):
            role = "system" if index == 0 else ("user" if index % 2 else "assistant")
            if (not isinstance(message, dict) or message.get("role") != role
                    or not isinstance(message.get("content"), str)
                    or not message["content"].strip()):
                raise ValueError(
                    f"{split} line {line_number}, message {index}: expected nonempty {role}"
                )
        rows.append(row)
    if not rows:
        raise ValueError(f"{split}: no conversations")
    if len({row["messages"][1]["content"] for row in rows}) != len(rows):
        raise ValueError(f"{split}: duplicate initial questions")
    if len({row["messages"][0]["content"] for row in rows}) != 1:
        raise ValueError(f"{split}: system prompts differ")
    return rows


def _check_splits(train_rows: list[dict], val_rows: list[dict] | None) -> None:
    if val_rows is None:
        return
    if train_rows[0]["messages"][0]["content"] != val_rows[0]["messages"][0]["content"]:
        raise ValueError("train and validation system prompts differ")
    train_questions = {row["messages"][1]["content"] for row in train_rows}
    val_questions = {row["messages"][1]["content"] for row in val_rows}
    if train_questions & val_questions:
        raise ValueError("train and validation share an initial question")
    all_questions = train_questions | val_questions
    doc_matches = [_ASSIGNED_DOC.search(question) for question in all_questions]
    if any(doc_matches):
        if not all(doc_matches):
            raise ValueError("assigned-paper data has a missing doc_id")
        train_docs = {_ASSIGNED_DOC.search(q).group(1) for q in train_questions}
        val_docs = {_ASSIGNED_DOC.search(q).group(1) for q in val_questions}
        if train_docs & val_docs:
            raise ValueError("train and validation share an assigned paper")


def _check_declared_manifest(
    path: Path | None, data: Path, val_data: Path | None, train_rows: list[dict],
) -> None:
    """Check pinned v5 split hashes and prompt when a split manifest is supplied."""
    if path is None:
        return
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("corpus manifest must be a JSON object")
    splits = manifest.get("splits")
    if splits is not None:
        if not isinstance(splits, dict) or "train" not in splits:
            raise ValueError("split manifest needs a train entry")
        for name, file_path in (("train", data), ("val", val_data)):
            if file_path is None:
                continue
            declared = splits.get(name)
            if not isinstance(declared, dict) or declared.get("sha256") != _sha256(file_path):
                raise ValueError(f"{name} file does not match split manifest hash")
    prompt_hash = manifest.get("system_prompt_sha256")
    if prompt_hash is not None:
        import hashlib
        actual = hashlib.sha256(train_rows[0]["messages"][0]["content"].encode()).hexdigest()
        if prompt_hash != actual:
            raise ValueError("system prompt does not match manifest hash")


def _checked_data(data: Path, val_data: Path | None, corpus_manifest: Path | None):
    rows = _load_conversations(data, "train")
    val_rows = _load_conversations(val_data, "validation") if val_data is not None else None
    _check_splits(rows, val_rows)
    _check_declared_manifest(corpus_manifest, data, val_data, rows)
    return rows, val_rows


@app.command("check-data")
def check_data(
    data: Path = typer.Option(..., "--data"),
    val_data: Path | None = typer.Option(None, "--val-data"),
    corpus_manifest: Path | None = typer.Option(None, "--corpus-manifest"),
) -> None:
    """Check conversation structure and declared split identity without GPU/model loading."""
    rows, val_rows = _checked_data(data, val_data, corpus_manifest)
    console.print(
        f"Validated {len(rows)} train and {len(val_rows) if val_rows is not None else 0} "
        "validation conversations; no shared questions or assigned papers."
    )


def _expand_per_action(rows: list[dict]) -> list[dict]:
    """Turn each trajectory into one next-action example per assistant turn.

    Qwen3's non-thinking generation prefix is present only for the assistant turn
    being generated. A whole-conversation SFT row therefore gives intermediate
    actions a different prefix from inference. Splitting by action preserves the
    complete preceding history while making every target the final assistant turn.
    """
    examples: list[dict] = []
    for row_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"row {row_index} has no conversational 'messages' list")
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            prompt = messages[:message_index]
            if not prompt or prompt[-1].get("role") != "user":
                raise ValueError(
                    f"row {row_index}, message {message_index}: assistant target must "
                    "immediately follow a user observation"
                )
            examples.append({"prompt": prompt, "completion": [message]})
    if not examples:
        raise ValueError("dataset contains no assistant actions")
    return examples


def _tokenize_per_action(
    tokenizer, examples: list[dict], max_seq_len: int, split_name: str
) -> tuple[list[dict], dict]:
    """Create labels only for the next action and prove prefix equality.

    Building labels here avoids relying on trainer-version-specific chat-template
    preprocessing. The exact token prefix used for training must equal the prompt
    produced by Qwen3 at inference with thinking disabled.
    """
    tokenized: list[dict] = []
    lengths: list[int] = []
    supervised_tokens = 0

    def input_ids(rendered) -> list[int]:
        # transformers 4.x returns a list here; 5.x may return a
        # BatchEncoding when the template contains generation markers.
        return rendered["input_ids"] if hasattr(rendered, "keys") else rendered

    for row_index, example in enumerate(examples):
        prompt = example["prompt"]
        completion = example["completion"]
        prompt_ids = input_ids(tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
        full_ids = input_ids(tokenizer.apply_chat_template(
            prompt + completion,
            tokenize=True,
            enable_thinking=False,
        ))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{split_name} action {row_index}: training tokens do not start with "
                "the inference generation prefix"
            )
        if len(full_ids) > max_seq_len:
            raise ValueError(
                f"{split_name} action {row_index}: {len(full_ids)} tokens exceeds "
                f"--max-seq-len {max_seq_len}; refusing silent target truncation"
            )
        target_tokens = len(full_ids) - len(prompt_ids)
        if target_tokens <= 0:
            raise ValueError(f"{split_name} action {row_index}: empty supervised target")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        tokenized.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        })
        lengths.append(len(full_ids))
        supervised_tokens += target_tokens

    return tokenized, {
        "actions": len(tokenized),
        "input_tokens": sum(lengths),
        "supervised_tokens": supervised_tokens,
        "max_tokens": max(lengths),
    }


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def _load_training_deps():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
        return (
            torch, Dataset, LoraConfig, PeftModel, AutoModelForCausalLM,
            AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer,
        )
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Run: pip install -e '.[training]'")
        sys.exit(1)


@app.command()
def train(
    data: Path = typer.Option(..., "--data", "-d", help="Conversation JSONL"),
    val_data: Path = typer.Option(None, "--val-data",
        help="Held-out JSONL. Eval loss diverging from train loss means memorisation."),
    out: Path = typer.Option(Path("checkpoints/sft_qwen3_8b"), "--out", "-o"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
    # LoRA needs a markedly higher LR than full fine-tuning; 2e-5 is a
    # full-fine-tune value and barely moves a rank-4 adapter.
    lr: float = typer.Option(2e-4, "--lr"),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(4, "--grad-accum"),
    max_seq_len: int = typer.Option(8192, "--max-seq-len"),
    base_model: str = typer.Option(BASE_MODEL, "--base-model"),
    base_revision: str = typer.Option(BASE_REVISION, "--base-revision"),
    adapter_checkpoint: str = typer.Option(
        None,
        "--adapter-checkpoint",
        help="Continue training an existing LoRA adapter instead of creating a new one",
    ),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    activation_offloading: bool = typer.Option(
        False,
        "--activation-offloading/--no-activation-offloading",
        help="Move saved forward activations to CPU to fit long sequences",
    ),
    per_action: bool = typer.Option(
        True,
        "--per-action/--full-conversation",
        help="Supervise each next action with the exact inference prefix",
    ),
    max_steps: int = typer.Option(-1, "--max-steps"),
    save_steps: int = typer.Option(
        0, "--save-steps", help="Save every N optimizer steps; 0 saves per epoch"
    ),
    save_total_limit: int = typer.Option(10, "--save-total-limit"),
    seed: int = typer.Option(42, "--seed"),
    corpus_manifest: Path = typer.Option(
        None, "--corpus-manifest", help="Source-corpus manifest recorded with the run"
    ),
) -> None:
    if (epochs <= 0 or batch_size <= 0 or grad_accum <= 0 or max_seq_len <= 0
            or save_steps < 0 or save_total_limit <= 0 or seed < 0
            or max_steps != -1 and max_steps <= 0
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError("invalid training budget, sequence length, seed, or learning rate")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"output directory is not empty: {out}")
    rows, val_rows = _checked_data(data, val_data, corpus_manifest)
    console.print(f"Loaded {len(rows)} conversations from {data}")
    if val_rows is not None:
        console.print(f"Loaded {len(val_rows)} held-out conversations from {val_data}")

    (
        torch, hf_dataset, lora_config_cls, peft_model_cls, auto_model_cls,
        auto_tokenizer_cls, bnb_config_cls, sft_config_cls, sft_trainer_cls,
    ) = _load_training_deps()

    # Tokenizer
    tokenizer = auto_tokenizer_cls.from_pretrained(
        base_model, revision=base_revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3's stock template has no {% generation %} block, so assistant_only_loss
    # would silently mask nothing and train on garbage. Raises if it can't verify.
    if "qwen3" in base_model.lower():
        from src.policies.qwen3_chat_template import patch_tokenizer_for_assistant_masking

        patch_tokenizer_for_assistant_masking(tokenizer)
        console.print("[green]Patched Qwen3 chat template for assistant-only loss[/green]")

    train_stats = {"conversations": len(rows)}
    val_stats = {"conversations": len(val_rows)} if val_rows is not None else None
    if per_action:
        action_rows = _expand_per_action(rows)
        tokenized_rows, action_stats = _tokenize_per_action(
            tokenizer, action_rows, max_seq_len, "train"
        )
        train_stats.update(action_stats)
        dataset = hf_dataset.from_list(tokenized_rows)
        eval_dataset = None
        if val_rows is not None:
            val_action_rows = _expand_per_action(val_rows)
            tokenized_val, val_action_stats = _tokenize_per_action(
                tokenizer, val_action_rows, max_seq_len, "validation"
            )
            val_stats.update(val_action_stats)
            eval_dataset = hf_dataset.from_list(tokenized_val)
        console.print(
            f"[green]Verified exact inference-prefix alignment for "
            f"{train_stats['actions']} training actions[/green]"
        )
    else:
        if "qwen3" in base_model.lower() and any(
            sum(m.get("role") == "assistant" for m in row.get("messages", [])) > 1
            for row in rows
        ):
            raise ValueError(
                "Qwen3 full-conversation training misaligns intermediate action prefixes. "
                "Use the default --per-action mode."
            )
        dataset = hf_dataset.from_list(rows)
        eval_dataset = hf_dataset.from_list(val_rows) if val_rows is not None else None

    # Model
    bnb_config = None
    if load_in_4bit:
        bnb_config = bnb_config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = auto_model_cls.from_pretrained(
        base_model,
        revision=base_revision,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if not load_in_4bit else None,
    )

    if adapter_checkpoint:
        model = peft_model_cls.from_pretrained(
            model, adapter_checkpoint, is_trainable=True
        )
        lora_config = None
        console.print(
            f"[green]Continuing trainable adapter from {adapter_checkpoint}[/green]"
        )
    else:
        lora_config = lora_config_cls(**LORA_CONFIG)

    out_dir = str(out)
    final_dir = str(out / "final")

    sft_config = sft_config_cls(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        # Must be set explicitly: HF defaults eval batch to 8, which OOMs on a
        # 24GB card once sequences approach the 8k limit even though training at
        # batch 1 fits comfortably.
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        # Per-epoch, not save_steps: a short run can finish in fewer steps than
        # save_steps, in which case no intermediate checkpoint is ever written and
        # a crash loses everything. Saving per epoch also makes the epoch-1
        # checkpoint available if epoch-2 eval loss shows memorisation.
        save_strategy="steps" if save_steps > 0 else "epoch",
        save_steps=save_steps if save_steps > 0 else 500,
        save_total_limit=save_total_limit,
        packing=False,
        max_length=max_seq_len,
        report_to="none",
        gradient_checkpointing=True,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        # Without this, loss is computed over the entire sequence — including the
        # long tool-output text (search results, REPL prints) in user turns, which
        # the model never needs to generate. On our data only ~20% of tokens are
        # assistant tokens, so this is the difference between training on the
        # trajectory and training mostly on search output. Qwen3 needs the
        # chat-template patch applied above for this to mask anything at all.
        # Per-action rows are pre-tokenized above with explicit -100 prompt
        # labels. Conversational rows use the generation markers in the patched
        # template instead.
        assistant_only_loss=not per_action,
        completion_only_loss=False if per_action else None,
        max_steps=max_steps,
        seed=seed,
        data_seed=seed,
        activation_offloading=activation_offloading,
    )

    tokenizer.model_max_length = max_seq_len

    trainer = sft_trainer_cls(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "qwen-sft-run-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "base_revision": base_revision,
        "adapter_checkpoint": adapter_checkpoint,
        "training_format": "per-action-prefix-aligned" if per_action else "conversation",
        "data": {"path": str(data), "sha256": _sha256(data), **train_stats},
        "validation": (
            {"path": str(val_data), "sha256": _sha256(val_data), **val_stats}
            if val_data is not None and val_stats is not None else None
        ),
        "corpus_manifest": (
            {"path": str(corpus_manifest), "sha256": _sha256(corpus_manifest)}
            if corpus_manifest is not None else None
        ),
        "lora": LORA_CONFIG if adapter_checkpoint is None else None,
        "training_script_sha256": _sha256(Path(__file__)),
        "training": {
            "epochs": epochs, "max_steps": max_steps, "learning_rate": lr,
            "batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
            "max_sequence_length": max_seq_len, "load_in_4bit": load_in_4bit,
            "activation_offloading": activation_offloading,
            "save_steps": save_steps, "save_total_limit": save_total_limit,
            "seed": seed,
        },
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status": _git_output("status", "--porcelain").splitlines(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    console.print(
        f"[bold]Training on {len(dataset)} action examples from {len(rows)} conversations, "
        f"{epochs} epoch(s)[/bold]"
    )
    trainer.train()
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"[green]Saved to {final_dir}[/green]")


def _sanity_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]


@app.command()
def sanity_check(
    model: Path = typer.Option(..., "--model", "-m", help="Path to saved LoRA checkpoint"),
    question: str = typer.Option(
        "Where was the CEO of Apex Corp born?",
        "--question", "-q",
    ),
    max_new_tokens: int = typer.Option(512, "--max-tokens"),
    base_model: str = typer.Option(BASE_MODEL, "--base-model"),
    base_revision: str = typer.Option(BASE_REVISION, "--base-revision"),
) -> None:
    """Greedy-decode one question with the non-thinking training prompt."""
    if max_new_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    (
        torch, _, _, peft_model_cls, auto_model_cls, auto_tokenizer_cls, _, _, _,
    ) = _load_training_deps()

    tokenizer = auto_tokenizer_cls.from_pretrained(
        base_model, revision=base_revision, trust_remote_code=True
    )
    base = auto_model_cls.from_pretrained(
        base_model, revision=base_revision, device_map="auto", dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    peft_model = peft_model_cls.from_pretrained(base, str(model))
    peft_model.eval()

    messages = _sanity_messages(question)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(peft_model.device)

    with torch.no_grad():
        out = peft_model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"\n[bold]Model output:[/bold]\n{response}")

    if response.strip().upper().startswith("SUBMIT:") or any(
        kw in response for kw in ["search(", "read(", "extract(", "search_within("]
    ):
        console.print("\n[green]✓ Output looks like Python code or SUBMIT — format correct[/green]")
    else:
        console.print("\n[red]✗ Output looks like prose — check chat template or data[/red]")


if __name__ == "__main__":
    app()
