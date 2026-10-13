"""Reload a QASPER adapter and parse one next action using a training prompt.

Run only after the one-step optimizer smoke. This loads the pinned Qwen3-8B
base and the saved adapter on a GPU, never opens the frozen holdout, and checks
protocol syntax rather than tool-call execution or answer correctness.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_sft import BASE_MODEL, BASE_REVISION
from src.env.document_env import initial_observation
from src.eval.qasper_reward import parse_strict
from src.policies.code_execution import QASPER_SYSTEM_PROMPT, clean_action

TOOLS = {"search", "read", "passage", "extract", "search_within", "verify", "list_docs"}


def first_training_prompt(train_data: Path) -> list[dict]:
    """Return the exact first user turn from a replayed training conversation."""
    with train_data.open(encoding="utf-8") as stream:
        first = next((line for line in stream if line.strip()), None)
    if first is None:
        raise ValueError("Training JSONL is empty")
    messages = json.loads(first)["messages"]
    if (
        len(messages) < 3
        or messages[0] != {"role": "system", "content": QASPER_SYSTEM_PROMPT}
        or messages[1].get("role") != "user"
        or messages[2].get("role") != "assistant"
    ):
        raise ValueError("First training row does not contain the QASPER agent prompt")
    observation = messages[1]["content"]
    marker = "\n\nQuestion: "
    if marker not in observation or not observation.endswith("\n"):
        raise ValueError("First training row has no EnvoyBench initial observation")
    question = observation.rsplit(marker, 1)[1][:-1]
    if not question or observation != initial_observation(
        question, require_evidence=True, include_preamble=True
    ):
        raise ValueError("First training row differs from the evidence/preamble protocol")
    return messages[:2]


def classify_next_action(raw: str) -> tuple[str, str]:
    """Parse an unmodified Python tool call or strict evidence submission.

    A Python call can still use unsupported arguments; this is a reload and
    syntax smoke, not a substitute for executing a benchmark episode.
    """
    action = clean_action(raw)
    if not action or action != raw.strip():
        raise ValueError("Model output needed cleanup or contained no action")
    if action.upper().startswith("SUBMIT:"):
        parsed = parse_strict(action)
        if parsed is None:
            raise ValueError("Submission lacks strict CITATIONS/EVIDENCE syntax")
        answer, citations, evidence = parsed
        if set(citations) != {span.get("doc_id") for span in evidence}:
            raise ValueError("Submission citations and evidence disagree")
        if answer.casefold() == "unanswerable" and (citations or evidence):
            raise ValueError("Unanswerable submission must have empty citations and evidence")
        return "submit", action
    try:
        tree = ast.parse(action)
    except SyntaxError as exc:
        raise ValueError("Model output is not valid Python") from exc
    if not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in TOOLS
        for node in ast.walk(tree)
    ):
        raise ValueError("Python action does not call a document tool")
    return "python_tool", action


def smoke(adapter_dir: Path, train_data: Path, max_new_tokens: int = 1024) -> dict:
    if max_new_tokens < 1 or max_new_tokens > 1024:
        raise ValueError("max_new_tokens must be 1..1024")
    messages = first_training_prompt(train_data)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("Saved adapter config and safetensors are required")
    config = json.loads(config_path.read_text())
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError("Saved adapter was not trained on pinned Qwen3-8B base")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, quantization_config=quantization,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).eval()
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    kind, action = classify_next_action(text)
    return {
        "status": "parseable_next_action_unexecuted",
        "kind": kind,
        "action": action,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "adapter_dir": str(adapter_dir),
        "train_data": str(train_data),
        "thinking_enabled": False,
        "max_new_tokens": max_new_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(smoke(args.adapter_dir, args.train_data, args.max_new_tokens), indent=2))


if __name__ == "__main__":
    main()
