"""Small paired inference probe for saved Qwen evidence-card adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


BASE = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QUESTION = (
    "What evidence supports adding domain mixing to quality-filtered web data "
    "for language-model pretraining?"
)
SYSTEM = (
    "Extract a source-grounded evidence card from only the supplied paper passage. "
    "Return one JSON object with exactly these string keys: method, comparator, "
    "measured_outcome, limitation, quote. The quote must be one contiguous, exact "
    "substring of the passage. If the passage does not report a direct experiment "
    "on adding domain mixing to quality filtering, state that clearly in limitation. "
    "Use 'not reported in passage' for missing measures. Do not infer measurements."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(bundle: Path, output: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = json.loads((bundle / "cases.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, bundle / "v5", adapter_name="v5")
    model.load_adapter(bundle / "rl9", adapter_name="rl9")
    model.eval()
    result = {
        "base": BASE, "revision": REVISION, "seed": 42,
        "decoding": {"do_sample": False, "max_new_tokens": 320,
                     "enable_thinking": False},
        "input_sha256": sha(bundle / "cases.json"),
        "adapters": {name: sha(bundle / name / "adapter_model.safetensors")
                     for name in ("v5", "rl9")},
        "hardware": torch.cuda.get_device_name(0),
        "results": [],
    }
    torch.manual_seed(42)
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION + "\n\nPaper: " + case["title"]
             + "\nPassage:\n" + case["passage"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        for name in ("v5", "rl9"):
            model.set_adapter(name)
            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**tokens, do_sample=False,
                                           max_new_tokens=320,
                                           pad_token_id=tokenizer.eos_token_id)
            new_tokens = generated[0, tokens.input_ids.shape[1]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            try:
                card = json.loads(raw)
            except json.JSONDecodeError:
                card = None
            keys = {"method", "comparator", "measured_outcome", "limitation", "quote"}
            valid = isinstance(card, dict) and set(card) == keys and all(
                isinstance(value, str) for value in card.values())
            result["results"].append({
                "case_id": case["case_id"], "adapter": name,
                "elapsed_seconds": round(time.monotonic() - start, 2),
                "completion_tokens": len(new_tokens), "raw": raw,
                "schema_valid": valid,
                "quote_exact": valid and bool(card["quote"])
                and card["quote"] in case["passage"],
            })
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.bundle, args.output)
