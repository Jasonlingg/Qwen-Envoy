"""Run a Claude model through the same Envoy QASPER protocol used for the Qwen evaluations.

Mirrors `collect()` in scripts/train_qasper_grpo.py (eval mode) so the numbers are
comparable with the Qwen SFT/GRPO episode files: question-only first observation,
ten-step budget, greedy decoding request, same tools and default search depth, same
`qasper-answer-evidence-v1` scorer, and the Claude-specific action parser built for
this harness (src/policies/claude_action_cleaning.py). Differences that cannot be
removed and are recorded in the manifest: Claude is not run with Qwen's non-thinking
template (each model's provider default applies), it gets a larger per-action token
allowance (thinking tokens count against `max_tokens`), it has no 8,192-token context
cap, and its system prompt is QASPER_SYSTEM_PROMPT plus three worked few-shot turns
(FEWSHOT_EXAMPLES below) that Qwen never saw at training time as literal text — Qwen
instead learned the same turn-taking shape from ~190 real SFT trajectories. Without
these examples Claude has to infer the format zero-shot from prose alone, which
produced near-total format collapse in an earlier run of this script (Haiku
hallucinated XML tool-call syntax on 35-36 of 40 episodes).

Spend is hard-capped by a thread-safe ledger that reserves a worst-case cost before every
request and settles it from the provider's reported usage. Prices (USD per million tokens)
were read from the Claude pricing page on 2026-09-21.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.qasper_reward import REWARD_VERSION, score_submission
from src.policies.claude_action_cleaning import clean_action
from src.policies.claude_policy import ClaudePolicy
from src.policies.code_execution import QASPER_SYSTEM_PROMPT, used_document_tool

# Qwen learned this protocol from ~190 real SFT trajectories. Claude sees it zero-shot
# from a prose description with zero worked examples — a plausible root cause of format
# failures (Haiku hallucinating XML tool-call syntax; Sonnet emitting one action per
# response but sometimes indenting it as if narrating). These three turns exist only to
# show the SHAPE of a turn (one action, then wait for real output) and are not meant to
# bias the answer content. QASPER_SYSTEM_PROMPT itself is untouched — it must stay
# byte-for-byte identical to what Qwen trained on for that comparison to remain valid.
FEWSHOT_EXAMPLES = '''

Three example turns, showing the shape of a response (ONE action, then you see its real
output before deciding the next action — never write more than one step per response):

Turn 1 (a plain search call, nothing else):
r = search_within("qasper_1234_56789", "training dataset size")
for w in r:
    print(w["text"][:300])

Turn 2, after seeing that output (reading a passage found in step 1):
p = passage("qasper_1234_56789", start=4200, length=800)
print(p["text"])

Turn 3, once the evidence is sufficient (a submission, nothing before or after it):
SUBMIT: The dataset contains 3,209 reviews. CITATIONS: ["qasper_1234_56789"] EVIDENCE: [{"doc_id":"qasper_1234_56789","start":4231,"end":4270}]

EVERY SUBMIT line must end with an EVIDENCE: clause, no exceptions — even for
Unanswerable, write EVIDENCE: []. A submission missing EVIDENCE: is rejected as
invalid and scores zero, regardless of whether the answer itself was correct.
'''

PROMPT = QASPER_SYSTEM_PROMPT + FEWSHOT_EXAMPLES

# model id -> (input $/MTok, output $/MTok)
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
}
ALIASES = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5"}


class BudgetStop(RuntimeError):
    pass


class SpendCap:
    """Thread-safe spend ledger: reserve worst case, then settle from actual usage."""

    def __init__(self, model: str, cap_usd: float, path: Path):
        if model not in PRICES:
            raise ValueError(f"No verified price for {model}")
        if not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("A finite positive budget is required")
        self.model, self.cap, self.path = model, cap_usd, path
        self.input_rate, self.output_rate = (p / 1e6 for p in PRICES[model])
        self.lock = threading.Lock()
        self.spent = 0.0
        self.in_flight = 0.0
        self.requests = 0
        if path.exists():
            saved = json.loads(path.read_text())
            self.spent, self.requests = saved["spent_usd"], saved["requests"]

    def worst_case(self, input_chars: int, max_output_tokens: int) -> float:
        # chars/2 over-counts tokens (the newer tokenizer is ~30% less efficient than 4 chars/token).
        return (input_chars / 2) * self.input_rate + max_output_tokens * self.output_rate

    def reserve(self, input_chars: int, max_output_tokens: int) -> float:
        amount = self.worst_case(input_chars, max_output_tokens)
        with self.lock:
            if self.spent + self.in_flight + amount > self.cap:
                raise BudgetStop(
                    f"Cap ${self.cap:.2f} cannot cover the next request "
                    f"(spent ${self.spent:.4f}, in flight ${self.in_flight:.4f})"
                )
            self.in_flight += amount
        return amount

    def settle(self, reserved: float, records: list[dict]) -> float:
        cost = sum(
            r.get("input_tokens", 0) * self.input_rate + r.get("output_tokens", 0) * self.output_rate
            for r in records
        )
        with self.lock:
            self.in_flight -= reserved
            self.spent += cost
            self.requests += len(records)
            self.path.write_text(json.dumps(
                {"model": self.model, "cap_usd": self.cap, "spent_usd": self.spent,
                 "requests": self.requests}, indent=2) + "\n")
        return cost


class CappedClaude(ClaudePolicy):
    """ClaudePolicy pinned to one model, QASPER prompt and parser, with a spend cap."""

    def __init__(self, model: str, cap: SpendCap, max_tokens: int):
        super().__init__(model=model, expected_model=model, system_prompt=PROMPT,
                         action_cleaner=clean_action, max_tokens=max_tokens, temperature=0.0)
        self.cap = cap
        self.episode_cost = 0.0

    def act(self, observation: str) -> str:
        chars = len(self.system_prompt) + len(observation) + sum(len(m["content"]) for m in self.history)
        reserved = self.cap.reserve(chars, self.max_tokens)
        before = len(self.usage_records)
        try:
            return super().act(observation)
        finally:
            # A request that failed before the provider answered has no usage, so it costs nothing.
            self.episode_cost += self.cap.settle(reserved, self.usage_records[before:])


def run_episode(question: dict, corpus: Corpus, corpus_path: str, model: str,
                cap: SpendCap, args) -> dict:
    env = DocumentExplorationEnv(corpus, [question], max_steps=args.max_steps, use_docker=False,
                                 corpus_path=corpus_path, include_preamble=False)
    policy = CappedClaude(model, cap, args.max_tokens)
    started = time.monotonic()
    trace, investigated = [], False
    terminal, finish, observation = "", "step_limit", env.reset(question_idx=0)
    try:
        for _ in range(args.max_steps):
            action = policy.act(observation)
            raw = policy.history[-1]["content"]
            usage = policy.usage_records[-1]
            observation, _, done, _ = env.step(action)
            trace.append({"action": action, "raw_action": raw, "observation": observation,
                          "done": done, "input_tokens": usage.get("input_tokens"),
                          "output_tokens": usage.get("output_tokens"),
                          "stop_reason": usage.get("stop_reason")})
            if not action.upper().startswith("SUBMIT:"):
                investigated = investigated or used_document_tool(action, observation)
            if done:
                if action.upper().startswith("SUBMIT:"):
                    terminal, finish = action, "submitted"
                break
    except BudgetStop:
        raise
    except Exception as exc:  # API or parsing failure: keep the partial episode, score it as unfinished
        finish = f"error:{type(exc).__name__}:{str(exc)[:200]}"
    finally:
        scoring = score_submission(terminal, question, env.corpus._documents, investigated=investigated)
        env.close()
    return {"question_id": question["id"], "question": question["question"],
            "expected_answerability": question["expected_answerability"], "trajectory": trace,
            "finish": finish, "duration_seconds": time.monotonic() - started, **scoring,
            "update": 0, "sample": 0, "model": model, "cost_usd": policy.episode_cost,
            "input_tokens": sum(t["input_tokens"] or 0 for t in trace),
            "output_tokens": sum(t["output_tokens"] or 0 for t in trace)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(ALIASES) + sorted(PRICES))
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    model = ALIASES.get(args.model, args.model)

    from dotenv import load_dotenv
    load_dotenv(override=True)
    data = json.loads(args.benchmark.read_text())
    if content_hash(args.corpus) != data["corpus_hash"]:
        p.error("Corpus differs from the frozen benchmark")
    questions = data["questions"][: args.limit or None]
    if any("answer_annotations" not in q for q in questions):
        p.error("Annotation-rich benchmark required")
    if args.out.exists() and not args.resume:
        p.error(f"{args.out} exists; pass --resume to continue it")
    args.out.mkdir(parents=True, exist_ok=True)

    episodes_path = args.out / "episodes.jsonl"
    done = set()
    if episodes_path.exists():
        done = {json.loads(line)["question_id"] for line in episodes_path.read_text().splitlines() if line.strip()}
    cap = SpendCap(model, args.budget_usd, args.out / "spend.json")
    manifest = {
        "model": model, "prices_usd_per_mtok": PRICES[model], "budget_usd": args.budget_usd,
        "benchmark": str(args.benchmark), "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "corpus_hash": data["corpus_hash"], "question_ids": [q["id"] for q in questions],
        "system_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "reward_version": REWARD_VERSION, "max_steps": args.max_steps, "max_tokens": args.max_tokens,
        "temperature_requested": 0.0, "workers": args.workers,
        "protocol": "mirrors scripts/train_qasper_grpo.py eval mode; provider-default thinking",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    corpus = Corpus(corpus_path=str(args.corpus))
    corpus.load(build_index=False)
    todo = [q for q in questions if q["id"] not in done]
    print(f"{model}: {len(todo)} episodes to run ({len(done)} already done), cap ${args.budget_usd:.2f}", flush=True)

    lock, stopped = threading.Lock(), None
    with ThreadPoolExecutor(max_workers=args.workers) as pool, episodes_path.open("a") as out:
        futures = {pool.submit(run_episode, q, corpus, str(args.corpus), model, cap, args): q for q in todo}
        for future in as_completed(futures):
            try:
                row = future.result()
            except BudgetStop as exc:
                stopped = str(exc)
                for pending in futures:
                    pending.cancel()
                continue
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
            progress = {k: row[k] for k in ("question_id", "reward", "finish", "cost_usd")}
            progress["spent_usd"] = round(cap.spent, 4)
            print(json.dumps(progress), flush=True)

    summary = {"model": model, "spent_usd": cap.spent, "requests": cap.requests, "stopped": stopped,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
