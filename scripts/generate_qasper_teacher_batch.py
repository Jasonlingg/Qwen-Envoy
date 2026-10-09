"""Generate QASPER-train code trajectories for review before Qwen3-8B SFT.

Unlike the earlier one-off repair scripts, this runs the full stratified benchmark
(out/research/qasper-train-teacher-v2, 300 questions: 100 "insufficient" + 200
"sufficient") through Claude Sonnet 5, writing incrementally so a crash, rate
limit, or Ctrl-C doesn't lose completed work.

"Insufficient" questions get the ground-truth answerability as teacher-only
supervision (TeacherHintPolicy) — never exported to the student, see
src/policies/teacher_hint_policy.py. Both labels receive the same evidence-first
workflow hint. The teacher uses the student's system prompt. No completed
trajectory is automatically accepted for SFT.

Supports --workers > 1 for concurrent generation: each worker gets its own fresh
DocumentExplorationEnv + ClaudePolicy instance (no shared mutable state), the
same pattern src/eval/harness.py already uses for parallel eval.

Usage:
  python scripts/generate_qasper_teacher_batch.py \
      --benchmark out/research/qasper-train-teacher-v2/benchmark.json \
      --corpus out/research/qasper-train-teacher-v2/corpus \
      --output out/research/qasper-teacher-batch-v3/trajectories.jsonl \
      --model claude-sonnet-5 \
      --budget-usd 10 --workers 4
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv
from src.eval.artifacts import content_hash
from src.eval.harness import run_single
from src.policies.claude_policy import ClaudePolicy, ModelIdentityError
from src.policies.code_execution import QASPER_SYSTEM_PROMPT as SYSTEM_PROMPT, used_document_tool, clean_action
from src.policies.teacher_budget import BudgetedMessages, BudgetStop, TeacherBudget
from src.policies.teacher_hint_policy import TeacherHintPolicy
from src.eval.qasper_reward import REWARD_VERSION, score_submission

WORKFLOW_HINT = (
    "Demonstrate efficient evidence gathering. The question supplies the exact paper doc_id: "
    "start by printing a relevant search_within() result or a bounded slice of read() for that "
    "paper. Do not search the whole library to rediscover its identity, and do not use verify() "
    "keyword matches as proof. Read the actual passage. After each observation, submit as soon "
    "as every requested part is supported. Aim for 2-6 actions INCLUDING SUBMIT; do not add "
    "searches just to fill a quota. If support is missing, inspect a different relevant section "
    "or use different terms within the paper. A failed search is not proof the paper lacks the "
    "answer. Distinguish an explicit negative finding from missing information. Never claim "
    "to have checked text you did not see, or infer missing numbers. If the bounded investigation "
    "still cannot support an answer, submit exactly Unanswerable with empty citations/evidence. "
    "Do not sacrifice completeness to meet the action target. "
    "Give the shortest complete answer covering ONLY what was asked: a name, a number, or a complete "
    "list if requested. Do not gather unrequested statistics or biographies. Never fill in "
    "BIBREF citation names from memory. Preserve distinctions such as range versus average, "
    "candidate entities versus documents, and an explicit negative versus an unknown. "
    "Every variable must have been assigned in an executed action before you use it. "
    "Do not print results already shown. When windows repeat or cut off a needed sentence, "
    "print a bounded surrounding slice of read(doc_id) at the returned offset, instead of "
    "issuing more nearly identical keyword searches."
    " search_within() returns dictionaries with keys text, offset, score (not chunk); "
    "print the result or item['text']. Never put an unseen reference-answer value, number, "
    "or list of answer entities into a search query. Start with the question's concepts; "
    "refine queries using facts actually returned by the tools."
    " CRITICAL: never estimate citation offsets. After search reveals a useful sentence, "
    "read the text and compute its offsets in Python, then print passage() to inspect the "
    "EXACT slice you will cite. For example, after observing a sentence, use: "
    "text = read(doc_id); quote = '<exact sentence already observed>'; "
    "start = text.find(quote); assert start >= 0; "
    "print(passage(doc_id, start=start, length=len(quote))). "
    "Copy start/end from that actual output. Include every passage needed for the answer. "
    "For yes/no questions, the answer field must be only Yes or No (no explanation). "
    "For a missing answer, the entire final action is exactly "
    "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []. "
    "Emit bare executable Python without commentary before or after it, or one final SUBMIT line. "
    "Aim to search once, inspect the relevant passages, then submit."
)

INSUFFICIENT_HINT = (
    "TEACHER-ONLY NOTE (ground truth, for your calibration only — do not reference "
    "this note in your search steps or final answer): annotators labeled this question "
    "UNANSWERABLE from this specific paper. The annotation can be imperfect: do not invent "
    "a rationale to force the label when the actual passage supports an answer. "
    "This label does not replace reading evidence: "
    "inspect the supplied paper and relevant passages before submitting. Establish what is "
    "missing without claiming an exhaustive search or treating the missing fact as false. "
    "Do not guess or mention this label in code, comments, or the answer."
)

REFERENCE_INSTRUCTION = (
    "TEACHER-ONLY REFERENCE (never quote this note, its label, or unseen evidence in your "
    "code/comments/answer): use the reference to identify the requested facts and a useful "
    "search. You must independently retrieve support via the runtime tools before SUBMIT. "
    "A reference answer is not itself observed evidence. If it conflicts with the paper or "
    "requires an unjustified inference, report only what the inspected source supports.\n"
)


def teacher_hint(question: dict, reference_guidance: bool = False) -> str:
    hint = WORKFLOW_HINT
    if question["expected_answerability"] == "insufficient":
        return hint + "\n\n" + INSUFFICIENT_HINT
    if reference_guidance:
        hint += "\n\n" + REFERENCE_INSTRUCTION + json.dumps({
            "reference_answer": question["answer"],
            "reference_passages": question.get("grader_notes", []),
        }, ensure_ascii=False)
    return hint


MAX_CONSECUTIVE_FAILURES = 8
TEACHER_MODEL = "claude-sonnet-5"

_write_lock = threading.Lock()
_consecutive_failures = 0
_stop = threading.Event()


class SonnetTeacherPolicy(ClaudePolicy):
    """Keep teacher generation pinned independently of the cheaper baseline default."""

    def __init__(self, model: str, budget: TeacherBudget | None = None, question_id: str = ""):
        super().__init__(model=model, expected_model=TEACHER_MODEL, system_prompt=SYSTEM_PROMPT,
                         action_cleaner=clean_action)
        if budget is not None:
            # No hidden retries that might charge twice after an interrupted response.
            self.client = self.client.with_options(max_retries=0, timeout=120)
            self.client.messages = BudgetedMessages(self.client.messages, budget, question_id)

    def act(self, observation: str) -> str:
        if _stop.is_set():
            raise RuntimeError("Teacher batch has stopped; refusing another API request")
        try:
            return super().act(observation)
        except (ModelIdentityError, BudgetStop):
            _stop.set()
            logger.exception("Teacher model/budget guard stopped subsequent requests")
            raise


def _run_one(q: dict, corpus: Corpus, corpus_path: str, max_steps: int, model: str,
             reference_guidance: bool = False, budget: TeacherBudget | None = None) -> dict:
    if _stop.is_set():
        raise RuntimeError("Teacher batch stopped before this episode began")
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=[q],
        max_steps=max_steps,
        use_docker=False,
        corpus_path=corpus_path,
        require_evidence=True,
        include_preamble=False,
    )
    inner = None
    try:
        inner = SonnetTeacherPolicy(model=model, budget=budget, question_id=q["id"])
        policy = TeacherHintPolicy(inner, hint=teacher_hint(q, reference_guidance))
        try:
            result = run_single(env, policy, 0)
            result_dict = json.loads(result.model_dump_json())
        except Exception as exc:
            result_dict = {"question": q["question"], "status": "error",
                           "error": f"{type(exc).__name__}: {exc}",
                           "trajectory": [s.model_dump() for s in env.get_trajectory()],
                           "reward": 0.0, "steps": len(env.get_trajectory())}
        raw_actions = [m["content"] for m in inner.history if m["role"] == "assistant"]
        for step, raw in zip(result_dict["trajectory"], raw_actions):
            step["raw_action"] = raw
        result_dict["question_id"] = q["id"]
        result_dict["expected_answerability"] = q["expected_answerability"]
        result_dict["teacher_model"] = model
        result_dict["teacher_usage"] = inner.usage_records
        result_dict["teacher_reference_guidance"] = reference_guidance
        result_dict["student_protocol"] = "qasper-span-v1"
        result_dict["initial_observation"] = f"Question: {q['question']}\n"
        # Keep the environment's legacy reward separate. This is only a proxy;
        # a real citation still requires semantic review of what it supports.
        investigated = any(
            used_document_tool(s["action"], s["observation"])
            for s in result_dict["trajectory"] if not s["action"].startswith("SUBMIT:")
        )
        terminal = result_dict["trajectory"][-1]["action"] if result_dict["trajectory"] else ""
        result_dict["qasper_score"] = score_submission(
            terminal, q, corpus._documents, investigated=bool(investigated))
        return result_dict
    finally:
        env.close()
        if inner is not None:
            inner.client.close()


def main(
    benchmark: Path = typer.Option(...),
    corpus_path: str = typer.Option(..., "--corpus"),
    output: Path = typer.Option(...),
    model: str = typer.Option(..., help=f"Required teacher: {TEACHER_MODEL}; other models are rejected"),
    budget_usd: float = typer.Option(..., min=0.01, help="Persistent API spending cap, including resumed calls"),
    budget_ledger: Path | None = typer.Option(None, help="Share one cap across explicitly reviewed generation attempts"),
    reference_guidance: bool = typer.Option(False, "--reference-guidance",
        help="Give training-only gold answers/passages to teacher; never export the hints"),
    max_steps: int = typer.Option(10),
    workers: int = typer.Option(4, min=1, max=8, help="Concurrent episodes. Each gets its own env+policy."),
    limit: int = typer.Option(None, help="Only run the first N questions (for smoke testing)"),
) -> None:
    if model != TEACHER_MODEL:
        raise typer.BadParameter(f"Teacher generation is pinned to {TEACHER_MODEL}; got {model!r}")
    load_dotenv(override=True)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    source = json.loads(benchmark.read_text())
    if source.get("source_split") != "train":
        raise typer.BadParameter("Teacher generation is restricted to QASPER's train split")
    questions = source["questions"]
    if limit:
        questions = questions[:limit]
    if not questions or any("answer_annotations" not in q for q in questions):
        raise typer.BadParameter("Annotation-rich training benchmark required")

    output.parent.mkdir(parents=True, exist_ok=True)
    actual_corpus_hash = content_hash(Path(corpus_path))
    if actual_corpus_hash != source["corpus_hash"]:
        raise typer.BadParameter("Corpus differs from the frozen benchmark")
    identity = {
        "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "corpus_hash": actual_corpus_hash, "model": model, "max_steps": max_steps,
        "required_response_model": TEACHER_MODEL,
        "student_protocol": "qasper-span-v1", "include_preamble": False,
        "action_cleaner": "src.policies.code_execution.clean_action",
        "require_evidence": True, "diagnostic_reward_version": REWARD_VERSION,
        "budget_usd": budget_usd, "sdk_retries": 0,
        "budget_ledger": str((budget_ledger or output.with_suffix('.budget.json')).resolve()),
        "max_tokens": 4096, "requested_temperature": 0.0,
        "temperature_note": "ClaudePolicy retries without temperature when the provider rejects it",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "teacher_hints_sha256": hashlib.sha256(
            (WORKFLOW_HINT + INSUFFICIENT_HINT + REFERENCE_INSTRUCTION).encode()
        ).hexdigest(),
        "reference_guidance": reference_guidance,
        "system_prompt": SYSTEM_PROMPT,
        "workflow_hint": WORKFLOW_HINT,
        "insufficient_hint": INSUFFICIENT_HINT,
        "reference_instruction": REFERENCE_INSTRUCTION,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            "scripts/generate_qasper_teacher_batch.py", "src/policies/teacher_budget.py",
            "src/policies/claude_policy.py", "src/policies/code_execution.py",
            "src/env/document_env.py", "src/env/tools.py", "src/eval/qasper_reward.py"]},
    }
    manifest = output.with_suffix(".manifest.json")
    if output.exists() and (not manifest.exists() or json.loads(manifest.read_text()) != identity):
        raise typer.BadParameter("Refusing to mix teacher configurations; use a new output file")
    manifest.write_text(json.dumps(identity, indent=2) + "\n")
    done_ids = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["question_id"])
        logger.info(f"Resuming: {len(done_ids)} already done, skipping those")

    corpus = Corpus(corpus_path=corpus_path)
    corpus.load(build_index=False)

    remaining = [q for q in questions if q["id"] not in done_ids]
    logger.info(f"Generating {len(remaining)}/{len(questions)} trajectories "
                f"-> {output} (workers={workers})")

    global _consecutive_failures
    _stop.clear()
    _consecutive_failures = 0
    ok, failed = 0, 0
    budget = TeacherBudget(budget_ledger or output.with_suffix(".budget.json"), budget_usd)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool, output.open("a") as f:
            futures = {
                pool.submit(_run_one, q, corpus, corpus_path, max_steps, model, reference_guidance, budget): q
                for q in remaining
            }
            for i, fut in enumerate(as_completed(futures)):
                if _stop.is_set():
                    for pending in futures:
                        pending.cancel()
                if fut.cancelled():
                    continue
                q = futures[fut]
                try:
                    result_dict = fut.result()
                    with _write_lock:
                        f.write(json.dumps(result_dict) + "\n")
                        f.flush()
                    if result_dict.get("status") == "error":
                        raise RuntimeError(result_dict["error"])
                    ok += 1
                    _consecutive_failures = 0
                    logger.info(f"[{i+1}/{len(remaining)}] {q['id']} "
                                f"({q['expected_answerability']}) reward={result_dict['reward']:.3f} "
                                f"steps={result_dict['steps']}")
                except Exception as exc:
                    failed += 1
                    _consecutive_failures += 1
                    logger.error(f"[{i+1}/{len(remaining)}] {q['id']} FAILED: {exc}")
                    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.error(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures — "
                                     "stopping (likely out of credits or rate-limited).")
                        _stop.set()
    finally:
        logger.info(f"Teacher budget: {budget.summary()}")
        budget.close()
    logger.info(f"Done. {ok} generated, {failed} failed. Output: {output}")


if __name__ == "__main__":
    typer.run(main)
