"""Evaluation harness: run policies through the environment and collect results."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, StepRecord
from src.env.evidence_state import EscalationEvent, VerifierEvent
from src.env.reward import REWARD_VERSION
from src.policies.protocol import Policy


class EvalResult(BaseModel):
    question_id: str
    question: str
    policy_name: str
    reward: float
    answer_score: float
    citation_precision: float
    citation_recall: float
    efficiency_bonus: float
    steps: int
    trajectory: list[StepRecord] = Field(default_factory=list)
    predicted_answer: str = ""
    predicted_citations: list[str] = Field(default_factory=list)
    predicted_evidence: list[dict] = Field(default_factory=list)
    duration_seconds: float = 0.0
    outcome_reward: float = 0.0
    shaping_reward: float = 0.0
    episode_return: float = 0.0
    reward_version: str = REWARD_VERSION
    status: str = "completed"
    error: str | None = None
    verifier_events: list[VerifierEvent] = Field(default_factory=list)
    escalation: EscalationEvent | None = None
    policy_metadata: dict = Field(default_factory=dict)


def _policy_eval_metadata(policy: Policy | None) -> dict:
    """Telemetry failure must not discard an episode or its original failure."""
    metadata_fn = getattr(policy, "eval_metadata", None)
    if not callable(metadata_fn):
        return {}
    try:
        metadata = metadata_fn()
        return metadata if isinstance(metadata, dict) else {"metadata_collection_failed": True}
    except Exception:
        return {"metadata_collection_failed": True}


def run_single(
    env: DocumentExplorationEnv,
    policy: Policy,
    question_idx: int,
) -> EvalResult:
    """Run a single policy on a single question through the environment."""
    start = time.perf_counter()
    # Reset
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset(question_idx=question_idx)
    q = env.questions[question_idx % len(env.questions)]

    steps = 0
    reward = 0.0
    episode_return = 0.0
    outcome_reward = 0.0
    predicted_answer = ""
    predicted_citations: list[str] = []
    predicted_evidence: list[dict] = []
    answer_score = 0.0
    cit_p = 0.0
    cit_r = 0.0
    eff = 0.0
    status = "completed"

    done = False
    while not done:
        action = policy.act(obs)
        reasoning = getattr(policy, "last_reasoning", None)
        logprob_diagnostics = getattr(policy, "last_logprob_diagnostics", None)
        obs, reward, done, info = env.step(action)
        if (isinstance(reasoning, str) and reasoning) or isinstance(
            logprob_diagnostics, dict
        ):
            # Generated reasoning and token telemetry are trace data, not
            # executable input or benchmark scores.
            trajectory = env.get_trajectory()
            if trajectory:
                if isinstance(reasoning, str) and reasoning:
                    trajectory[-1].reasoning = reasoning
                if isinstance(logprob_diagnostics, dict):
                    trajectory[-1].logprob_diagnostics = logprob_diagnostics
        episode_return += reward
        steps += 1

        if done and "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            answer_score = rb.answer_score
            cit_p = rb.citation_precision
            cit_r = rb.citation_recall
            eff = rb.efficiency_bonus
            outcome_reward = rb.total - rb.efficiency_bonus
            predicted_answer = info.get("predicted_answer", "")
            predicted_citations = info.get("predicted_citations", [])
            predicted_evidence = info.get("predicted_evidence", [])
        elif done and info.get("escalated"):
            status = "escalated"

    duration = time.perf_counter() - start
    trajectory = env.get_trajectory()
    episode = env.get_episode_info()
    policy_metadata = _policy_eval_metadata(policy)

    return EvalResult(
        question_id=q["id"],
        question=q["question"],
        policy_name=type(policy).__name__,
        reward=reward,
        answer_score=answer_score,
        citation_precision=cit_p,
        citation_recall=cit_r,
        efficiency_bonus=eff,
        steps=steps,
        trajectory=trajectory,
        predicted_answer=predicted_answer,
        predicted_citations=predicted_citations,
        predicted_evidence=predicted_evidence,
        duration_seconds=duration,
        outcome_reward=outcome_reward,
        shaping_reward=episode_return - outcome_reward,
        episode_return=episode_return,
        status=status,
        verifier_events=episode.verifier_events if episode is not None else [],
        escalation=episode.escalation if episode is not None else None,
        policy_metadata=policy_metadata,
    )


def _run_one_question(
    corpus: Corpus,
    questions: list[dict],
    q_idx: int,
    policy_name: str,
    policy_factory: Callable[[], Policy],
    max_steps: int,
    use_docker: bool | None,
    corpus_path: str,
    require_evidence: bool = False,
    include_preamble: bool = True,
    evidence_verifier: bool = False,
    verifier_feedback_budget: int = 1,
    escalate_after_verifier_failure: bool = False,
) -> EvalResult:
    """Run one question with a fresh env + policy instance (safe for parallel use)."""
    q = questions[q_idx]
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=max_steps,
        use_docker=use_docker,
        corpus_path=corpus_path,
        require_evidence=require_evidence,
        include_preamble=include_preamble,
        evidence_verifier=evidence_verifier,
        verifier_feedback_budget=verifier_feedback_budget,
        escalate_after_verifier_failure=escalate_after_verifier_failure,
    )
    # Measure the same attempt boundary on success and failure. Include policy
    # initialization and episode reset; exclude environment disposal below.
    started = time.perf_counter()
    policy = None
    try:
        policy = policy_factory()
        result = run_single(env, policy, q_idx)
        result.policy_name = policy_name
        result.duration_seconds = time.perf_counter() - started
        return result
    except Exception as e:
        logger.error(f"  → FAILED {q['id']}: {e}")
        episode = env.get_episode_info()
        trajectory = env.get_trajectory()
        return EvalResult(
            question_id=q["id"],
            question=q["question"],
            policy_name=policy_name,
            reward=0.0,
            answer_score=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            efficiency_bonus=0.0,
            steps=len(trajectory),
            duration_seconds=time.perf_counter() - started,
            trajectory=trajectory,
            status="error",
            error=f"{type(e).__name__}: {e}",
            verifier_events=episode.verifier_events if episode is not None else [],
            escalation=episode.escalation if episode is not None else None,
            policy_metadata=_policy_eval_metadata(policy),
        )
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_eval(
    corpus: Corpus,
    questions: list[dict],
    policies: dict[str, Policy | Callable[[], Policy]],
    max_steps: int = 10,
    use_docker: bool | None = None,
    corpus_path: str = "data/corpus",
    question_ids: list[str] | None = None,
    workers: int = 1,
    require_evidence: bool = False,
    include_preamble: bool = True,
    evidence_verifier: bool = False,
    verifier_feedback_budget: int = 1,
    escalate_after_verifier_failure: bool = False,
) -> list[EvalResult]:
    """Run all policies on all (or selected) questions.

    Set workers > 1 to parallelize across questions — each worker gets its own
    env + policy instance so there is no shared mutable state.
    policy values may be either instances (workers=1) or zero-arg callables
    (workers >= 1). Instances are wrapped in a lambda automatically.
    """
    results: list[EvalResult] = []

    if question_ids:
        q_indices = [i for i, q in enumerate(questions) if q["id"] in question_ids]
    else:
        q_indices = list(range(len(questions)))

    # Normalise: wrap plain instances in a factory so parallel path always has a callable.
    factories: dict[str, Callable[[], Policy]] = {}
    for name, p in policies.items():
        factories[name] = p if callable(p) and not hasattr(p, "act") else (lambda _p=p: _p)

    try:
        for policy_name, factory in factories.items():
            logger.info(f"Running policy: {policy_name} (workers={workers})")

            if workers <= 1:
                for q_idx in q_indices:
                    q = questions[q_idx]
                    logger.info(f"  Question {q['id']}: {q['question'][:60]}...")
                    result = _run_one_question(
                        corpus, questions, q_idx, policy_name, factory,
                        max_steps, use_docker, corpus_path, require_evidence,
                        include_preamble,
                        evidence_verifier,
                        verifier_feedback_budget,
                        escalate_after_verifier_failure,
                    )
                    results.append(result)
                    logger.info(
                        f"  → reward={result.reward:.3f}, steps={result.steps}, "
                        f"time={result.duration_seconds:.1f}s"
                    )
            else:
                futures = {}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for q_idx in q_indices:
                        fut = pool.submit(
                            _run_one_question,
                            corpus, questions, q_idx, policy_name, factory,
                            max_steps, use_docker, corpus_path, require_evidence,
                            include_preamble,
                            evidence_verifier,
                            verifier_feedback_budget,
                            escalate_after_verifier_failure,
                        )
                        futures[fut] = q_idx

                    for fut in as_completed(futures):
                        result = fut.result()
                        results.append(result)
                        logger.info(
                            f"  {result.question_id} → reward={result.reward:.3f}, "
                            f"steps={result.steps}, time={result.duration_seconds:.1f}s"
                        )

    except KeyboardInterrupt:
        logger.warning("Eval interrupted — returning partial results")

    return results
