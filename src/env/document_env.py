"""Gym-compatible RL environment for document exploration.

Observation space: string (question + execution output from last step)
Action space: string (Python code to execute, or "SUBMIT: <answer> CITATIONS: [...]")
Reward: 0 during exploration, verifiable score on submission
"""

from __future__ import annotations

import ast

from loguru import logger
from pydantic import BaseModel, Field

from src.env.corpus import Corpus
from src.env.evidence_state import (
    EscalationEvent,
    EvidenceState,
    VerifierEvent,
    format_escalation,
    format_feedback,
)
from src.env.repl import PersistentREPL
from src.env.reward import (
    RewardBreakdown,
    compute_reward,
    parse_submission,
    parse_submission_details,
)

# Re-exported so `from src.env.document_env import parse_submission` keeps working.
__all__ = [
    "DocumentExplorationEnv",
    "EpisodeInfo",
    "StepRecord",
    "SYSTEM_PREAMBLE",
    "parse_submission",
]


class StepRecord(BaseModel):
    step: int
    action: str
    observation: str
    reward: float
    done: bool
    reasoning: str | None = None
    logprob_diagnostics: dict | None = None


class EpisodeInfo(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    trajectory: list[StepRecord] = Field(default_factory=list)
    verifier_events: list[VerifierEvent] = Field(default_factory=list)
    escalation: EscalationEvent | None = None
    final_reward: RewardBreakdown | None = None


SYSTEM_PREAMBLE = """You have a Python REPL with these functions loaded:
  search(query, top_k=5)              → [{"doc_id", "title", "chunk", "score"}]
  search(query, method="chunk")       → chunk-level search (finds buried facts)
  read(doc_id)                        → full document text
  passage(doc_id, start, length)      → exact text with stable character offsets
  extract(doc_id, pattern)            → regex matches
  search_within(doc_id, query)        → search inside a specific document
  verify(doc_id, claim)               → check if a claim is supported by a doc
  list_docs()                         → [{"doc_id", "title", "chars"}]

TIP: Track discoveries as you go: known_facts = {}
     known_facts["company_x"] = "Apex Corp"
     Then use known_facts values in subsequent searches.

Respond with ONLY Python code. Use print() to see output.
When done, respond: SUBMIT: <answer> CITATIONS: ["id1", "id2"]
"""


def _submission_format(require_evidence: bool) -> str:
    format_line = 'SUBMIT: <answer> CITATIONS: ["doc_id"]'
    if require_evidence:
        format_line += ' EVIDENCE: [{"doc_id":"doc_id","start":0,"end":100}]'
    return format_line


class DocumentExplorationEnv:
    """Gym-compatible RL environment for document exploration via code execution."""

    def __init__(
        self,
        corpus: Corpus,
        questions: list[dict],
        max_steps: int = 10,
        use_docker: bool | None = None,
        corpus_path: str = "data/corpus",
        require_evidence: bool = False,
        include_preamble: bool = True,
        evidence_verifier: bool = False,
        verifier_feedback_budget: int = 1,
        escalate_after_verifier_failure: bool = True,
    ) -> None:
        self.corpus = corpus
        self.questions = questions
        self.max_steps = max_steps
        self._use_docker = use_docker
        self._corpus_path = corpus_path
        self.require_evidence = require_evidence
        self.include_preamble = include_preamble
        self.evidence_verifier = evidence_verifier
        if verifier_feedback_budget < 0:
            raise ValueError("verifier_feedback_budget must be non-negative")
        self.verifier_feedback_budget = verifier_feedback_budget
        # Explicit False is retained only for reproducing legacy fail-open runs.
        self.escalate_after_verifier_failure = escalate_after_verifier_failure
        self.repl = PersistentREPL(
            use_docker=use_docker, corpus_path=corpus_path,
        )
        self._episode: EpisodeInfo | None = None
        self._step_count: int = 0
        self._done: bool = False
        self._question_idx: int = 0
        self._evidence_state: EvidenceState | None = None

    def reset(self, question_idx: int | None = None) -> str:
        """Start a new episode. Returns initial observation (question + tools)."""
        # Clean up previous session
        try:
            self.repl.kill_session()
        except Exception:
            pass

        # Pick question
        if question_idx is not None:
            self._question_idx = question_idx
        q = self.questions[self._question_idx % len(self.questions)]

        self._episode = EpisodeInfo(
            question_id=q["id"],
            question=q["question"],
            gold_answer=q["answer"],
            gold_citations=q.get("expected_citations", []),
        )
        self._step_count = 0
        self._done = False
        self._evidence_state = None
        if self.evidence_verifier:
            self._evidence_state = EvidenceState({
                doc.doc_id: doc.chars for doc in self.corpus.list_documents()
            })

        # Start fresh REPL
        self.repl = PersistentREPL(
            use_docker=self._use_docker, corpus_path=self._corpus_path,
        )
        self.repl.start_session()

        # Build initial observation
        preamble = SYSTEM_PREAMBLE if self.include_preamble else ""
        if self.require_evidence and self.include_preamble:
            preamble += (
                "\nFor this research evaluation, inspect exact passages and append source spans:\n"
                'SUBMIT: <answer> CITATIONS: ["id"] EVIDENCE: '
                '[{"doc_id":"id","start":0,"end":100}]\n'
                "Citations identify papers; EVIDENCE identifies the exact passages supporting "
                "your important claims. The evaluator checks spans separately from answer "
                "quality.\n"
            )
        observation = (
            f"{preamble}\n\nQuestion: {q['question']}\n"
            if preamble
            else f"Question: {q['question']}\n"
        )
        logger.info(f"Episode started: {q['id']} — {q['question'][:80]}")
        return observation

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Take an action. Returns (observation, reward, done, info)."""
        if self._done:
            return "", 0.0, True, {"error": "Episode already done"}
        if self._episode is None:
            raise RuntimeError("Call reset() before step()")

        self._step_count += 1

        # Check if this is a submission
        submission = parse_submission_details(action)

        if submission is not None:
            answer, citations, evidence = submission
            reasons = self._submission_verifier_reasons(answer, citations, evidence)
            if reasons:
                if self._can_give_verifier_feedback():
                    return self._reject_with_verifier_feedback(action, "submission", reasons)
                if self.escalate_after_verifier_failure:
                    return self._escalate(action, "submission", reasons)
            reward_breakdown = compute_reward(
                predicted_answer=answer,
                predicted_citations=citations,
                gold_answer=self._episode.gold_answer,
                gold_citations=self._episode.gold_citations,
                steps_taken=self._step_count,
                max_steps=self.max_steps,
            )
            self._episode.final_reward = reward_breakdown
            self._done = True

            record = StepRecord(
                step=self._step_count,
                action=action,
                observation=f"Submitted. Reward: {reward_breakdown.total:.3f}",
                reward=reward_breakdown.total,
                done=True,
            )
            self._episode.trajectory.append(record)

            logger.info(
                f"Episode ended: reward={reward_breakdown.total:.3f} "
                f"(answer={reward_breakdown.answer_score:.2f}, "
                f"cit_p={reward_breakdown.citation_precision:.2f}, "
                f"cit_r={reward_breakdown.citation_recall:.2f}, "
                f"eff={reward_breakdown.efficiency_bonus:.2f})"
            )

            info = {
                "step": self._step_count,
                "reward_breakdown": reward_breakdown,
                "predicted_answer": answer,
                "predicted_citations": citations,
                "predicted_evidence": evidence,
            }
            return "", reward_breakdown.total, True, info

        if self._evidence_state is not None:
            duplicate = self._evidence_state.duplicate_reason(action)
            if duplicate:
                if self._can_give_verifier_feedback():
                    return self._reject_with_verifier_feedback(
                        action, "duplicate_action", [duplicate]
                    )
                if self.escalate_after_verifier_failure:
                    return self._escalate(action, "duplicate_action", [duplicate])

        # Invalid Python is a protocol mistake, so give a useful correction
        # without starting a sandbox execution. A SUBMIT action was handled above.
        try:
            ast.parse(action)
        except SyntaxError as exc:
            observation = (
                f"SyntaxError: {exc.msg} (line {exc.lineno}). "
                "This action was not run."
            )
        else:
            observation = self.repl.execute(action, timeout=30)
        if self._evidence_state is not None:
            self._evidence_state.observe(action, observation)

        # Distinguish a final-answer format mistake from a need to keep searching.
        if "SyntaxError" in observation:
            observation += (
                "\n\nHINT: Each turn must be either executable Python or one SUBMIT line. "
                "If the inspected evidence is sufficient, respond exactly in this format:\n"
                f"{_submission_format(self.require_evidence)}\n"
                "Use only document IDs and spans you inspected. If evidence is missing, "
                'continue with Python, for example: print(search("your query"))'
            )

        # Add step counter so agent knows urgency
        remaining = self.max_steps - self._step_count
        if remaining <= 1:
            observation += (
                "\n\n*** FINAL STEP — you MUST respond with: "
                f"{_submission_format(self.require_evidence)} ***"
            )
        elif remaining <= 5:
            observation += (
                f"\n\n[Step {self._step_count}/{self.max_steps} — {remaining} steps left. "
                "Run sufficiency check: can you answer now? If yes → SUBMIT immediately.]"
            )
        else:
            observation += f"\n\n[Step {self._step_count}/{self.max_steps}/{self.max_steps}]"

        # Check if max steps reached
        done = self._step_count >= self.max_steps
        if done:
            self._done = True
            logger.warning(f"Max steps ({self.max_steps}) reached — episode timeout")

        # Arbitrary stdout, errors, and our own hints are not verified evidence.
        # The baseline rewards only the final submitted answer and citations.
        step_reward = 0.0

        record = StepRecord(
            step=self._step_count,
            action=action,
            observation=observation,
            reward=step_reward,
            done=done,
        )
        self._episode.trajectory.append(record)

        info = {"step": self._step_count, "code": action, "output": observation}
        return observation, step_reward, done, info

    def _submission_verifier_reasons(
        self, answer: str, citations: list[str], evidence: list[dict]
    ) -> list[str]:
        if not self.evidence_verifier or self._evidence_state is None:
            return []
        return self._evidence_state.submission_reasons(
            answer=answer,
            citations=citations,
            evidence=evidence,
            require_evidence=self.require_evidence,
        )

    def _can_give_verifier_feedback(self) -> bool:
        return bool(
            self.evidence_verifier
            and self._evidence_state is not None
            and self._evidence_state.feedback_used < self.verifier_feedback_budget
            and self._step_count < self.max_steps
        )

    def _reject_with_verifier_feedback(
        self,
        action: str,
        kind: str,
        reasons: list[str],
    ) -> tuple[str, float, bool, dict]:
        assert self._episode is not None
        assert self._evidence_state is not None
        if any("refusal is premature" in reason.lower() for reason in reasons):
            self._evidence_state.abstention_recovery_active = True
        self._evidence_state.feedback_used += 1
        feedback = format_feedback(
            reasons=reasons,
            state=self._evidence_state,
            feedback_number=self._evidence_state.feedback_used,
            feedback_budget=self.verifier_feedback_budget,
            kind=kind,
        )
        event = VerifierEvent(
            step=self._step_count,
            kind=kind,
            reasons=reasons,
            feedback=feedback,
            state=self._evidence_state.snapshot(),
        )
        self._episode.verifier_events.append(event)
        self._episode.trajectory.append(StepRecord(
            step=self._step_count,
            action=action,
            observation=feedback,
            reward=0.0,
            done=False,
        ))
        info = {
            "step": self._step_count,
            "verifier_intervention": event,
            "output": feedback,
        }
        return feedback, 0.0, False, info

    def _escalate(
        self,
        action: str,
        kind: str,
        reasons: list[str],
    ) -> tuple[str, float, bool, dict]:
        """End the small-model episode with a structured host-model handoff."""
        assert self._episode is not None
        assert self._evidence_state is not None
        message = format_escalation(
            reasons=reasons,
            state=self._evidence_state,
            feedback_budget=self.verifier_feedback_budget,
        )
        escalation = EscalationEvent(
            step=self._step_count,
            kind=kind,
            reasons=reasons,
            candidate_action=action,
            message=message,
            state=self._evidence_state.snapshot(),
        )
        self._episode.escalation = escalation
        self._episode.trajectory.append(StepRecord(
            step=self._step_count,
            action=action,
            observation=message,
            reward=0.0,
            done=True,
        ))
        self._done = True
        logger.info(
            f"Episode escalated after {self._evidence_state.feedback_used} "
            "verifier recoveries"
        )
        return message, 0.0, True, {
            "step": self._step_count,
            "escalated": True,
            "escalation": escalation,
            "output": message,
        }

    def get_trajectory(self) -> list[StepRecord]:
        """Return the full trajectory for this episode."""
        if self._episode is None:
            return []
        return self._episode.trajectory

    def get_episode_info(self) -> EpisodeInfo | None:
        """Return full episode info including question and reward."""
        return self._episode

    def close(self) -> None:
        """Clean up REPL session."""
        try:
            self.repl.kill_session()
        except Exception:
            pass
