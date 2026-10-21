# Repaired-harness passage-reader rerun

User instruction: “ok rerun it”, after proposing the same three cases with the
repaired harness, before deciding whether Qwen needs training. This authorizes
one new attempt, not a reset of the original consumed allowance.

Hypothesis: bounded arithmetic normalization and explicit literal read examples
remove the observed read-call friction, allowing the unchanged v5 checkpoint to
finish the two answerable cases and the scoped missing-evidence procedure.
Expected signal: fewer rejected document actions and supported final answers or
a completed gap report within the unchanged four-action limit per case.

Decision rule: retain and report every action and case. Both answerable cases
must cover the requested facts with their own adequate citations; the gap case
must complete the gap procedure without inventing a target measurement. Review
claims against the actual submitted passages. Mechanical acceptance is separate
from semantic support; independent human review stays blank. Compare descriptively
with the previous run, without attributing changes to arithmetic alone: explicit
read examples also changed the prompt, and remote determinism is not guaranteed.

Same exposed development cases, original source windows, checkpoint, decoding,
Docker sandbox and request limits. No training, reward change, new questions,
reserved evaluation access or inference resubmission. The host supplies relevant
windows, so this does not test paper discovery or autonomous research.

Limits: one existing A5000, 900 seconds including startup, twelve requests,
98,304 reserved input/output tokens, maximum four requests per case. Fresh price
and empty parked endpoint reads precede activation. The established $0.50 local
allowance is guarded by runtime and request/token limits, not a provider dollar
cap. Use a new exclusive `out/passage-reader-three-case-RERUN-1` marker and new
output directory; preserve every original artifact. Native startup monitoring,
owned-job/container cleanup, full configuration restoration and independent
shutdown reads are required. No additional paid attempt is included.
