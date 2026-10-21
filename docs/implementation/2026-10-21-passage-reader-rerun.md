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

## Actual result

Scope commit `2ffb7d9`; frozen preparation and live implementation commit
`df8f243`. Focused checks: 25 passed in 12.48 seconds. Full relevant suite:
1925 passed in 160.61 seconds, no skips and three existing SWIG warnings.
The local three-case rehearsal used synthetic responses; live results below
came from nine new Qwen v5 calls through the actual in-process MCP/Docker path.

QLoRA finished in three calls: two reads, then the requested storage/computation
dtypes and LoRA-only weight gradients with both adequate citations. AWQ finished
in two calls with the same six supported conditions as the first attempt. The
implementing assistant checked each claim against its own submitted source spans;
independent human review remains blank. PagedAttention read both assigned windows
successfully, but submitted Unanswerable at step three before its gap report.
After rejection it recorded the gap at step four with no final action remaining.
No invented p95 was observed; no substantive partial findings were returned.
It remained incomplete. The predeclared overall rule was not met.

All five passage reads used explicit literal lengths and succeeded; no model
Python execution error occurred. There was one rejected submission. Both the
normalizer and literal read examples changed, so improvement cannot be attributed
to arithmetic handling alone; live arithmetic was not exercised. Supported
answerable cases increased descriptively from one to two with unchanged weights.
This tiny exposed-development rerun does not establish generalization, a model
training gain, or effort/cost benefit over ordinary assistant research.

Usage: 25,784 input + 518 output = 26,302 reported tokens; 73,728 reserved tokens
of the 98,304 cap; nine requests of twelve. All jobs completed, zero new failed
or retried jobs. Native startup identity/readiness succeeded with thirteen received
log snapshots. Supervisor time before cleanup was 185.907 seconds; watchdog did
not fire. Fresh endpoint/health/worker reads at 23:19:41 UTC verified full original
configuration restored, zero workers and no pending jobs. All six owned local
rehearsal/live containers were independently checked absent. Actual invoice cost
is unknown. Eleven protected files and 381 captured prior artifacts were unchanged.
The new allowance is consumed; no further live run is included.

[Full results, per-action observations and claim review](../../examples/passage-reader-rerun-1/README.md).
The remaining failure is the abstention/gap action order under a four-call cap,
plus failure to return useful partial findings. No further harness or training
change was made after observing this rerun.
