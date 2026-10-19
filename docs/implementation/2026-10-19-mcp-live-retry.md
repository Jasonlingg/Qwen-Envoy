# Research MCP readiness and bounded live retry

Status before execution: local connection verified; live retry not started.
This continues the previously authorized US$5 frozen-paper pilot. It is not
the proposed native-web comparison and performs no training.

## Hypothesis and decision

Hypothesis: a truthful project client identifier resolves the observed edge
filtering, allowing the existing pinned Qwen reader to execute its Docker paper
workflow and return inspectable evidence. Expected signal: authenticated gateway
health, exact advertised adapter identity, then actual budgeted Qwen actions.
Stop on gateway/auth/model readiness failures, unknown or excessive provider
usage, or the existing first-question submission/delegation gate. Preserve all
partial artifacts. This is a connectivity/workflow test, not proof of answer
quality, fine-tuning improvement, or superiority to native web research.

The prior pod is deleted. Its now-posted billing read reports US$0.4771146106
total (GPU 0.4678553515, disk 0.0092592591). The prior report's empty billing read
describes the earlier snapshot; a reconciliation will accompany the new result.
No inference request occurred in that attempt.

## Frozen limits and procedure

- Keep the five existing development questions and source versions. Reprepare
  the bundle because runtime code has changed; do not overwrite the old bundle.
- Same Nemotron host and pinned Envoy v5, thinking off; Docker-only generated
  Python. Two active pilot readers, four steps per paper, optional reasoner off.
- Keep the shared 96-request, 300,000-input-token, 98,304-reserved-output-token
  ledger and 1,024-output-token request maximum. Every dispatched attempt counts;
  no automatic retries. Missing usage blocks subsequent requests.
- One temporary 48 GB GPU; choose a currently available class below US$1.10/hour
  from the live catalog, record the actual hardware, and do not compare timing
  against the old A40 as if hardware were identical. No persistent/network volume.
- Start an authenticated stdlib gateway on 8000 before installing vLLM or loading
  weights. Bind vLLM to 127.0.0.1:8001 and expose only 8000/http. Preserve the base
  and adapter revisions and adapter weight SHA from the established launcher.
- Probe valid and absent/wrong bearer behavior and reject extra routes before
  inference. Use the same truthful client identifier as the inference transport.
  GET /health establishes gateway readiness only; GET /models establishes the
  declared adapter name only. Neither is a model answer or weight attestation.
- Bound gateway health wait to 90 seconds after the container starts, model
  readiness to 10 minutes after creation, and the complete temporary session to
  25 minutes. A tool-side cleanup timer supplements root supervision; it is not
  an external durable watchdog or a provider-enforced spending cap.
- Readiness must pass before the pilot starts. Run its existing first-question
  gate and preserve failures. Delete the owned temporary pod immediately when
  work ends or any gate fails, and verify get-pod 404. Do not modify other pods.

Compute at the permitted rate ceiling is at most approximately US$0.46 over
25 minutes, excluding disk and teardown delays. Catalog host-token estimates and
local request reservations are not a provider-enforced dollar cap. Record actual
usage, billing, startup/warm timing and cleanup; do not count unknowns as free.
The total US$5 planning approval includes the earlier posted charge.

## Local completion requirements

The source-only profile must advertise only five source tools. A real stdio
session must initialize and call cached paper tools. Codex must resolve the
project-local configuration without stored credentials. Restarting/reopening
the native client may be necessary to load its new tool inventory; registration
alone does not establish a native assistant's successful research answer.

Preflight/model HTTP must reject redirects, use bounded responses, sanitize
error details and keep operator-selected credentials isolated. Independent
review and local adversarial tests precede the live retry. Any model-assisted
claim review is explicitly labeled; human review stays blank until performed.
