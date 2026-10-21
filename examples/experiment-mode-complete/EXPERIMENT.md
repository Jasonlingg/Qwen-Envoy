# Experiment brief

Can two Qwen v5 reader requests share one A5000, and should we claim a latency win?

Status: frozen · revision 2

## Current setup

- task: Retrospective review of a fixed-input serving diagnostic; not full agents
- model: Qwen/Qwen3-8B
- checkpoint: jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5/artifacts/full/checkpoint-50
- revision: b968826d9c46dd6066d109eabc6255188de91218/7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3
- hardware: NVIDIA RTX A5000
- workload: Two measured rounds over the same pinned historical input pair
- serving: vLLM 0.31.0; torch 2.13.0+cu130

## Goal

latency

## Proposed change

Compare previously recorded concurrent dispatch with serial dispatch.

## Hypothesis

Concurrent requests can share one engine; latency benefit is uncertain.

## Expected signal

Complete matched outputs and two-request overlap; admit latency only when the original uncertainty/comparability check allows it.

## Applicability

Two short fixed inputs only; reconstruction after results were seen.

## Constraints

Not established yet

## Alternatives considered

Not established yet

## Differences from source settings

Not established yet

## Procedure

- Verify original saved-source checksums and derive selected observations.
- Import prior research, freeze this retrospective analysis, and record measurements.
- Restart MCP and recover the same result, evidence and unresolved performance gate.

## Unknowns

- Representative reader evidence quality remains unmeasured.
- Reliable latency benefit and all-in invoice cost remain unknown.

## Limitations

- Reconstruction, not preregistration or a new model/optimization run.
- No human usefulness review or fine-tuning improvement demonstrated.

## Observations


## Baseline

- description: Recorded serial pair
- identity:
  - hardware: NVIDIA RTX A5000
  - model revision: b968826d9c46dd6066d109eabc6255188de91218/7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3
  - software revision: vLLM 0.31.0; torch 2.13.0+cu130
  - workload: 5940a8efb98774ce2f820a391f94615ee8dce1a0825f4ed0fd6ce9950f8d657e

## Candidate

- description: Recorded concurrent pair
- identity:
  - hardware: NVIDIA RTX A5000
  - model revision: b968826d9c46dd6066d109eabc6255188de91218/7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3
  - software revision: vLLM 0.31.0; torch 2.13.0+cu130
  - workload: 5940a8efb98774ce2f820a391f94615ee8dce1a0825f4ed0fd6ce9950f8d657e

## Evaluation sample

- population: Two measured rounds over the same pinned historical input pair
- purpose: historical
- expected cases: 2
- sample ids:
  - pair\_round\_1
  - pair\_round\_2
- sample set sha256: Unknown
- dataset revision: 5940a8efb98774ce2f820a391f94615ee8dce1a0825f4ed0fd6ce9950f8d657e

## Controlled conditions

- Item 1:
  - name: dispatch\_mode
  - role: proposed\_change
  - baseline: serial
  - candidate: concurrent
- Item 2:
  - name: runtime\_config
  - role: held\_constant
  - baseline: \{&quot;ENABLE\_PREFIX\_CACHING&quot;: &quot;false&quot;, &quot;MAX\_CONCURRENCY&quot;: &quot;2&quot;, &quot;MAX\_MODEL\_LEN&quot;: &quot;8192&quot;, &quot;MAX\_NUM\_SEQS&quot;: &quot;2&quot;\}
  - candidate: \{&quot;ENABLE\_PREFIX\_CACHING&quot;: &quot;false&quot;, &quot;MAX\_CONCURRENCY&quot;: &quot;2&quot;, &quot;MAX\_MODEL\_LEN&quot;: &quot;8192&quot;, &quot;MAX\_NUM\_SEQS&quot;: &quot;2&quot;\}

## Metrics

- Item 1:
  - metric id: completed\_calls
  - unit: count
  - aggregation: total
  - population: Two measured rounds over the same pinned historical input pair
  - direction: higher
  - measurement kind: other
  - timing scope: Unknown
  - failure policy: unspecified
  - requires semantic review: False
- Item 2:
  - metric id: engine\_overlap
  - unit: count
  - aggregation: minimum round peak
  - population: Two measured rounds over the same pinned historical input pair
  - direction: higher
  - measurement kind: other
  - timing scope: Unknown
  - failure policy: unspecified
  - requires semantic review: False
- Item 3:
  - metric id: pair\_seconds
  - unit: seconds
  - aggregation: mean
  - population: Two measured rounds over the same pinned historical input pair
  - direction: lower
  - measurement kind: latency
  - timing scope: warm
  - failure policy: unspecified
  - requires semantic review: False

## Decision criteria

- Item 1:
  - criterion id: all\_timed\_calls\_completed
  - metric id: completed\_calls
  - operation: absolute\_gte
  - threshold: 4.0
  - required: True
- Item 2:
  - criterion id: same\_engine\_overlap
  - metric id: engine\_overlap
  - operation: absolute\_gte
  - threshold: 2.0
  - required: True
- Item 3:
  - criterion id: resolved\_pair\_time\_nonregression
  - metric id: pair\_seconds
  - operation: delta\_lte
  - threshold: 0.0
  - required: True

## Resource ceilings (recorded, not enforced)

- max model requests: 0
- max input tokens: 0
- max output tokens: 0
- max gpu seconds: 0.0
- max elapsed seconds: 120.0
- max cost usd: 0.0
- cost unknown reason: Unknown
- stop conditions:
  - Source hash mismatch or MCP error; never dispatch inference.

## Evidence

Source checks establish provenance; semantic support remains unreviewed.

### observed\_runtime

Verification: artifact_value_verified
Source SHA-256: 6169e7cf458406343756022590503d26936002374660f9c2038998798e5f9902
Reference: \{&quot;json\_pointer&quot;: &quot;/live\_identity&quot;, &quot;relative\_path&quot;: &quot;serving-observations.json&quot;\}
> \{&quot;adapter\_rank&quot;: 4, &quot;adapter\_repo&quot;: &quot;jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5&quot;, &quot;adapter\_revision&quot;: &quot;27b912a863ff914ad45baa03624d8911dc1e17fb&quot;, &quot;adapter\_sha256&quot;: &quot;7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3&quot;, &quot;adapter\_subfolder&quot;: &quot;artifacts/full/checkpoint-50&quot;, &quot;base\_repo&quot;: &quot;Qwen/Qwen3-8B&quot;, &quot;base\_revision&quot;: &quot;b968826d9c46dd6066d109eabc6255188de91218&quot;, &quot;cuda\_runtime&quot;: &quot;13.0&quot;, &quot;gpu&quot;: &quot;NVIDIA RTX A5000&quot;, &quot;gpu\_memory\_bytes&quot;: 25285623808, &quot;identity\_log\_ts&quot;: &quot;2026-10-21T21:01:20.479741192Z&quot;, &quot;served\_adapter&quot;: &quot;envoy&quot;, &quot;torch\_version&quot;: &quot;2.13.0+cu130&quot;, &quot;vllm\_version&quot;: &quot;0.31.0&quot;, &quot;worker\_id&quot;: &quot;w0771kah7ysaxu&quot;\}

### original\_comparability

Verification: artifact_value_verified
Source SHA-256: 6169e7cf458406343756022590503d26936002374660f9c2038998798e5f9902
Reference: \{&quot;json\_pointer&quot;: &quot;/original\_assessment&quot;, &quot;relative\_path&quot;: &quot;serving-observations.json&quot;\}
> \{&quot;comparable&quot;: false, &quot;reasons&quot;: \[&quot;difference does not exceed the 0.4-second polling uncertainty margin&quot;\], &quot;rounds&quot;: \[\{&quot;absolute\_difference\_seconds&quot;: 0.0677962088957429, &quot;concurrent&quot;: 3.355291415937245, &quot;repeat&quot;: 1, &quot;serial&quot;: 3.423087624832988, &quot;serial\_over\_concurrent&quot;: 1.0202057587527924\}, \{&quot;absolute\_difference\_seconds&quot;: 0.9789846250787377, &quot;concurrent&quot;: 2.850205041002482, &quot;repeat&quot;: 2, &quot;serial&quot;: 3.82918966608122, &quot;serial\_over\_concurrent&quot;: 1.3434786659188585\}\]\}

## Linked research (unreviewed)

Note: measurement\_decision
Written against brief revision: 2
Question: Can two Qwen v5 reader requests share one A5000, and should we claim a latency win?
Data: recorded
Source checks prove provenance, not semantic support.
- Item 1:
  - claim: Keep one shared model with up to two requests; concurrent admission worked. Do not claim a speedup or choose another latency optimization from these two short-output rounds. The next useful decision needs representative reader evidence and answer checks, not more serving changes or automatic training.
  - reference ids:
    - completed\_serving\_run
    - observed\_overlap
    - original\_uncertainty
Uncertainties:
- Representative reader evidence quality remains unmeasured.
- Reliable latency benefit and all-in invoice cost remain unknown.
Proposed test (not executed): None
Research record SHA-256: 89e944f7ceeeb89bba5b5870f77ec954e0ffde3e9f5a6c784c7fc6432d606da4

## Planning status

Missing: No structural fields missing
Candidate outcomes already seen: yes
Author: agent_authored
Freezing records intent and grants no permission to execute or spend.
Frozen plan SHA-256: e0400b688e26407a7ccbe9e54c60507391d7db5845e9d57107473e5b70cda27c

## Recorded result

Run: recorded\_automatic\_profile
Data: recorded
Timing: retrospective
Criteria: inconclusive

- all\_timed\_calls\_completed: met (4.0 count); Compared against the frozen threshold
- same\_engine\_overlap: met (2.0 count); Compared against the frozen threshold
- resolved\_pair\_time\_nonregression: unknown (Unknown); baseline: measurement is missing or unknown

- Criteria compare supplied measurements; they prove no causality or adoption.
- Runtime identity, coverage, and controlled conditions remain caller declarations.
- Reviewer identity and timestamps are declarations, not authenticated attestations.
