**Defer adoption.** Quality is met on the supplied population; latency and cost remain unknown. Because all gates are required, the saved results do not establish eligibility for adoption.

| Original gate | Status | Observation and conclusion |
|---|---|---|
| Quality: candidate supported-answer rate no more than 1 percentage point below baseline | **Met** | Baseline: 96/100 = **96%**. Candidate: 97/100 = **97%**. The candidate is 1 percentage point higher on the same source-reviewed cases and dataset revision. This satisfies the gate for this population; it does not establish performance on unseen cases. |
| Latency: at least 20% lower p95 end-to-end latency on the same complete workload under matched conditions | **Unknown** | Reported p95 falls from 5.0 to 3.8 seconds, an apparent **24% reduction**. But baseline timing covers only 80 cases versus the candidate’s 100, omitting 20 baseline failures. Cache, warm/cold state and load conditions are unrecorded. This is not a complete matched comparison. |
| Cost: all-in allocated cost ≤ **$0.020/request**, including host and lifecycle overhead | **Unknown** | Candidate GPU-active-only cost is **$0.014/request**. At that active charge, only **$0.006/request** remains for loading, idle, cleanup, storage and host inference. Those costs are unknown, so the all-in threshold is unresolved. Baseline’s GPU-active-only figure does not establish an all-in comparison either. |

No gate is demonstrably **not met** from these results. Unknown gates still prevent adoption under the frozen rule.

The smallest useful next measurement is to complete the missing latency evidence and candidate cost accounting, retaining the existing quality result if the evaluated artifacts remain unchanged.

1. **First recover existing records**, if available: the 20 missing baseline terminal timings, condition metadata for both systems, and candidate overhead accounting. Recovery is sufficient only if it establishes a complete, matched comparison.
2. **If recovery cannot do that, perform one matched replay of both systems on the same complete 100-case workload**, with candidate all-in cost accounting attached. Keep artifacts and request settings fixed; match and record cache state, warm/cold conditions and offered load. Preserve all failures and measure every request through its terminal outcome. Recover the original timeout/failure treatment and p95 estimator before scoring; do not silently exclude failures or introduce favorable handling.

**Metrics and decision rule:** retain supported-answer rates of 96% and 97%; measure complete-workload end-to-end p95 for each system, attempted/terminal/failure counts, and candidate all-in allocated dollars per request. Adopt only if candidate p95 is **≤0.80 × matched baseline p95**, candidate all-in cost is **≤$0.020/request**, and the quality gate remains applicable and met. Do not add significance requirements or relax thresholds.

**Unknown setup details:** model/runtime configuration, original latency estimator and failure treatment, deployment accounting window, request volume and cost-allocation method. Cost measurement must cover lifecycle overhead and allocate shared expenses over a documented workload and period; a short replay alone may not resolve deployment cost.

**Stop condition:** stop once one valid comparison and complete accounting resolve the gates. A failed gate means do not adopt; missing evidence means continue deferral. Do not repeat runs merely to obtain passing numbers.

*Source: supplied MLP05 synthetic historical results and frozen rule, unversioned. No external facts were needed; no external sources were used and no experiments were executed.*