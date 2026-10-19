"""Synthetic paired measurements; never measured model outcomes."""

from copy import deepcopy

from tests.ml_experiment_fixtures import complete_brief


def planned_brief():
    brief = complete_brief()
    brief["metrics"][0].update(measurement_kind="latency", timing_scope="warm")
    brief["evaluation"]["dataset_revision"] = {"value": "synthetic-v1"}
    for label in ("baseline", "candidate"):
        brief[label]["identity"] = {name: {"value": value} for name, value in {
            "model_revision": "synthetic-" + label, "workload": "synthetic-workload-v1",
            "hardware": "synthetic-cpu", "software_revision": "synthetic-code-v1",
            "prompt_sha256": "synthetic-prompt", "decoding": "synthetic-greedy",
        }.items()}
    return brief


def paired_result(brief=None, run_id="synthetic_run"):
    brief = brief or planned_brief()
    result = {"run_id": run_id, "timing": "retrospective", "data_kind": "synthetic"}
    for label, value in (("baseline", 100.0), ("candidate", 75.0)):
        result[label] = {"identity": deepcopy(brief[label]["identity"]),
                         "conditions": {c["name"]: deepcopy(c[label]) for c in brief["conditions"]},
                         "evaluation": deepcopy(brief["evaluation"]),
                         "measurements": [{"metric_id": "latency", "value": value,
                             "unit": "milliseconds", "aggregation": "mean",
                             "population": "Synthetic questions", "sample_count": 2,
                             "coverage": "complete", "timing_scope": "warm"}]}
    return result
