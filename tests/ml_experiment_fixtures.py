"""Synthetic experiment plans; these numbers are not project measurements."""


def complete_brief():
    return {
        "question": "Would a smaller reader help this synthetic workload?",
        "goal": "model_comparison",
        "current_setup": {
            "task": {"value": "Answer synthetic document questions"},
            "model": {"value": "synthetic-baseline"},
            "workload": {"value": "Two synthetic questions"},
        },
        "proposed_change": "Replace the synthetic baseline with a smaller reader",
        "hypothesis": "The reader may reduce latency with acceptable quality",
        "expected_signal": "Lower elapsed time on the same two inputs",
        "applicability": "Both candidates support the same synthetic input format",
        "baseline": {"description": "Current synthetic reader"},
        "candidate": {"description": "Smaller synthetic reader"},
        "evaluation": {"population": "Synthetic questions", "purpose": "development",
                       "expected_cases": 2, "sample_ids": ["synthetic-1", "synthetic-2"]},
        "conditions": [{"name": "reader", "role": "proposed_change",
                        "baseline": {"value": "synthetic-baseline"},
                        "candidate": {"value": "synthetic-candidate"}}],
        "metrics": [{"metric_id": "latency", "unit": "milliseconds", "aggregation": "mean",
                     "population": "Synthetic questions", "direction": "lower"}],
        "criteria": [{"criterion_id": "speed", "metric_id": "latency",
                      "operation": "relative_improvement_gte", "threshold": 20.0}],
        "budget": {"max_model_requests": 0, "max_input_tokens": 0, "max_output_tokens": 0,
                   "max_gpu_seconds": 0.0, "max_elapsed_seconds": 60.0, "max_cost_usd": 0.0,
                   "stop_conditions": ["Stop if saved synthetic data is missing"]},
        "procedure": ["Compare the two saved synthetic measurements"],
        "limitations": ["Synthetic fixture, not a measured model result"],
        "candidate_outcomes_already_seen": "yes",
    }
