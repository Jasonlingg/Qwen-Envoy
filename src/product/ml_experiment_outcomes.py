"""Append-only outcome persistence shared by the local experiment store."""

from __future__ import annotations

from datetime import datetime, timezone

from src.product.ml_experiment_compare import assess_result
from src.product.ml_experiment_io import decode, encode
from src.product.ml_experiment_models import ResultPointer
from src.product.ml_experiment_results import ExperimentResult, ResultRecord


class ExperimentOutcomes:
    """Uses the parent store's locked, content-addressed files and plan loader."""

    def _read_results(self, directory, index, plan):
        records = []
        for pointer in index.results:
            record = ResultRecord.model_validate(self.files.read_object(directory, pointer.sha256))
            if (record.experiment_id != index.experiment_id
                    or record.plan_sha256 != index.frozen_plan_sha256
                    or record.result.run_id != pointer.run_id):
                raise ValueError("Recorded outcome identity or plan binding mismatch")
            expected = assess_result(plan, record.result, record.evidence_receipts)
            if record.assessment != expected:
                raise ValueError("Recorded assessment differs from the frozen criteria")
            records.append((pointer.sha256, record))
        return records

    @staticmethod
    def _result_view(records, run_id=None):
        if run_id is not None and not any(r.result.run_id == run_id for _, r in records):
            raise ValueError("No recorded result has this run ID")
        selected = next(((sha, r) for sha, r in records if r.result.run_id == run_id),
                        records[-1] if records else None)
        return {"results": [{"run_id": r.result.run_id, "sha256": sha,
                             "data_kind": r.result.data_kind,
                             "recorded_at": r.recorded_at, "overall": r.assessment.overall,
                             "effective_timing": r.assessment.effective_timing}
                            for sha, r in records],
                "selected_result": ({**selected[1].model_dump(mode="json"),
                                     "sha256": selected[0]} if selected else None)}

    def record_result(self, experiment_id: str, plan_sha256: str, result: dict) -> dict:
        parsed = ExperimentResult.model_validate(decode(encode(result)))
        with self.files.locked() as directory:
            index, revisions, plan, records, research = self._load(directory, experiment_id)
            if plan is None or plan_sha256 != index.frozen_plan_sha256:
                raise ValueError("Result must bind to this experiment's exact frozen plan SHA-256")
            existing = next((r for _, r in records if r.result.run_id == parsed.run_id), None)
            if existing:
                if parsed != existing.result:
                    raise ValueError("Run ID already has a different immutable result")
                return self._view(index, revisions, plan, records, research, run_id=parsed.run_id)
            if len(records) >= 100:
                raise ValueError("Experiment has reached its 100-result limit")
            if parsed.references and self.evidence_resolver is None:
                raise ValueError("Result references require configured source verification")
            receipts = self.evidence_resolver(parsed.references) if self.evidence_resolver else []
            assessment = assess_result(plan, parsed, receipts)
            record = ResultRecord(experiment_id=experiment_id, plan_sha256=plan_sha256,
                                  recorded_at=datetime.now(timezone.utc).isoformat(),
                                  result=parsed, evidence_receipts=receipts, assessment=assessment)
            sha = self.files.object(directory, record.model_dump(mode="json"))
            index = index.model_copy(update={"results": [*index.results,
                ResultPointer(run_id=parsed.run_id, sha256=sha)]})
            self._write_index(directory, index)
            return self._view(index, revisions, plan, [*records, (sha, record)], research,
                              run_id=parsed.run_id)
