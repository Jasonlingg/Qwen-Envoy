"""Bind measurements to selected artifact content without upgrading caller declarations."""

from __future__ import annotations

from src.product.ml_experiment_measurements import Measurement


def measurement_provenance(result, receipts) -> dict[str, str]:
    by_id = {receipt.reference_id: receipt for receipt in receipts}
    provenance = {}
    for label in ("baseline", "candidate"):
        for measurement in getattr(result, label).measurements:
            key = f"{label}.{measurement.metric_id}"
            provenance[key] = "caller_reported"
            if measurement.reference_id is None:
                continue
            receipt = by_id.get(measurement.reference_id)
            if receipt is None or receipt.verification != "artifact_value_verified":
                raise ValueError("A measurement must reference a verified local artifact value")
            supplied = receipt.value
            declared = measurement.model_dump(mode="json")
            if isinstance(supplied, dict):
                if "value" not in supplied or set(supplied) - set(declared):
                    raise ValueError("Selected artifact needs measurement fields including value")
                source = {**supplied, "reference_id": measurement.reference_id}
                checked = Measurement.model_validate({**declared, **source})
                if checked != measurement:
                    raise ValueError("Measurement differs from its selected artifact")
                required = set(declared) - {"reference_id"}
                provenance[key] = ("artifact_measurement_verified" if required <= set(supplied)
                                   else "artifact_value_only")
            elif (supplied is None and measurement.value is None
                  or type(supplied) in (int, float) and supplied == measurement.value):
                provenance[key] = "artifact_value_only"
            else:
                raise ValueError("Measurement value differs from its selected artifact")
    return provenance
