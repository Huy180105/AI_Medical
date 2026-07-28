from datetime import datetime, timezone
from typing import Any, Optional


class FHIRObservationBuilder:
    """
    Builds LOINC-coded FHIR R4 Observation resources for vital signs and physiological metrics.
    """

    LOINC_CODES = {
        "heart_rate": {
            "code": "8867-4",
            "display": "Heart rate",
            "unit": "beats/min",
            "unit_code": "/min"
        },
        "spo2": {
            "code": "59408-5",
            "display": "Oxygen saturation in Arterial blood by Pulse oximetry",
            "unit": "%",
            "unit_code": "%"
        },
        "skin_temp": {
            "code": "8310-5",
            "display": "Body temperature",
            "unit": "degC",
            "unit_code": "Cel"
        },
        "ecg": {
            "code": "13132-6",
            "display": "MDC_ECG_ELEC_POTL",
            "unit": "mV",
            "unit_code": "mV"
        }
    }

    @classmethod
    def build_observation(
        cls,
        obs_id: str,
        metric_name: str,
        value: float,
        patient_id: str = "patient_001",
        timestamp: Optional[float] = None
    ) -> dict[str, Any]:
        info = cls.LOINC_CODES.get(metric_name, {
            "code": "85353-1",
            "display": "Vital signs",
            "unit": "1",
            "unit_code": "1"
        })

        dt_str = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
            if timestamp
            else datetime.now(timezone.utc).isoformat()
        )

        return {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": info["code"],
                        "display": info["display"]
                    }
                ]
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "effectiveDateTime": dt_str,
            "valueQuantity": {
                "value": round(float(value), 2),
                "unit": info["unit"],
                "system": "http://unitsofmeasure.org",
                "code": info["unit_code"]
            }
        }
