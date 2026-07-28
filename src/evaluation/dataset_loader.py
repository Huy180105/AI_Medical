from typing import Any


class EvaluationDatasetLoader:
    """
    Loads ground-truth datasets for NER entity extraction, CDSS clinical decisions,
    and synthetic wearable anomaly streams for systematic evaluation.
    """

    @staticmethod
    def get_ner_test_samples() -> list[dict[str, Any]]:
        """Returns annotated text samples and expected ground-truth entities."""
        return [
            {
                "text": "Bệnh nhân sốt cao và ho kéo dài, bác sĩ kê đơn Paracetamol.",
                "expected_entities": [
                    {"text": "sốt cao", "type": "SYMPTOM"},
                    {"text": "ho kéo dài", "type": "SYMPTOM"},
                    {"text": "Paracetamol", "type": "MEDICINE"}
                ]
            },
            {
                "text": "Khó thở và đau ngực dữ dội, nghi ngờ nhồi máu cơ tim, cần chụp X-quang ngực.",
                "expected_entities": [
                    {"text": "Khó thở", "type": "SYMPTOM"},
                    {"text": "đau ngực", "type": "SYMPTOM"},
                    {"text": "nhồi máu cơ tim", "type": "DISEASE"},
                    {"text": "chụp X-quang ngực", "type": "TEST"}
                ]
            },
            {
                "text": "Bệnh nhân bị suy thận cấp, chống chỉ định dùng Ibuprofen.",
                "expected_entities": [
                    {"text": "suy thận cấp", "type": "DISEASE"},
                    {"text": "Ibuprofen", "type": "MEDICINE"}
                ]
            }
        ]

    @staticmethod
    def get_cdss_test_samples() -> list[dict[str, Any]]:
        """Returns CDSS test cases with ground-truth expected risk levels and safety rules."""
        return [
            {
                "id": "cdss_case_01",
                "text": "Khó thở nặng và SpO2 84%, tiền sử suy thận.",
                "entities": [
                    {"text": "khó thở", "type": "SYMPTOM"},
                    {"text": "suy thận", "type": "DISEASE"}
                ],
                "expected_risk": "Emergency",
                "expected_labs": ["Renal function test", "Chest X-ray"],
                "contraindicated_drugs": ["Ibuprofen"]
            },
            {
                "id": "cdss_case_02",
                "text": "Sốt nhẹ 37.8 độ C, đau đầu nhẹ.",
                "entities": [
                    {"text": "sốt", "type": "SYMPTOM"},
                    {"text": "đau đầu", "type": "SYMPTOM"}
                ],
                "expected_risk": "Low",
                "expected_labs": [],
                "indicated_drugs": ["Paracetamol"]
            }
        ]

    @staticmethod
    def get_wearable_test_samples() -> list[dict[str, Any]]:
        """Returns wearable telemetry samples with ground-truth anomaly classifications."""
        return [
            {
                "state": "Hypoxia",
                "telemetry": {"heart_rate": 105.0, "spo2": 84.0, "skin_temp": 36.6},
                "expected_anomaly": "Hypoxia",
                "expected_severity": "Emergency"
            },
            {
                "state": "Arrhythmia",
                "telemetry": {"heart_rate": 130.0, "spo2": 97.0, "skin_temp": 36.6},
                "expected_anomaly": "Tachycardia",
                "expected_severity": "High"
            },
            {
                "state": "Fever",
                "telemetry": {"heart_rate": 100.0, "spo2": 96.5, "skin_temp": 39.2},
                "expected_anomaly": "Fever Spike",
                "expected_severity": "High"
            },
            {
                "state": "Normal",
                "telemetry": {"heart_rate": 70.0, "spo2": 98.0, "skin_temp": 36.6},
                "expected_anomaly": None,
                "expected_severity": None
            }
        ]
