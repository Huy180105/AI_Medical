from typing import Any


class FHIRDiagnosticReportBuilder:
    """
    Builds ICD-10-coded FHIR R4 Condition and DiagnosticReport resources representing CDSS diagnostic findings.
    """

    @staticmethod
    def build_condition(
        cond_id: str,
        disease_name: str,
        icd10_code: str = "R69",
        patient_id: str = "patient_001"
    ) -> dict[str, Any]:
        return {
            "resourceType": "Condition",
            "id": cond_id,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active"
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "provisional",
                        "display": "Provisional"
                    }
                ]
            },
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": icd10_code,
                        "display": disease_name
                    }
                ],
                "text": disease_name
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            }
        }

    @staticmethod
    def build_diagnostic_report(
        report_id: str,
        patient_id: str = "patient_001",
        conclusion: str = "",
        result_references: list[str] = None
    ) -> dict[str, Any]:
        references = [{"reference": ref} for ref in (result_references or [])]
        
        return {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11502-2",
                        "display": "Laboratory report"
                    }
                ],
                "text": "Medical AI CDSS Decision Report"
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "result": references,
            "conclusion": conclusion
        }
