from typing import Any


class FHIRMedicationBuilder:
    """
    Builds RxNorm-coded FHIR R4 MedicationRequest resources.
    """

    @staticmethod
    def build_medication_request(
        req_id: str,
        drug_name: str,
        patient_id: str = "patient_001",
        rxnorm_code: str = "0000",
        status: str = "active",
        note: str = ""
    ) -> dict[str, Any]:
        resource = {
            "resourceType": "MedicationRequest",
            "id": req_id,
            "status": status,
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": rxnorm_code,
                        "display": drug_name
                    }
                ],
                "text": drug_name
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            }
        }
        
        if note:
            resource["note"] = [{"text": note}]
            
        return resource
