from typing import Any, Optional


class FHIRPatientBuilder:
    """
    Builds FHIR R4 Patient resources matching international HL7 standards.
    """

    @staticmethod
    def build_patient(
        patient_id: str = "patient_001",
        name: str = "Bệnh nhân thử nghiệm",
        gender: str = "unknown",
        birth_date: str = "1990-01-01",
        mrn: Optional[str] = None
    ) -> dict[str, Any]:
        mrn_value = mrn or f"MRN-{patient_id}"
        
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "identifier": [
                {
                    "use": "official",
                    "system": "http://hospital.smarthealth.vn/mrn",
                    "value": mrn_value
                }
            ],
            "active": True,
            "name": [
                {
                    "use": "official",
                    "text": name
                }
            ],
            "gender": gender.lower() if gender.lower() in ("male", "female", "other", "unknown") else "unknown",
            "birthDate": birth_date
        }
