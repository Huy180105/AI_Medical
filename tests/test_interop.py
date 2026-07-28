import pytest
from fastapi.testclient import TestClient

from src.interop.patient import FHIRPatientBuilder
from src.interop.observation import FHIRObservationBuilder
from src.interop.medication import FHIRMedicationBuilder
from src.interop.diagnostic_report import FHIRDiagnosticReportBuilder
from src.interop.fhir_exporter import FHIRExporter
from src.api.app import create_app


def test_fhir_patient_builder():
    pat = FHIRPatientBuilder.build_patient(
        patient_id="patient_123",
        name="Nguyen Van A",
        gender="male",
        birth_date="1992-04-12",
        mrn="MRN-9999"
    )
    assert pat["resourceType"] == "Patient"
    assert pat["id"] == "patient_123"
    assert pat["gender"] == "male"
    assert pat["identifier"][0]["value"] == "MRN-9999"


def test_fhir_observation_builder():
    # Heart rate LOINC
    obs_hr = FHIRObservationBuilder.build_observation("obs_1", "heart_rate", 128.0, "patient_123")
    assert obs_hr["resourceType"] == "Observation"
    assert obs_hr["code"]["coding"][0]["code"] == "8867-4"
    assert obs_hr["valueQuantity"]["value"] == 128.0

    # SpO2 LOINC
    obs_spo2 = FHIRObservationBuilder.build_observation("obs_2", "spo2", 89.0, "patient_123")
    assert obs_spo2["code"]["coding"][0]["code"] == "59408-5"

    # Temp LOINC
    obs_temp = FHIRObservationBuilder.build_observation("obs_3", "skin_temp", 38.5, "patient_123")
    assert obs_temp["code"]["coding"][0]["code"] == "8310-5"


def test_fhir_medication_builder():
    med = FHIRMedicationBuilder.build_medication_request(
        req_id="med_1",
        drug_name="Paracetamol",
        patient_id="patient_123",
        rxnorm_code="161",
        status="active"
    )
    assert med["resourceType"] == "MedicationRequest"
    assert med["medicationCodeableConcept"]["coding"][0]["code"] == "161"
    assert med["status"] == "active"


def test_fhir_diagnostic_report_builder():
    cond = FHIRDiagnosticReportBuilder.build_condition("cond_1", "Pneumonia", "J18.9", "patient_123")
    assert cond["resourceType"] == "Condition"
    assert cond["code"]["coding"][0]["code"] == "J18.9"

    report = FHIRDiagnosticReportBuilder.build_diagnostic_report("rep_1", "patient_123", "High risk")
    assert report["resourceType"] == "DiagnosticReport"
    assert report["conclusion"] == "High risk"


def test_fhir_exporter_bundle():
    payload = {
        "patient_id": "patient_123",
        "telemetry": {"heart_rate": 128.0, "spo2": 89.0, "skin_temp": 38.5},
        "anomalies": [{"type": "Hypoxia"}],
        "risk_level": "Emergency",
        "clinical_decision": {
            "diagnosis_candidates": [{"disease": "Pneumonia", "icd10_code": "J18.9"}],
            "recommendations": {
                "recommended_medication_categories": [{"category": "Paracetamol", "status": "Indicated"}]
            }
        }
    }
    
    bundle = FHIRExporter.export_bundle(payload)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) >= 5
    
    # Assert resourceTypes inside bundle
    resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in resource_types
    assert "Observation" in resource_types
    assert "Condition" in resource_types
    assert "MedicationRequest" in resource_types
    assert "DiagnosticReport" in resource_types


def test_fhir_api_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. Get Patient
    res_pat = client.get("/fhir/patient/test_pat_99")
    assert res_pat.status_code == 200
    assert res_pat.json()["resourceType"] == "Patient"
    assert res_pat.json()["id"] == "test_pat_99"

    # 2. Export Bundle
    payload = {
        "patient_id": "test_pat_99",
        "telemetry": {"heart_rate": 100.0},
        "risk_level": "Medium"
    }
    res_bundle = client.post("/fhir/bundle", json=payload)
    assert res_bundle.status_code == 200
    assert res_bundle.json()["resourceType"] == "Bundle"
    assert res_bundle.json()["type"] == "collection"
