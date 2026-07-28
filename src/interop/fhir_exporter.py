import uuid
from datetime import datetime, timezone
from typing import Any
from src.interop.patient import FHIRPatientBuilder
from src.interop.observation import FHIRObservationBuilder
from src.interop.medication import FHIRMedicationBuilder
from src.interop.diagnostic_report import FHIRDiagnosticReportBuilder


class FHIRExporter:
    """
    Orchestrates the transformation of internal clinical decision states and watch sensor telemetry
    into a standardized FHIR R4 JSON Bundle object for hospital interoperability.
    """

    @classmethod
    def export_bundle(cls, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Parses payload state and returns a complete FHIR R4 Collection Bundle.
        """
        patient_id = str(payload.get("patient_id") or payload.get("session_id") or "patient_001")
        telemetry = payload.get("telemetry", {})
        anomalies = payload.get("anomalies", [])
        risk_level = payload.get("risk_level", "Low")
        
        cdss = payload.get("cdss_alert") or payload.get("clinical_decision") or {}
        recs = payload.get("recommendations") or cdss.get("recommendations", {})
        candidates = cdss.get("diagnosis_candidates", [])
        
        entries = []
        result_references = []

        # 1. Patient Resource
        patient_res = FHIRPatientBuilder.build_patient(patient_id=patient_id)
        entries.append({
            "fullUrl": f"Patient/{patient_id}",
            "resource": patient_res
        })

        # 2. Vital Sign Observations
        ts = telemetry.get("timestamp")
        if "heart_rate" in telemetry:
            obs_hr_id = f"obs_hr_{uuid.uuid4().hex[:8]}"
            obs_hr = FHIRObservationBuilder.build_observation(
                obs_id=obs_hr_id,
                metric_name="heart_rate",
                value=telemetry["heart_rate"],
                patient_id=patient_id,
                timestamp=ts
            )
            entries.append({"fullUrl": f"Observation/{obs_hr_id}", "resource": obs_hr})
            result_references.append(f"Observation/{obs_hr_id}")

        if "spo2" in telemetry:
            obs_spo2_id = f"obs_spo2_{uuid.uuid4().hex[:8]}"
            obs_spo2 = FHIRObservationBuilder.build_observation(
                obs_id=obs_spo2_id,
                metric_name="spo2",
                value=telemetry["spo2"],
                patient_id=patient_id,
                timestamp=ts
            )
            entries.append({"fullUrl": f"Observation/{obs_spo2_id}", "resource": obs_spo2})
            result_references.append(f"Observation/{obs_spo2_id}")

        if "skin_temp" in telemetry:
            obs_temp_id = f"obs_temp_{uuid.uuid4().hex[:8]}"
            obs_temp = FHIRObservationBuilder.build_observation(
                obs_id=obs_temp_id,
                metric_name="skin_temp",
                value=telemetry["skin_temp"],
                patient_id=patient_id,
                timestamp=ts
            )
            entries.append({"fullUrl": f"Observation/{obs_temp_id}", "resource": obs_temp})
            result_references.append(f"Observation/{obs_temp_id}")

        # 3. Diagnosis Conditions (ICD-10)
        for idx, cand in enumerate(candidates):
            disease = cand.get("disease", "Condition")
            icd_code = cand.get("icd10_code") or "R69"
            cond_id = f"cond_{idx}_{uuid.uuid4().hex[:6]}"
            
            cond_res = FHIRDiagnosticReportBuilder.build_condition(
                cond_id=cond_id,
                disease_name=disease,
                icd10_code=icd_code,
                patient_id=patient_id
            )
            entries.append({"fullUrl": f"Condition/{cond_id}", "resource": cond_res})
            result_references.append(f"Condition/{cond_id}")

        # 4. Medication Requests (RxNorm)
        med_cats = recs.get("recommended_medication_categories", [])
        for idx, med in enumerate(med_cats):
            cat_name = med.get("category", "")
            status = "active" if med.get("status") != "Contraindicated" else "cancelled"
            note = med.get("note", "")
            med_id = f"med_req_{idx}_{uuid.uuid4().hex[:6]}"
            
            med_res = FHIRMedicationBuilder.build_medication_request(
                req_id=med_id,
                drug_name=cat_name,
                patient_id=patient_id,
                status=status,
                note=note
            )
            entries.append({"fullUrl": f"MedicationRequest/{med_id}", "resource": med_res})

        # 5. Diagnostic Report Summary
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        conclusion = (
            f"Risk Level: {risk_level}. Active Anomalies: {len(anomalies)}. "
            f"Referral: {recs.get('referral_suggestion', 'Routine follow-up')}."
        )
        report_res = FHIRDiagnosticReportBuilder.build_diagnostic_report(
            report_id=report_id,
            patient_id=patient_id,
            conclusion=conclusion,
            result_references=result_references
        )
        entries.append({"fullUrl": f"DiagnosticReport/{report_id}", "resource": report_res})

        # Assemble FHIR Bundle
        bundle_id = f"bundle_{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        
        return {
            "resourceType": "Bundle",
            "id": bundle_id,
            "type": "collection",
            "timestamp": now_utc,
            "entry": entries
        }
