import os
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock

from src.agents.context import AgentContext
from src.agents.executor import AgentExecutor
from src.agents.triage_agent import TriageTimelineAgent
from src.agents.lab_vital_agent import VitalLabDynamicsAgent
from src.agents.cxr_agent import CXRDiagnosticAgent
from src.agents.temporal_guard import TemporalLeakageGuardAgent
from src.agents.multimodal_orchestrator import MultimodalConsensusOrchestrator

from src.multimodal.temporal_aligner import TemporalAligner, ClinicalEvent
from src.multimodal.dataset_builder import (
    MedFusionDatasetBuilder,
    CXRImageRecord,
    VitalRecord,
    SymptomRecord,
    TargetLabel
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_FIXTURE_DIR = os.path.join(BASE_DIR, "tests", "fixtures", "synthetic_cxr")


def test_temporal_aligner_strict_security():
    aligner = TemporalAligner(buffer_minutes=0.0, require_available_time=True, strict_subject_check=True)
    anchor_time = datetime(2026, 10, 15, 12, 0, 0)
    expected_patient = "SUBJ_1001"

    # Case 1: Valid event (measured and available before anchor, correct patient)
    ev_valid = ClinicalEvent(
        event_id="EV_01",
        event_type="vital",
        event_time=datetime(2026, 10, 15, 10, 30, 0),
        available_time=datetime(2026, 10, 15, 10, 35, 0),
        data={"heart_rate": 88},
        source_table="vitals",
        subject_id=expected_patient
    )

    # Case 2: Subject mismatch (wrong patient data injected)
    ev_wrong_patient = ClinicalEvent(
        event_id="EV_02",
        event_type="vital",
        event_time=datetime(2026, 10, 15, 10, 30, 0),
        available_time=datetime(2026, 10, 15, 10, 35, 0),
        data={"heart_rate": 140},
        source_table="vitals",
        subject_id="SUBJ_9999"
    )

    # Case 3: Missing event_time
    ev_missing_time = ClinicalEvent(
        event_id="EV_03",
        event_type="vital",
        event_time=None,
        available_time=datetime(2026, 10, 15, 10, 0, 0),
        data={"heart_rate": 90},
        source_table="vitals",
        subject_id=expected_patient
    )

    # Case 4: Future event (drawn after anchor)
    ev_future = ClinicalEvent(
        event_id="EV_04",
        event_type="note",
        event_time=datetime(2026, 10, 15, 14, 0, 0),
        available_time=datetime(2026, 10, 15, 14, 0, 0),
        data={"text": "Discharge summary"},
        source_table="notes",
        subject_id=expected_patient
    )

    # Case 5: Retrospective result availability leakage (drawn at 11:30, but lab machine finished at 13:00)
    ev_retrospective_result = ClinicalEvent(
        event_id="EV_05",
        event_type="lab",
        event_time=datetime(2026, 10, 15, 11, 30, 0),  # Prior
        available_time=datetime(2026, 10, 15, 13, 0, 0),  # Future!
        data={"troponin": 0.15},
        source_table="labs",
        subject_id=expected_patient
    )

    # Case 6: Timezone mismatch safe normalization
    ev_tz_aware = ClinicalEvent(
        event_id="EV_06",
        event_type="vital",
        event_time=datetime(2026, 10, 15, 11, 0, 0, tzinfo=timezone.utc),
        available_time=datetime(2026, 10, 15, 11, 0, 0, tzinfo=timezone.utc),
        data={"spo2": 98},
        source_table="vitals",
        subject_id=expected_patient
    )

    all_events = [ev_valid, ev_wrong_patient, ev_missing_time, ev_future, ev_retrospective_result, ev_tz_aware]
    result = aligner.filter_events_by_anchor(anchor_time, all_events, subject_id=expected_patient)

    # Only EV_01 and EV_06 should be valid
    valid_ids = [e.event_id for e in result.valid_events]
    assert valid_ids == ["EV_01", "EV_06"]

    # Leaked events should contain EV_04 and EV_05
    leaked_ids = [e.event_id for e in result.leaked_events]
    assert "EV_04" in leaked_ids
    assert "EV_05" in leaked_ids

    # Rejected list must contain wrong patient and missing time
    rejected_reasons = {r["event_id"]: r["reason"] for r in result.rejected_events}
    assert "REJECTED_SUBJECT_MISMATCH" in rejected_reasons["EV_02"]
    assert "REJECTED_MISSING_EVENT_TIME" in rejected_reasons["EV_03"]


def test_dataset_builder_provenance_and_three_tier_availability():
    builder = MedFusionDatasetBuilder(buffer_minutes=0.0, strict_mode=True)
    anchor = datetime(2026, 10, 15, 8, 30, 0)

    # Sample A: No CXR image
    sample_a = builder.create_cohort_sample(
        subject_id="SUBJ_001",
        anchor_time=anchor,
        hadm_id="HADM_999",
        image=None
    )
    avail_a = sample_a.compute_modality_availability()
    assert avail_a.image_has_record is False
    assert avail_a.image_has_valid_data is False

    # Sample B: With synthetic test fixture -> Must NOT pass as valid clinical image!
    synthetic_image_path = os.path.join(SYNTHETIC_FIXTURE_DIR, "synthetic_pneumonia.png")
    img_synthetic = CXRImageRecord(
        study_id="S_TEST",
        dicom_id="D_TEST",
        study_time=anchor.isoformat(),
        view_position="PA",
        image_path=synthetic_image_path,
        source_dataset="SYNTHETIC_FIXTURE_TEST_ONLY",
        linkage_status="SYNTHETIC_FIXTURE",
        linkage_verified=False,
        is_synthetic=True
    )
    sample_b = builder.create_cohort_sample(
        subject_id="SUBJ_002",
        anchor_time=anchor,
        image=img_synthetic
    )
    avail_b = sample_b.compute_modality_availability()
    assert avail_b.image_has_record is True
    # Crucial assertion: synthetic fixture is explicitly quarantined
    assert avail_b.image_has_valid_data is False
    assert avail_b.image_has_valid_at_anchor is False


def test_cxr_agent_strict_status_codes():
    agent = CXRDiagnosticAgent()

    # Case 1: NOT_PROVIDED
    ctx_no_img = AgentContext(text="Bệnh nhân khó thở")
    ctx_no_img = agent.process(ctx_no_img)
    assert ctx_no_img.metadata["cxr_status"] == "NOT_PROVIDED"
    assert ctx_no_img.metadata["cxr_available"] is False

    # Case 2: INVALID_INPUT (corrupt or non-existent file)
    ctx_invalid = AgentContext(text="Bệnh nhân khó thở")
    ctx_invalid.metadata["image_path"] = "non_existent_image_path_12345.png"
    ctx_invalid = agent.process(ctx_invalid)
    assert ctx_invalid.metadata["cxr_status"] == "INVALID_INPUT"
    assert ctx_invalid.metadata["cxr_available"] is False

    # Case 3: SYNTHETIC_FIXTURE warning
    synthetic_path = os.path.join(SYNTHETIC_FIXTURE_DIR, "synthetic_pneumonia.png")
    if os.path.exists(synthetic_path):
        ctx_synth = AgentContext(text="Bệnh nhân khó thở")
        ctx_synth.metadata["image_path"] = synthetic_path
        ctx_synth = agent.process(ctx_synth)
        assert ctx_synth.metadata["cxr_is_synthetic"] is True
        assert "SYNTHETIC_FIXTURE_WARNING" in ctx_synth.metadata["cxr_is_synthetic_warning"]


def test_triage_timeline_agent():
    agent = TriageTimelineAgent()
    context = AgentContext(text="Bệnh nhân bị sốt cao kèm khó thở 3 ngày nay, đau ngực nhẹ.")
    context.metadata["prediction_anchor_time"] = "2026-10-15T12:00:00"

    context = agent.process(context)
    structured = context.metadata.get("structured_symptoms", [])

    assert len(structured) >= 2
    assert any("khó thở" in s["symptom_name"].lower() for s in structured)
    assert context.metadata["chief_duration_hours"] == 72.0


def test_vital_lab_dynamics_agent():
    agent = VitalLabDynamicsAgent()
    context = AgentContext(text="Cấp cứu tiếp nhận")
    context.metadata["vitals"] = {
        "heart_rate": 115.0,
        "respiratory_rate": 24.0,  # qSOFA +1
        "systolic_bp": 95.0,       # qSOFA +1
        "spo2": 89.0,              # Severe hypoxemia
    }
    context.metadata["labs"] = {
        "wbc": 16.2,
        "crp": 45.0
    }

    context = agent.process(context)
    vital_dyn = context.metadata.get("vital_dynamics", {})

    assert vital_dyn["qsofa_score"] >= 2
    assert vital_dyn["hypoxemia_level"] == "Severe Hypoxemia"
    assert context.metadata["physiological_risk"] == "Critical / High"
    assert any("Shock Index" in a for a in vital_dyn["active_alerts"])


def test_temporal_guard_agent():
    guard = TemporalLeakageGuardAgent()
    context = AgentContext(text="Bệnh nhân nhập viện cấp cứu vì ho ra máu.")
    context.metadata["prediction_anchor_time"] = "2026-10-15T12:00:00"
    context.metadata["clinical_events"] = [
        {"id": "1", "timestamp": "2026-10-15T10:00:00", "type": "vital"},
        {"id": "2", "timestamp": "2026-10-15T18:00:00", "type": "discharge"},  # Leaked!
    ]

    context = guard.process(context)
    audit = context.metadata.get("temporal_audit", {})
    assert audit["status"] == "LEAKAGE_DETECTED"
    assert audit["leaked_event_count"] == 1
    assert len(context.metadata["clinical_events"]) == 1


def test_multimodal_consensus_pipeline_with_unlinked_cxr():
    """
    Verifies that when a patient has clinical vitals/symptoms but NO CXR,
    the pipeline executes safely without falsely claiming normal lung findings.
    """
    agents = [
        TemporalLeakageGuardAgent(),
        TriageTimelineAgent(),
        VitalLabDynamicsAgent(),
        CXRDiagnosticAgent(),
        MultimodalConsensusOrchestrator()
    ]
    executor = AgentExecutor(agents)

    context = AgentContext(text="Bệnh nhân sốt cao, ho và khó thở 2 ngày nay.")
    context.metadata["prediction_anchor_time"] = "2026-10-15T10:00:00"
    # No image_path provided (typical of ED triage before X-ray order)
    context.metadata["vitals"] = {
        "heart_rate": 105.0,
        "respiratory_rate": 22.0,
        "systolic_bp": 110.0,
        "spo2": 93.0
    }
    context.metadata["labs"] = {
        "wbc": 14.8,
        "crp": 38.0
    }

    pipeline = ["temporal_guard", "triage_agent", "lab_vital_agent", "cxr_agent", "multimodal_orchestrator"]
    context = executor.execute_pipeline(pipeline, context)

    assert context.response is not None
    decision = context.clinical_decision
    assert decision["modalities_used"]["cxr_imaging"] is False
    assert decision["modalities_used"]["symptoms_history"] is True
    assert decision["modalities_used"]["vital_signs"] is True
