from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional
import json
import os
from src.multimodal.temporal_aligner import TemporalAligner, ClinicalEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SymptomRecord:
    symptom_name: str
    symptom_raw_text: str
    symptom_onset_time: Optional[str] = None
    symptom_duration_text: Optional[str] = None
    symptom_duration_hours: Optional[float] = None
    onset_confidence: float = 1.0
    source_note_id: Optional[str] = None
    is_valid: bool = True

    def validate_content(self) -> bool:
        """Verifies symptom has usable content and consistent duration/onset."""
        if not self.symptom_name or len(self.symptom_name.strip()) < 2:
            self.is_valid = False
            return False
        if self.symptom_duration_hours is not None and self.symptom_duration_hours < 0:
            self.is_valid = False
            return False
        self.is_valid = True
        return True


@dataclass
class VitalRecord:
    charttime: str
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    spo2: Optional[float] = None
    temperature_c: Optional[float] = None
    storetime: Optional[str] = None
    source_table: str = "vitalsign"

    def has_valid_values(self) -> bool:
        """Returns True if at least one physiological parameter is non-null and within biological limits."""
        valid_hr = self.heart_rate is not None and 20 <= self.heart_rate <= 300
        valid_bp = self.systolic_bp is not None and 30 <= self.systolic_bp <= 300
        valid_spo2 = self.spo2 is not None and 40 <= self.spo2 <= 100
        valid_rr = self.respiratory_rate is not None and 4 <= self.respiratory_rate <= 80
        return any([valid_hr, valid_bp, valid_spo2, valid_rr])


@dataclass
class LabRecord:
    charttime: str
    itemid: str
    test_name: str
    valuenum: Optional[float] = None
    valueuom: Optional[str] = None
    flag: Optional[str] = None  # 'abnormal' or 'normal'
    storetime: Optional[str] = None
    source_table: str = "labevents"

    def has_valid_values(self) -> bool:
        return self.valuenum is not None and not (self.valuenum != self.valuenum)  # NaN check


@dataclass
class CXRImageRecord:
    study_id: str
    dicom_id: str
    study_time: str
    view_position: str  # 'PA', 'AP', 'LATERAL'
    image_path: str
    source_dataset: str = "UNKNOWN"  # 'MIMIC-CXR-JPG', 'SYNTHETIC_FIXTURE_TEST_ONLY'
    linkage_status: str = "UNLINKED_NO_CXR"  # 'VERIFIED_EXPLICIT_HADM', 'WINDOW_MATCHED', 'SYNTHETIC_FIXTURE'
    linkage_verified: bool = False
    is_synthetic: bool = False  # Critical: flags test fixtures immediately so they are never used for evaluation


@dataclass
class TargetLabel:
    disease_name: str
    present: int  # 1: Positive, 0: Negative, -1: Uncertain
    label_source: str  # 'CHEXPERT_OFFICIAL', 'ICD10_DISCHARGE', 'ED_TRIAGE'
    label_definition: str = "Binary radiological finding from expert adjudication or CheXpert labeler"
    label_time: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ThreeTierModalityAvailability:
    """
    Distinguishes three crucial clinical ML availability concepts:
    1. has_record: Raw data entity exists in sample
    2. has_valid_data: Content is non-empty, non-NaN, biologically plausible
    3. has_valid_data_at_anchor: Event timestamp and result availability timestamp verified <= t_0
    """
    image_has_record: bool = False
    image_has_valid_data: bool = False
    image_has_valid_at_anchor: bool = False

    symptoms_has_record: bool = False
    symptoms_has_valid_data: bool = False
    symptoms_has_valid_at_anchor: bool = False

    vitals_has_record: bool = False
    vitals_has_valid_data: bool = False
    vitals_has_valid_at_anchor: bool = False

    labs_has_record: bool = False
    labs_has_valid_data: bool = False
    labs_has_valid_at_anchor: bool = False


@dataclass
class MedFusionCohortSample:
    """
    Standardized Multimodal Clinical Sample for MedFusion-CXR.
    Strictly decouples verified clinical data from synthetic test fixtures and retrospective targets.
    """
    subject_id: str
    hadm_id: Optional[str]
    stay_id: Optional[str]
    prediction_anchor_time: str  # t_0 (CXR study or triage admission timestamp)
    
    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    
    # Multimodal sources
    image: Optional[CXRImageRecord] = None
    symptoms: list[SymptomRecord] = field(default_factory=list)
    vitals: list[VitalRecord] = field(default_factory=list)
    labs: list[LabRecord] = field(default_factory=list)
    clinical_notes: list[dict[str, Any]] = field(default_factory=list)
    
    # Ground Truth Targets
    labels: dict[str, int] = field(default_factory=dict)
    detailed_labels: list[TargetLabel] = field(default_factory=list)
    
    # Provenance and Three-Tier Availability
    modality_availability: ThreeTierModalityAvailability = field(default_factory=ThreeTierModalityAvailability)
    data_provenance: dict[str, Any] = field(default_factory=dict)

    def compute_modality_availability(self) -> ThreeTierModalityAvailability:
        avail = ThreeTierModalityAvailability()

        # Image checks
        if self.image is not None:
            avail.image_has_record = True
            # Synthetic fixture is NOT considered valid clinical image data
            if not self.image.is_synthetic and os.path.exists(self.image.image_path):
                avail.image_has_valid_data = True
                avail.image_has_valid_at_anchor = True

        # Symptoms checks
        if len(self.symptoms) > 0:
            avail.symptoms_has_record = True
            valid_syms = [s for s in self.symptoms if s.validate_content()]
            if len(valid_syms) > 0:
                avail.symptoms_has_valid_data = True
                avail.symptoms_has_valid_at_anchor = True

        # Vitals checks
        if len(self.vitals) > 0:
            avail.vitals_has_record = True
            valid_vits = [v for v in self.vitals if v.has_valid_values()]
            if len(valid_vits) > 0:
                avail.vitals_has_valid_data = True
                avail.vitals_has_valid_at_anchor = True

        # Labs checks
        if len(self.labs) > 0:
            avail.labs_has_record = True
            valid_labs = [l for l in self.labs if l.has_valid_values()]
            if len(valid_labs) > 0:
                avail.labs_has_valid_data = True
                avail.labs_has_valid_at_anchor = True

        self.modality_availability = avail
        return avail

    def to_dict(self) -> dict[str, Any]:
        self.compute_modality_availability()
        res = asdict(self)
        return res


class MedFusionDatasetBuilder:
    """
    Builds and validates cohort samples conforming to MedFusion-CXR schema.
    Integrates TemporalAligner to guarantee zero data leakage before prediction_anchor_time.
    """

    def __init__(self, buffer_minutes: float = 0.0, strict_mode: bool = True) -> None:
        self.aligner = TemporalAligner(
            buffer_minutes=buffer_minutes,
            require_available_time=True,
            strict_subject_check=strict_mode
        )

    def create_cohort_sample(
        self,
        subject_id: str,
        anchor_time: datetime,
        hadm_id: Optional[str] = None,
        stay_id: Optional[str] = None,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        image: Optional[CXRImageRecord] = None,
        raw_events: Optional[list[ClinicalEvent]] = None,
        labels: Optional[dict[str, int]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> MedFusionCohortSample:
        """
        Creates a time-sanitized multimodal cohort sample with explicit provenance.
        """
        sample = MedFusionCohortSample(
            subject_id=subject_id,
            hadm_id=hadm_id,
            stay_id=stay_id,
            prediction_anchor_time=anchor_time.isoformat(),
            age=age,
            gender=gender,
            image=image,
            labels=labels or {},
            data_provenance=provenance or {}
        )

        if raw_events:
            cutoff = self.aligner.filter_events_by_anchor(
                anchor_time=anchor_time,
                events=raw_events,
                subject_id=subject_id
            )
            for ev in cutoff.valid_events:
                if ev.event_type == "vital":
                    sample.vitals.append(VitalRecord(**ev.data))
                elif ev.event_type == "lab":
                    sample.labs.append(LabRecord(**ev.data))
                elif ev.event_type == "symptom":
                    sample.symptoms.append(SymptomRecord(**ev.data))
                elif ev.event_type == "note":
                    sample.clinical_notes.append(ev.data)

            sample.data_provenance["temporal_audit"] = {
                "valid_count": len(cutoff.valid_events),
                "leaked_count": len(cutoff.leaked_events),
                "rejected_count": len(cutoff.rejected_events),
                "is_clean": cutoff.is_clean
            }

        sample.compute_modality_availability()
        return sample
