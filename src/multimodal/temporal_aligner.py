from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClinicalEvent:
    event_id: str
    event_type: str  # 'vital', 'lab', 'note', 'symptom', 'image'
    event_time: Optional[datetime]  # Thời điểm đo đạc lâm sàng (charttime)
    data: dict[str, Any]
    source_table: str
    available_time: Optional[datetime] = None  # Thời điểm kết quả sẵn sàng (storetime/result_time)
    subject_id: Optional[str] = None
    hadm_id: Optional[str] = None
    stay_id: Optional[str] = None
    time_source: str = "charttime"  # 'charttime', 'storetime', 'intime', etc.
    timestamp_quality: str = "VERIFIED"  # 'VERIFIED', 'ESTIMATED', 'MISSING', 'CORRUPT'

    @property
    def timestamp(self) -> Optional[datetime]:
        """Backward compatibility helper property."""
        return self.event_time


@dataclass
class TemporalCutoffResult:
    anchor_time: datetime
    expected_subject_id: Optional[str]
    valid_events: list[ClinicalEvent]
    leaked_events: list[ClinicalEvent]
    rejected_events: list[dict[str, Any]]  # Rejected due to missing time, mismatch subject, timezone
    is_clean: bool


class TemporalAligner:
    """
    Guarantees strict temporal integrity and auditability across multimodal clinical data.
    Enforces that only verified events occurring and available at or before anchor time t_0
    are admitted into predictive reasoning.
    """

    def __init__(
        self,
        buffer_minutes: float = 0.0,
        require_available_time: bool = True,
        strict_subject_check: bool = True
    ) -> None:
        """
        :param buffer_minutes: Allowed window in minutes (must be 0.0 for strict point-in-time prediction).
        :param require_available_time: When True, requires available_time <= anchor_time
                                       (preventing using lab results taken before t_0 whose results returned after t_0).
        :param strict_subject_check: When True, rejects any event where subject_id does not match expected_subject_id.
        """
        self.buffer_minutes = buffer_minutes
        self.require_available_time = require_available_time
        self.strict_subject_check = strict_subject_check

    def _normalize_timezone(self, dt: datetime, reference_dt: datetime) -> Optional[datetime]:
        """
        Safely reconciles timezone-aware and timezone-naive datetimes.
        If reference_dt is naive, strips tzinfo from dt.
        If reference_dt is aware, converts naive dt to UTC.
        """
        if dt is None:
            return None
        if reference_dt.tzinfo is None and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        if reference_dt.tzinfo is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def is_usable_at_anchor(
        self,
        event: ClinicalEvent,
        anchor_time: datetime,
        expected_subject_id: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Evaluates admissibility of a single clinical event at anchor time t_0.
        Returns (is_usable: bool, rejection_reason: str).
        """
        # 1. Subject identity matching
        if self.strict_subject_check and expected_subject_id:
            if not event.subject_id:
                return False, f"REJECTED_MISSING_SUBJECT_ID: Event {event.event_id} has no subject_id"
            if str(event.subject_id).strip() != str(expected_subject_id).strip():
                return False, (
                    f"REJECTED_SUBJECT_MISMATCH: Event subject '{event.subject_id}' "
                    f"!= expected '{expected_subject_id}'"
                )

        # 2. Event time validity
        if event.event_time is None:
            return False, f"REJECTED_MISSING_EVENT_TIME: Event {event.event_id} has null event_time"

        norm_event_time = self._normalize_timezone(event.event_time, anchor_time)
        norm_anchor_time = anchor_time

        # 3. Temporal leakage check (event occurred after anchor + buffer)
        diff_event_seconds = (norm_event_time - norm_anchor_time).total_seconds()
        if diff_event_seconds > (self.buffer_minutes * 60.0):
            return False, (
                f"LEAKED_FUTURE_EVENT: Event time {norm_event_time.isoformat()} occurs after "
                f"anchor {norm_anchor_time.isoformat()} (delta: +{diff_event_seconds:.1f}s)"
            )

        # 4. Result availability check (charttime vs storetime)
        # Even if blood was drawn before t_0, if the analyzer posted results after t_0, it was not known at t_0.
        if self.require_available_time:
            avail_time = event.available_time or event.event_time
            norm_avail_time = self._normalize_timezone(avail_time, anchor_time)
            diff_avail_seconds = (norm_avail_time - norm_anchor_time).total_seconds()
            if diff_avail_seconds > (self.buffer_minutes * 60.0):
                return False, (
                    f"LEAKED_FUTURE_RESULT_AVAILABILITY: Result became available at "
                    f"{norm_avail_time.isoformat()} after anchor {norm_anchor_time.isoformat()}"
                )

        # 5. Quality status check
        if event.timestamp_quality == "CORRUPT":
            return False, f"REJECTED_CORRUPT_TIMESTAMP: Event {event.event_id} timestamp marked CORRUPT"

        return True, "USABLE"

    def filter_events_by_anchor(
        self,
        anchor_time: datetime,
        events: list[ClinicalEvent],
        subject_id: Optional[str] = None
    ) -> TemporalCutoffResult:
        """
        Partitions clinical events into usable historical records, leaked records, and rejected records.
        """
        valid_events = []
        leaked_events = []
        rejected_events = []

        for ev in events:
            usable, reason = self.is_usable_at_anchor(ev, anchor_time, expected_subject_id=subject_id)
            if usable:
                valid_events.append(ev)
            elif reason.startswith("LEAKED_"):
                leaked_events.append(ev)
                rejected_events.append({"event_id": ev.event_id, "reason": reason})
            else:
                rejected_events.append({"event_id": ev.event_id, "reason": reason})

        if leaked_events:
            logger.warning(
                "Temporal leakage audited for subject %s: %d events post-dated anchor %s",
                subject_id or "UNKNOWN",
                len(leaked_events),
                anchor_time.isoformat()
            )

        if rejected_events:
            logger.info(
                "Temporal aligner filtered %d non-conforming events for subject %s",
                len(rejected_events),
                subject_id or "UNKNOWN"
            )

        return TemporalCutoffResult(
            anchor_time=anchor_time,
            expected_subject_id=subject_id,
            valid_events=valid_events,
            leaked_events=leaked_events,
            rejected_events=rejected_events,
            is_clean=(len(leaked_events) == 0 and len(rejected_events) == 0)
        )

    def calculate_relative_hours(self, event_time: datetime, anchor_time: datetime) -> float:
        norm_event_time = self._normalize_timezone(event_time, anchor_time)
        delta = norm_event_time - anchor_time
        return round(delta.total_seconds() / 3600.0, 2)
