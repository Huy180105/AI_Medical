import re
from typing import Any, Optional
from datetime import datetime, timedelta
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TriageTimelineAgent(BaseAgent):
    """
    Triage & Clinical History Agent (ED & Clinical Notes Specialist).
    Extracts symptoms, chief complaints, and timeline of symptom onset
    (duration in hours, relative onset time) from clinical text and ED triage records.
    """

    def __init__(self, ner_predictor: Optional[Any] = None) -> None:
        self.ner_predictor = ner_predictor
        self.confidence = 0.95

    @property
    def name(self) -> str:
        return "triage_agent"

    def _extract_duration_hours(self, text: str) -> tuple[Optional[str], Optional[float]]:
        """
        Extracts duration phrases and converts to hours.
        Examples:
        - "3 ngày" / "3 days" -> ("3 ngày", 72.0)
        - "24 giờ" / "24h" / "24 hours" -> ("24 giờ", 24.0)
        - "2 tuần" / "2 weeks" -> ("2 tuần", 336.0)
        - "đột ngột" / "acute / sudden" -> ("đột ngột", 2.0)
        """
        lower = text.lower()
        
        # Patterns for day(s)
        day_match = re.search(r"(\d+)\s*(?:ngày|ngay|days?|d)", lower)
        if day_match:
            days = float(day_match.group(1))
            return (day_match.group(0), days * 24.0)

        # Patterns for hour(s)
        hour_match = re.search(r"(\d+)\s*(?:giờ|gio|tiếng|tieng|hours?|hrs?|h)", lower)
        if hour_match:
            hours = float(hour_match.group(1))
            return (hour_match.group(0), hours)

        # Patterns for week(s)
        week_match = re.search(r"(\d+)\s*(?:tuần|tuan|weeks?|w)", lower)
        if week_match:
            weeks = float(week_match.group(1))
            return (week_match.group(0), weeks * 168.0)

        # Acute / sudden
        if any(w in lower for w in ["đột ngột", "dot ngot", "cấp tính", "sudden", "acute"]):
            return ("đột ngột / cấp tính", 2.0)

        return (None, None)

    def process(self, context: AgentContext) -> AgentContext:
        text = context.text
        if not text:
            return context

        note_id = context.metadata.get("note_id", "ED_NOTE_001")
        extracted_symptoms = []

        # 1. Identify symptom entities
        symptom_entities = [e for e in context.entities if str(e.get("type", "")).upper() == "SYMPTOM"]

        # If NER hasn't run yet and predictor is available, extract symptoms directly
        if not symptom_entities and self.ner_predictor:
            try:
                all_entities = self.ner_predictor.predict(text)
                context.entities.extend(all_entities)
                symptom_entities = [e for e in all_entities if str(e.get("type", "")).upper() == "SYMPTOM"]
            except Exception as e:
                logger.warning("NER prediction in TriageTimelineAgent encountered: %s", e)

        # Fallback keyword detection for common chest symptoms if NER yielded none
        if not symptom_entities:
            common_symptoms = [
                ("khó thở", "Dyspnea"),
                ("ho", "Cough"),
                ("sốt", "Fever"),
                ("đau ngực", "Chest Pain"),
                ("thở dốc", "Shortness of breath"),
                ("khạc đờm", "Sputum production"),
            ]
            for kw, norm in common_symptoms:
                if kw in text.lower():
                    symptom_entities.append({"text": kw, "type": "SYMPTOM", "name": norm, "score": 0.85})

        # 2. Extract timeline and duration
        duration_text, duration_hours = self._extract_duration_hours(text)

        anchor_str = context.metadata.get("prediction_anchor_time")
        onset_timestamp_str = None
        if anchor_str and duration_hours is not None:
            try:
                anchor_dt = datetime.fromisoformat(anchor_str)
                onset_dt = anchor_dt - timedelta(hours=duration_hours)
                onset_timestamp_str = onset_dt.isoformat()
            except Exception:
                pass

        for sym in symptom_entities:
            sym_text = sym.get("text", "")
            rec = {
                "symptom_name": sym_text,
                "symptom_raw_text": sym_text,
                "symptom_onset_time": onset_timestamp_str,
                "symptom_duration_text": duration_text,
                "symptom_duration_hours": duration_hours,
                "onset_confidence": float(sym.get("score", 0.9)),
                "source_note_id": note_id,
            }
            extracted_symptoms.append(rec)

        context.metadata["structured_symptoms"] = extracted_symptoms
        context.metadata["symptom_count"] = len(extracted_symptoms)
        context.metadata["chief_duration_hours"] = duration_hours

        logger.info(
            "TriageTimelineAgent: Extracted %d structured symptoms (duration: %s, ~%.1f hours).",
            len(extracted_symptoms),
            duration_text or "N/A",
            duration_hours or 0.0
        )
        return context
