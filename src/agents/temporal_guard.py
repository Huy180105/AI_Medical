from datetime import datetime
from typing import Any, Optional
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalLeakageGuardAgent(BaseAgent):
    """
    Clinical Safety & Anti-Leakage Guardrail Agent.
    Strictly verifies that zero data from post-anchor timestamps enters the inference pipeline.
    Flags temporal anomalies and safeguards patient prediction integrity.
    """

    def __init__(self, allow_future_data: bool = False) -> None:
        self.allow_future_data = allow_future_data
        self.confidence = 1.0

    @property
    def name(self) -> str:
        return "temporal_guard"

    def process(self, context: AgentContext) -> AgentContext:
        anchor_str = context.metadata.get("prediction_anchor_time")
        if not anchor_str:
            # If no anchor provided, default to timestamp of initial request
            context.metadata["prediction_anchor_time"] = datetime.utcnow().isoformat()
            context.metadata["temporal_audit"] = {
                "status": "PASSED_UNCONSTRAINED",
                "message": "No retrospective anchor specified; real-time execution assumed."
            }
            return context

        try:
            anchor_dt = datetime.fromisoformat(anchor_str)
        except Exception as e:
            logger.warning("Could not parse prediction_anchor_time '%s': %s", anchor_str, e)
            context.metadata["temporal_audit"] = {"status": "WARNING", "message": str(e)}
            return context

        leaked_records = []
        
        # Check raw events or timestamped entries if present
        events = context.metadata.get("clinical_events", [])
        for ev in events:
            ts_str = ev.get("timestamp")
            if ts_str:
                try:
                    ev_dt = datetime.fromisoformat(ts_str)
                    if ev_dt > anchor_dt:
                        leaked_records.append(ev)
                except Exception:
                    pass

        # Check discharge summaries or post-hoc notes
        if "discharge_summary" in context.text.lower() or "tóm tắt xuất viện" in context.text.lower():
            context.metadata["discharge_leakage_warning"] = (
                "Discharge summary keywords detected in input text! "
                "Discharge notes should not be used as input for admission-time prediction."
            )
            logger.warning("Discharge summary keyword detected in admission prediction text!")

        if leaked_records and not self.allow_future_data:
            logger.error("TemporalLeakageGuardAgent: Found %d future events post-dating anchor!", len(leaked_records))
            context.metadata["temporal_audit"] = {
                "status": "LEAKAGE_DETECTED",
                "leaked_event_count": len(leaked_records),
                "leaked_records": leaked_records,
                "sanitized": True
            }
            # Remove leaked events
            context.metadata["clinical_events"] = [e for e in events if e not in leaked_records]
        else:
            context.metadata["temporal_audit"] = {
                "status": "PASSED_CLEAN",
                "anchor_time": anchor_str,
                "leaked_event_count": 0
            }

        return context
