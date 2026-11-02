from typing import Any, Optional
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VitalLabDynamicsAgent(BaseAgent):
    """
    Structured Clinical & Lab/Vital Agent (Time-series & Physiology Specialist).
    Analyzes hemodynamic vitals (HR, BP, SpO2, RR, Temp) and lab values (WBC, CRP, Troponin).
    Computes real-time clinical indices: Shock Index, qSOFA, and physiological instability score.
    """

    def __init__(self) -> None:
        self.confidence = 0.96

    @property
    def name(self) -> str:
        return "lab_vital_agent"

    def process(self, context: AgentContext) -> AgentContext:
        vitals = context.metadata.get("vitals") or {}
        labs = context.metadata.get("labs") or {}

        if not vitals and not labs:
            context.metadata["vitals_available"] = False
            return context

        context.metadata["vitals_available"] = True
        alerts = []

        # 1. Hemodynamics & Vitals
        hr = vitals.get("heart_rate")
        rr = vitals.get("respiratory_rate")
        sbp = vitals.get("systolic_bp")
        dbp = vitals.get("diastolic_bp")
        spo2 = vitals.get("spo2")
        temp = vitals.get("temperature_c")

        # Shock Index (SI = HR / SBP)
        shock_index = None
        if hr and sbp and sbp > 0:
            shock_index = round(hr / sbp, 2)
            if shock_index >= 0.9:
                alerts.append(f"Elevated Shock Index ({shock_index}): Potential circulatory shock")

        # qSOFA score (quick Sepsis-related Organ Failure Assessment)
        qsofa = 0
        if rr and rr >= 22:
            qsofa += 1
            alerts.append(f"Tachypnea (RR {rr}/min >= 22): Sepsis indicator (+1)")
        if sbp and sbp <= 100:
            qsofa += 1
            alerts.append(f"Hypotension (SBP {sbp} mmHg <= 100): Sepsis indicator (+1)")
        if vitals.get("altered_mental_state", False):
            qsofa += 1
            alerts.append("Altered mental status: Sepsis indicator (+1)")

        # Hypoxemia check
        hypoxemia_level = "Normal"
        if spo2:
            if spo2 < 90:
                hypoxemia_level = "Severe Hypoxemia"
                alerts.append(f"Severe Hypoxemia (SpO2 {spo2}% < 90%): Immediate O2 therapy required")
            elif spo2 < 94:
                hypoxemia_level = "Mild-Moderate Hypoxemia"
                alerts.append(f"Mild-to-moderate Hypoxemia (SpO2 {spo2}%)")

        # 2. Lab analysis (Inflammation & Cardiac)
        wbc = labs.get("wbc")
        crp = labs.get("crp")
        troponin = labs.get("troponin")

        if wbc and wbc > 12.0:
            alerts.append(f"Leukocytosis (WBC {wbc} k/uL): High likelihood of bacterial infection")
        if crp and crp > 10.0:
            alerts.append(f"Elevated CRP ({crp} mg/L): Significant systemic inflammation")
        if troponin and troponin > 0.04:
            alerts.append(f"Elevated Troponin ({troponin} ng/mL): Myocardial injury or strain")

        # Risk Classification
        if qsofa >= 2 or (spo2 and spo2 < 90) or (shock_index and shock_index >= 1.0):
            risk_tier = "Critical / High"
        elif qsofa == 1 or (spo2 and spo2 < 95) or alerts:
            risk_tier = "Moderate"
        else:
            risk_tier = "Low"

        vitals_summary = {
            "shock_index": shock_index,
            "qsofa_score": qsofa,
            "hypoxemia_level": hypoxemia_level,
            "active_alerts": alerts,
            "risk_tier": risk_tier,
        }

        context.metadata["vital_dynamics"] = vitals_summary
        context.metadata["physiological_risk"] = risk_tier

        logger.info(
            "VitalLabDynamicsAgent: Computed risk_tier=%s, qSOFA=%d, ShockIndex=%s, alerts=%d",
            risk_tier,
            qsofa,
            shock_index,
            len(alerts)
        )
        return context
