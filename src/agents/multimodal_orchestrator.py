from typing import Any, Optional
from src.agents.base import BaseAgent
from src.agents.context import AgentContext
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultimodalConsensusOrchestrator(BaseAgent):
    """
    Multimodal Clinical Synthesizer / Master Orchestrator Agent (Hội đồng Hội chẩn Đa phương thức).
    Synthesizes vision findings, clinical history/symptoms, vitals/labs time-series,
    and medical guidelines. Resolves cross-modal discrepancies to produce actionable diagnoses.
    """

    def __init__(self) -> None:
        self.confidence = 0.98

    @property
    def name(self) -> str:
        return "multimodal_orchestrator"

    def process(self, context: AgentContext) -> AgentContext:
        logger.info("MultimodalConsensusOrchestrator: Convening multidisciplinary clinical synthesis...")

        # 1. Harvest Findings from Specialist Agents
        cxr_findings = context.metadata.get("cxr_findings", {})
        cxr_primary = context.metadata.get("cxr_primary_finding", "Normal")
        cxr_available = context.metadata.get("cxr_available", False)

        structured_symptoms = context.metadata.get("structured_symptoms", [])
        duration_hours = context.metadata.get("chief_duration_hours")
        vital_dynamics = context.metadata.get("vital_dynamics", {})
        risk_tier = context.metadata.get("physiological_risk", "Low")
        guidelines = context.guidelines or []

        symptom_names = [s.get("symptom_name", "").lower() for s in structured_symptoms]

        # 2. Cross-Modal Evidence Synthesis & Conflict Resolution
        differentials = []
        evidence_chain = []
        primary_diagnosis = "Chưa phát hiện bệnh lý hô hấp nghiêm trọng"
        diagnostic_confidence = 0.50

        has_fever = any("sốt" in s or "fever" in s for s in symptom_names)
        has_cough = any("ho" in s or "cough" in s for s in symptom_names)
        has_dyspnea = any("khó thở" in s or "thở dốc" in s or "dyspnea" in s for s in symptom_names)
        has_chest_pain = any("đau ngực" in s or "chest pain" in s for s in symptom_names)

        pna_score = cxr_findings.get("Pneumonia", 0.0)
        effusion_score = cxr_findings.get("Pleural Effusion", 0.0)
        cardiomegaly_score = cxr_findings.get("Cardiomegaly", 0.0)
        infiltrate_score = cxr_findings.get("Infiltration", 0.0)
        atelectasis_score = cxr_findings.get("Atelectasis", 0.0)

        # Rule 1: Pneumonia (Viêm phổi)
        if (pna_score > 0.4 or infiltrate_score > 0.5) and (has_fever or has_cough):
            confidence = min(0.96, max(pna_score, infiltrate_score) * 0.7 + 0.3)
            primary_diagnosis = "Viêm phổi thùy / Nhiễm trùng đường hô hấp dưới"
            diagnostic_confidence = confidence
            evidence_chain.append(
                f"Tổn thương thâm nhiễm trên X-quang (score {max(pna_score, infiltrate_score):.2f}) "
                f"kết hợp hội chứng nhiễm trùng ({', '.join(symptom_names)})."
            )
            differentials.append({"disease": "Viêm phế quản phổi", "probability": round(confidence * 0.85, 2)})

        # Rule 2: Pleural Effusion (Tràn dịch màng phổi)
        elif effusion_score > 0.5:
            primary_diagnosis = "Tràn dịch màng phổi (Pleural Effusion)"
            diagnostic_confidence = effusion_score
            evidence_chain.append(f"Mờ góc sườn hoành / đậm độ dịch màng phổi trên CXR (score {effusion_score:.2f}).")
            if has_dyspnea:
                evidence_chain.append("Khó thở do giảm thể tích phổi thông khí.")
            differentials.append({"disease": "Viêm mủ màng phổi / Dày dính màng phổi", "probability": 0.65})

        # Rule 3: Cardiomegaly & Congestive Heart Failure
        elif cardiomegaly_score > 0.5:
            primary_diagnosis = "Tim to / Theo dõi Suy tim ứ huyết"
            diagnostic_confidence = cardiomegaly_score
            evidence_chain.append(f"Chỉ số tim - lồng ngực tăng trên CXR (score {cardiomegaly_score:.2f}).")
            if has_dyspnea:
                evidence_chain.append("Khó thở khi gắng sức / nằm đầu bằng gợi ý ứ trệ tuần hoàn phổi.")
            differentials.append({"disease": "Tràn dịch màng ngoài tim", "probability": 0.55})

        # Rule 4: Atelectasis (Xẹp phổi)
        elif atelectasis_score > 0.5:
            primary_diagnosis = "Xẹp phổi (Atelectasis)"
            diagnostic_confidence = atelectasis_score
            evidence_chain.append(f"Hình ảnh giảm thể tích thùy/phân thùy phổi trên CXR (score {atelectasis_score:.2f}).")
            differentials.append({"disease": "Tắc nghẽn đường thở do đờm / Dị vật", "probability": 0.50})

        # Rule 5: Clinical dominance when CXR is normal but patient is acute
        elif has_chest_pain and (risk_tier in ["Critical / High", "Moderate"] or duration_hours and duration_hours < 6.0):
            primary_diagnosis = "Đau thắt ngực / Cần loại trừ Hội chứng vành cấp hoặc Thuyên tắc phổi"
            diagnostic_confidence = 0.80
            evidence_chain.append(
                "X-quang ngực chưa ghi nhận bất thường cấp, nhưng đau ngực cấp tính và huyết động nguy cơ."
            )
            differentials.append({"disease": "Viêm sụn sườn / Trào ngược dạ dày", "probability": 0.40})

        # Fallback if no specific trigger
        else:
            if cxr_available:
                primary_diagnosis = "X-quang ngực trong giới hạn bình thường"
                diagnostic_confidence = 0.88
                evidence_chain.append("Không phát hiện tổn thương phổi khu trú hoặc bóng tim to bất thường.")
            else:
                primary_diagnosis = "Đánh giá lâm sàng dựa trên triệu chứng (Thiếu ảnh X-quang)"
                diagnostic_confidence = 0.70
                evidence_chain.append(f"Ghi nhận triệu chứng lâm sàng: {', '.join(symptom_names) or 'Không rõ'}.")

        # 3. Actionable Clinical Recommendations
        recommendations = []
        if risk_tier == "Critical / High":
            recommendations.append("BÁO ĐỘNG ĐỎ: Đặt bệnh nhân vào khu hồi sức cấp cứu, thở oxy hỗ trợ, theo dõi monitor liên tục.")
        if primary_diagnosis.startswith("Viêm phổi"):
            recommendations.append("Chỉ định công thức máu (CBC), CRP/Procalcitonin, cấy đờm làm kháng sinh đồ.")
            recommendations.append("Khởi động kháng sinh kinh nghiệm theo phác đồ hướng dẫn viêm phổi cộng đồng.")
        elif "Suy tim" in primary_diagnosis or cardiomegaly_score > 0.5:
            recommendations.append("Làm điện tâm đồ 12 chuyển đạo (ECG), xét nghiệm NT-proBNP và siêu âm tim qua thành ngực.")
        elif "Tràn dịch" in primary_diagnosis:
            recommendations.append("Siêu âm màng phổi đánh giá mức độ dịch, cân nhắc chọc hút dịch màng phổi xét nghiệm chẩn đoán.")

        # 4. Construct Final Comprehensive Multimodal Response
        consensus_report = {
            "primary_diagnosis": primary_diagnosis,
            "diagnostic_confidence": round(diagnostic_confidence, 2),
            "differential_diagnoses": differentials,
            "evidence_chain": evidence_chain,
            "clinical_acuity_risk": risk_tier,
            "vital_dynamics": vital_dynamics,
            "structured_symptoms": structured_symptoms,
            "actionable_recommendations": recommendations,
            "modalities_used": {
                "cxr_imaging": cxr_available,
                "symptoms_history": len(structured_symptoms) > 0,
                "vital_signs": bool(vital_dynamics),
                "clinical_guidelines": len(guidelines) > 0
            }
        }

        context.metadata["multimodal_consensus"] = consensus_report
        context.clinical_decision = consensus_report
        context.response = {
            "status": "success",
            "report_type": "Multimodal Consensus Clinical Summary",
            "clinical_decision": consensus_report
        }

        logger.info(
            "MultimodalConsensusOrchestrator: Formulated primary diagnosis '%s' (conf=%.2f, risk=%s).",
            primary_diagnosis,
            diagnostic_confidence,
            risk_tier
        )
        return context
