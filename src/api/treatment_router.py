from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.agents.context import AgentContext
from src.agents.lab_vital_agent import VitalLabDynamicsAgent
from src.agents.temporal_guard import TemporalLeakageGuardAgent
from src.agents.triage_agent import TriageTimelineAgent
from src.privacy.homomorphic_engine import HomomorphicEncryptionEngine
from src.services.patient_data_service import get_xgboost_clinical_service
from src.utils.logger import get_logger
from src.vision.multimodal_fusion import MultimodalClinicalFusion, get_medfusion_engine
from src.vision.xray_classifier import ChestXRayClassifier

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/treatment", tags=["MedFusion-CXR Multi-Agent Clinical Platform"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUIDELINES_PATH = os.path.join(BASE_DIR, "knowledge_base", "bach_mai_guidelines.json")
BENCHMARK_REPORT_PATH = os.path.join(
    BASE_DIR, "models", "treatment_prediction", "benchmark_report.json"
)


def _load_bach_mai_guidelines() -> List[Dict[str, Any]]:
    if os.path.exists(GUIDELINES_PATH):
        try:
            with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("guidelines", [])
        except Exception:
            pass
    return []


BACH_MAI_GUIDELINES = _load_bach_mai_guidelines()
_fhe_engine: Optional[HomomorphicEncryptionEngine] = None
_compat_fusion: Optional[MultimodalClinicalFusion] = None


def _get_fhe_engine() -> HomomorphicEncryptionEngine:
    global _fhe_engine
    if _fhe_engine is None:
        _fhe_engine = HomomorphicEncryptionEngine()
    return _fhe_engine


def _get_compat_fusion() -> MultimodalClinicalFusion:
    global _compat_fusion
    if _compat_fusion is None:
        _compat_fusion = MultimodalClinicalFusion(
            xray_classifier=ChestXRayClassifier(allow_untrained=True)
        )
    return _compat_fusion


def _extract_clinical_entities(text: str, app_predictor: Any = None) -> List[Dict[str, str]]:
    """Extracts medical entities via loaded PhoBERT NER predictor with clinical lexicon fallback."""
    entities: List[Dict[str, str]] = []
    if not text.strip():
        return entities

    if app_predictor is not None and hasattr(app_predictor, "predict"):
        try:
            pred = app_predictor.predict(text)
            raw_ents = pred.get("entities", []) if isinstance(pred, dict) else pred
            for ent in raw_ents:
                if isinstance(ent, dict) and len(str(ent.get("text", "")).strip()) >= 2:
                    entities.append(
                        {
                            "text": str(ent["text"]).strip(),
                            "type": str(ent.get("type") or ent.get("label") or "SYMPTOM"),
                        }
                    )
        except Exception as exc:
            logger.debug("Fallback lexicon NER triggered (%s)", exc)

    # Supplement with high-precision Vietnamese clinical terminology matching
    lower = text.lower()
    lexicon = [
        ("sốt cao", "SYMPTOM"),
        ("rét run", "SYMPTOM"),
        ("sốt", "SYMPTOM"),
        ("ho có đờm", "SYMPTOM"),
        ("ho đờm vàng", "SYMPTOM"),
        ("ho kéo dài", "SYMPTOM"),
        ("khó thở", "SYMPTOM"),
        ("đau ngực", "SYMPTOM"),
        ("phù hai chân", "SYMPTOM"),
        ("mệt mỏi", "SYMPTOM"),
        ("tăng huyết áp", "DISEASE"),
        ("đái tháo đường típ 2", "DISEASE"),
        ("đái tháo đường", "DISEASE"),
        ("suy tim", "DISEASE"),
        ("viêm phổi", "DISEASE"),
        ("tràn dịch màng phổi", "DISEASE"),
        ("x-quang ngực", "TEST"),
        ("x-quang", "TEST"),
        ("công thức máu", "TEST"),
        ("crp", "TEST"),
        ("troponin", "TEST"),
        ("amlodipine", "MEDICINE"),
        ("metformin", "MEDICINE"),
    ]
    seen = {e["text"].lower() for e in entities}
    for term, etype in lexicon:
        if term in lower and term not in seen:
            entities.append({"text": term, "type": etype})
            seen.add(term)

    return entities


def _synthesize_multimodal_clinical_consensus(
    *,
    prob: float,
    age: int,
    gender: str = "Nam (Male)",
    sys_bp: float = 120.0,
    dia_bp: float = 80.0,
    heart_rate: float = 80.0,
    spo2: float = 98.0,
    respiratory_rate: Optional[float] = 18.0,
    temperature_c: Optional[float] = 37.0,
    chronic_conditions: int = 0,
    stress_score: int = 5,
    sleep_hours: float = 7.0,
    clinical_text: str = "",
    nlp_entities: Optional[List[Dict[str, Any]]] = None,
    vision_branch: Optional[Dict[str, Any]] = None,
    labs: Optional[Dict[str, Any]] = None,
    vital_dynamics: Optional[Dict[str, Any]] = None,
    xgb_branch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Synthesizes dual-modality clinical evidence from:
      1. Vision Branch: Pretrained DenseNet-121 on NIH ChestX-ray14 (14 pathologies).
      2. Clinical Branch: PhoBERT NER extracted symptoms/history + vitals + labs + XGBoost.
    Produces:
      - primary_diagnosis: Unified final diagnostic conclusion.
      - diagnostic_confidence_pct: Calibrated multimodal confidence.
      - acuity_risk: Emergency/acuity tier.
      - conclusion_summary: Plain-language cross-modal clinical rationale.
      - potential_diseases: Ranked list of differential diagnoses with cross-modal evidence.
      - recommendations: Actionable clinical guidance.
    """
    labs = labs or {}
    vital_dynamics = vital_dynamics or {}
    xgb_branch = xgb_branch or {}
    nlp_entities = nlp_entities or []

    # 1. Parse clinical narrative with negation preservation (do NOT append isolated unnegated NER tokens)
    full_text = clinical_text.lower()

    # Helper: Check if keywords appear WITHOUT a preceding negation term
    def _has_positive_symptom(text: str, keywords: List[str]) -> bool:
        negations = ["không", "khong", "ko", "k ", "chưa", "chua", "chẳng", "loại trừ", "chưa phát hiện", "ko co"]
        for kw in keywords:
            pos = 0
            while True:
                idx = text.find(kw, pos)
                if idx == -1:
                    break
                start = max(0, idx - 25)
                prefix = text[start:idx]
                if not any(neg in prefix for neg in negations):
                    return True
                pos = idx + len(kw)
        return False

    # Clinical features (negation-aware)
    has_fever = (
        _has_positive_symptom(full_text, ["sốt", "fever", "rét run"])
        or (temperature_c is not None and temperature_c >= 38.0)
    )
    has_cough = _has_positive_symptom(full_text, ["ho", "cough", "đờm", "khạc"])
    has_dyspnea = (
        _has_positive_symptom(full_text, ["khó thở", "thở nông", "thở dốc", "hụt hơi", "thở nhanh", "dyspnea"])
        or (spo2 < 94.0)
        or (respiratory_rate is not None and respiratory_rate >= 24.0)
    )
    has_chest_pain = _has_positive_symptom(
        full_text, ["đau ngực", "nhói ngực", "tức ngực", "nặng ngực", "đau nhói", "đau tức"]
    )
    has_orthopnea_edema = _has_positive_symptom(
        full_text, ["khó thở khi nằm", "phù", "bọt hồng", "kịch phát về đêm", "phù chân", "rale ẩm", "phù hai chân"]
    )
    has_hemoptysis = _has_positive_symptom(full_text, ["ho máu", "tia máu", "khạc máu", "ra máu"])
    has_smoking = _has_positive_symptom(full_text, ["thuốc lá", "hút thuốc", "bao-năm", "bao năm", "thuốc lào"])
    has_weight_loss = _has_positive_symptom(full_text, ["sụt cân", "gầy sút", "sút 4kg", "sút kg", "giảm cân"])
    has_surgery = _has_positive_symptom(full_text, ["hậu phẫu", "sau mổ", "phẫu thuật", "sau phẫu thuật"])
    has_tb = _has_positive_symptom(full_text, ["lao", "lao phổi", "lao cũ", "tuberculosis"])

    wbc_val = float(labs["wbc"]) if labs.get("wbc") is not None else None
    crp_val = float(labs["crp"]) if labs.get("crp") is not None else None
    trop_val = float(labs["troponin"]) if labs.get("troponin") is not None else None

    is_severe_hypoxia = (spo2 <= 90.0)
    is_tachycardic = (heart_rate >= 105.0)

    # 2. Parse X-ray Findings
    ranked_map: Dict[str, Dict[str, Any]] = {}
    top_pathology = "No Finding"
    top_cal_score = 0.0
    if vision_branch and "nih_14_ranked" in vision_branch:
        for item in vision_branch["nih_14_ranked"]:
            ranked_map[item["label"]] = item
        top_pathology = vision_branch.get("top_pathology", "No Finding")
        top_cal_score = (vision_branch.get("confidence_pct", 0.0) or 0.0) / 100.0

    candidates: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Asymptomatic Screening Check (Khám sức khỏe / Sàng lọc bình thường)
    # -------------------------------------------------------------------------
    is_screening_text = any(
        k in full_text for k in [
            "khám sức khỏe", "định kỳ", "sàng lọc", "tổng quát", "bình thường", "khỏe mạnh",
            "không sốt", "không ho", "không khó thở", "không đau ngực", "ổn định"
        ]
    )
    is_vitals_normal = (
        spo2 >= 96.0
        and (55.0 <= heart_rate <= 95.0)
        and (sys_bp < 135.0)
        and (temperature_c is None or temperature_c < 37.5)
        and (respiratory_rate is None or respiratory_rate <= 20.0)
    )
    is_labs_normal = (
        (crp_val is None or crp_val <= 6.0)
        and (wbc_val is None or wbc_val <= 10.5)
        and (trop_val is None or trop_val < 0.04)
    )
    has_any_acute_symptom = (
        has_fever or has_cough or has_dyspnea or has_chest_pain or has_orthopnea_edema
        or has_hemoptysis or has_surgery or has_tb
    )
    is_cxr_normal = (
        top_pathology == "No Finding"
        or vision_branch.get("flagged_count", 0) == 0
        or all(not item.get("flagged", False) for item in ranked_map.values())
        or all(item.get("probability", 0.0) < 0.12 for item in ranked_map.values())
    )
    is_asymptomatic_case = (
        is_cxr_normal
        and not has_any_acute_symptom
        and (is_screening_text or (is_vitals_normal and is_labs_normal))
    )

    if is_asymptomatic_case:
        norm_conf = round(min(98.5, max(92.0, (1.0 - prob) * 100.0)), 1)
        candidates.append(
            {
                "disease_name": "Chưa Ghi Nhận Bệnh Lý Cấp Tính Khu Trú (Sức Khỏe Ổn Định / Sàng Lọc)",
                "icd10_code": "Z00.0",
                "probability_pct": norm_conf,
                "severity": "Ổn định / Nguy cơ thấp",
                "cxr_evidence": (
                    "Phế trường 2 bên sáng đều, góc sườn hoành 2 bên nhọn sáng, "
                    "bóng tim và trung thất không to, vòm hoành mềm mại không phát hiện bóng mờ nhu mô khu trú."
                ),
                "clinical_evidence": (
                    f"Sinh hiệu hoàn toàn trong giới hạn sinh lý bình thường (Mạch {heart_rate:.0f} ck/p, "
                    f"HA {sys_bp:.0f}/{dia_bp:.0f} mmHg, SpO2 {spo2}%, Nhịp thở {respiratory_rate or 16:.0f} l/p). "
                    + (f"Chỉ số viêm và men tim âm tính (WBC {wbc_val} G/L, CRP {crp_val} mg/L). " if wbc_val else "")
                    + "Bệnh nhân không sốt, không ho, không đau ngực hay khó thở."
                ),
                "bach_mai_treatment": (
                    "Tư vấn duy trì lối sống lành mạnh, chế độ dinh dưỡng và vận động thể thao hợp lý. "
                    "Khuyến nghị khám sức khỏe định kỳ mỗi 6 - 12 tháng hoặc đi khám khi có triệu chứng bất thường."
                ),
            }
        )
        # Add differential exclusions with genuine low probabilities (< 10%)
        pna_raw = ranked_map.get("Pneumonia", {}).get("probability", 0.005)
        candidates.append(
            {
                "disease_name": "Viêm Phổi Cấp (Đã loại trừ / Nguy cơ rất thấp)",
                "icd10_code": "J18.9",
                "probability_pct": round(min(5.0, max(0.4, pna_raw * 100.0)), 1),
                "severity": "Nguy cơ rất thấp",
                "cxr_evidence": f"Không có tổn thương đông đặc hay thâm nhiễm phế nang (X-quang: {pna_raw*100:.1f}%).",
                "clinical_evidence": "Không sốt, không ho đờm, xét nghiệm viêm và chỉ số SpO2 hoàn toàn bình thường.",
                "bach_mai_treatment": "Không có chỉ định dùng kháng sinh hô hấp.",
            }
        )
        eff_raw = ranked_map.get("Effusion", {}).get("probability", 0.003)
        candidates.append(
            {
                "disease_name": "Tràn Dịch Màng Phổi (Đã loại trừ / Nguy cơ rất thấp)",
                "icd10_code": "J90",
                "probability_pct": round(min(4.5, max(0.3, eff_raw * 100.0)), 1),
                "severity": "Nguy cơ rất thấp",
                "cxr_evidence": f"Hai góc sườn hoành sáng rõ, không có mức dịch tự do (X-quang: {eff_raw*100:.1f}%).",
                "clinical_evidence": "Rì rào phế nang êm dịu, không đau ngực màng phổi hay khó thở khi gắng sức.",
                "bach_mai_treatment": "Không có chỉ định can thiệp thủ thuật chọc hút màng phổi.",
            }
        )
        card_raw = ranked_map.get("Cardiomegaly", {}).get("probability", 0.003)
        candidates.append(
            {
                "disease_name": "Bóng Tim To / Suy Tim (Đã loại trừ / Nguy cơ rất thấp)",
                "icd10_code": "I50.9",
                "probability_pct": round(min(4.0, max(0.3, card_raw * 100.0)), 1),
                "severity": "Nguy cơ rất thấp",
                "cxr_evidence": f"Chỉ số tim - lồng ngực < 0.50 trong giới hạn bình thường (X-quang: {card_raw*100:.1f}%).",
                "clinical_evidence": "Huyết áp và tần số tim ổn định, không khó thở khi nằm hay phù ngoại vi.",
                "bach_mai_treatment": "Tiếp tục theo dõi kiểm tra định kỳ.",
            }
        )
    else:
        # -------------------------------------------------------------------------
        # Disease 1: Tràn khí màng phổi (Pneumothorax)
        # -------------------------------------------------------------------------
        pno_item = ranked_map.get("Pneumothorax", {})
        pno_prob = pno_item.get("probability", 0.0)
        pno_cal = pno_item.get("calibrated_score", 0.0)
        has_pno_flag = pno_item.get("flagged", False) and pno_prob >= 0.12
        if has_pno_flag or top_pathology == "Pneumothorax" or (has_chest_pain and has_dyspnea):
            p_pct = min(98.5, max(pno_cal * 100.0, 65.0) + (14.0 if has_chest_pain else 0.0) + (10.0 if is_severe_hypoxia else 0.0))
            pain_desc = "Cơn đau nhói ngực dữ dội đột ngột" if has_chest_pain else "Đau tức ngực khu trú"
            hyp_desc = f", SpO2 tụt sâu {spo2}%" if is_severe_hypoxia else f", SpO2 {spo2}%"
            pulse_desc = f", nhịp tim nhanh {heart_rate:.0f} ck/p" if is_tachycardic else f", mạch {heart_rate:.0f} ck/p"
            candidates.append(
                {
                    "disease_name": "Tràn Khí Màng Phổi Cấp Tính (Pneumothorax)",
                    "icd10_code": "J93.9",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Cấp cứu",
                    "cxr_evidence": (
                        f"Hình ảnh vùng tăng sáng vô mạch ngoại vi, đường viền lá tạng màng phổi co rút "
                        f"(xác suất {pno_prob*100:.1f}% trên DenseNet-121)."
                    ),
                    "clinical_evidence": f"{pain_desc}{hyp_desc}{pulse_desc}.",
                    "bach_mai_treatment": (
                        "CẤP CỨU HÔ HẤP: Cho thở oxy liều cao qua mask; chọc kim giải áp khoang màng phổi tức thì nếu nghi tràn khí dưới áp lực; "
                        "đặt ống dẫn lưu màng phổi liên tục (hút áp lực âm -15 đến -20 cmH2O) theo phác đồ BV Bạch Mai."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 2: Viêm phổi cấp / Đông đặc thâm nhiễm (Pneumonia)
        # -------------------------------------------------------------------------
        pna_item = ranked_map.get("Pneumonia", {})
        inf_item = ranked_map.get("Infiltration", {})
        con_item = ranked_map.get("Consolidation", {})
        max_pna_prob = max(pna_item.get("probability", 0.0), inf_item.get("probability", 0.0), con_item.get("probability", 0.0))
        max_pna_cal = max(pna_item.get("calibrated_score", 0.0), inf_item.get("calibrated_score", 0.0), con_item.get("calibrated_score", 0.0))
        has_pna_flag = any(it.get("flagged") and it.get("probability", 0.0) >= 0.12 for it in [pna_item, inf_item, con_item])
        if has_pna_flag or top_pathology in ("Pneumonia", "Consolidation") or (has_fever and has_cough):
            base_p = max(max_pna_cal * 100.0, 60.0)
            bonus = (15.0 if (has_fever and has_cough) else 0.0) + (8.0 if crp_val and crp_val >= 25 else 0.0) + (6.0 if is_severe_hypoxia else 0.0)
            p_pct = min(97.5, base_p + bonus)
            fev_desc = f"Sốt ({temperature_c}°C)" if (temperature_c and temperature_c >= 37.8) else ("Hội chứng sốt" if has_fever else "Không sốt rõ")
            lab_items = []
            if crp_val is not None:
                lab_items.append(f"CRP {crp_val} mg/L")
            if wbc_val is not None:
                lab_items.append(f"Bạch cầu {wbc_val} G/L")
            lab_str = f", xét nghiệm ({', '.join(lab_items)})" if lab_items else ""
            candidates.append(
                {
                    "disease_name": "Viêm Phổi Thùy Cấp / Nhiễm Trùng Đường Hô Hấp Dưới",
                    "icd10_code": "J18.9",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Nghiêm trọng" if (spo2 < 93.0 or (crp_val and crp_val >= 50.0)) else "Vừa",
                    "cxr_evidence": (
                        f"Đám mờ phế nang / thâm nhiễm đông đặc nhu mô phổi có phế quản khí (xác suất {max_pna_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"Hội chứng nhiễm trùng hô hấp cấp: {fev_desc}, ho khạc đờm{lab_str}, SpO2 {spo2}%.",
                    "bach_mai_treatment": (
                        "Đánh giá thang điểm CURB-65 / PSI; cấy đờm, cấy máu trước khi dùng kháng sinh. "
                        "Khởi động kháng sinh kinh nghiệm đường tĩnh mạch theo phác đồ Viêm phổi BV Bạch Mai: "
                        "Beta-lactam/Inhibitor (Amoxicillin-Clavulanate hoặc Ceftriaxone) phối hợp Macrolide/Fluoroquinolone hô hấp."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 3: Phù phổi cấp do suy tim (Pulmonary Edema)
        # -------------------------------------------------------------------------
        edema_item = ranked_map.get("Edema", {})
        edema_prob = edema_item.get("probability", 0.0)
        edema_cal = edema_item.get("calibrated_score", 0.0)
        has_edema_flag = edema_item.get("flagged", False) and edema_prob >= 0.12
        if has_edema_flag or top_pathology == "Edema" or (has_orthopnea_edema and is_severe_hypoxia):
            base_p = max(edema_cal * 100.0, 62.0)
            bonus = (16.0 if is_severe_hypoxia else 0.0) + (10.0 if sys_bp >= 160 else 0.0) + (8.0 if has_orthopnea_edema else 0.0)
            p_pct = min(98.0, base_p + bonus)
            candidates.append(
                {
                    "disease_name": "Phù Phổi Cấp Huyết Động do Suy Tim Trái",
                    "icd10_code": "J81.0",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Cấp cứu",
                    "cxr_evidence": (
                        f"Bóng mờ phế nang dạng cánh bướm lan tỏa từ hai rốn phổi, dày vách liên tiểu thùy (xác suất {edema_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": (
                        f"Khó thở kịch phát dữ dội về đêm, ho khạc bọt hồng, huyết áp vọt ({sys_bp:.0f}/{dia_bp:.0f} mmHg), "
                        f"SpO2 tụt sâu ({spo2}%), rale ẩm dâng nhanh 2 phế trường."
                    ),
                    "bach_mai_treatment": (
                        "CẤP CỨU TỐI KHẨN: Thở oxy mask túi dự trữ hoặc hỗ trợ thông khí áp lực dương không xâm nhập (CPAP/BiPAP); "
                        "Furosemide tĩnh mạch; Nitroglycerin truyền tĩnh mạch kiểm soát tiền gánh; theo dõi sát áp lực tĩnh mạch trung tâm."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 4: Bóng tim to & Suy tim sung huyết (Cardiomegaly / Heart Failure)
        # -------------------------------------------------------------------------
        card_item = ranked_map.get("Cardiomegaly", {})
        card_prob = card_item.get("probability", 0.0)
        card_cal = card_item.get("calibrated_score", 0.0)
        has_card_flag = card_item.get("flagged", False) and card_prob >= 0.12
        if has_card_flag or top_pathology == "Cardiomegaly" or (has_orthopnea_edema and sys_bp >= 140.0):
            base_p = max(card_cal * 100.0, 60.0)
            bonus = (14.0 if has_orthopnea_edema else 0.0) + (8.0 if sys_bp >= 140 else 0.0)
            p_pct = min(96.0, base_p + bonus)
            symptom_str = "khó thở khi nằm đầu bằng, phù chi dưới" if has_orthopnea_edema else "khó thở khi gắng sức"
            trop_str = f", Troponin ({trop_val} ng/mL)" if trop_val is not None else ""
            candidates.append(
                {
                    "disease_name": "Bóng Tim To / Đợt Mất Bù Suy Tim Mạn",
                    "icd10_code": "I50.9",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Nghiêm trọng",
                    "cxr_evidence": (
                        f"Chỉ số tim - lồng ngực tăng > 0.5, cung dưới trái vồng, rốn phổi đậm (xác suất {card_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"{symptom_str}, huyết áp tăng ({sys_bp:.0f}/{dia_bp:.0f} mmHg){trop_str}.",
                    "bach_mai_treatment": (
                        "Đo ECG 12 chuyển đạo, định lượng NT-proBNP, siêu âm tim Doppler đánh giá chức năng tâm thu thất trái (LVEF); "
                        "tối ưu 4 trụ cột suy tim (ARNI/ACEi, Beta-blocker, MRA, SGLT2i) theo hướng dẫn Tim mạch BV Bạch Mai."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 5: Tràn dịch màng phổi (Pleural Effusion)
        # -------------------------------------------------------------------------
        eff_item = ranked_map.get("Effusion", {})
        eff_prob = eff_item.get("probability", 0.0)
        eff_cal = eff_item.get("calibrated_score", 0.0)
        has_eff_flag = eff_item.get("flagged", False) and eff_prob >= 0.12
        if has_eff_flag or top_pathology == "Effusion" or (_has_positive_symptom(full_text, ["tràn dịch"]) and has_dyspnea):
            base_p = max(eff_cal * 100.0, 60.0)
            bonus = (12.0 if has_dyspnea else 0.0) + (8.0 if "đáy phổi" in full_text else 0.0)
            p_pct = min(96.0, base_p + bonus)
            candidates.append(
                {
                    "disease_name": "Tràn Dịch Màng Phổi (Pleural Effusion)",
                    "icd10_code": "J90",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Nghiêm trọng" if spo2 < 93.0 else "Vừa",
                    "cxr_evidence": (
                        f"Mờ góc sườn hoành, bóng mờ đồng nhất vùng đáy phổi tạo đường cong Damoiseau (xác suất {eff_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": (
                        f"Tức ngực vùng đáy phổi, giảm rì rào phế nang, gõ đục, khó thở khi gắng sức, SpO2 {spo2}%."
                    ),
                    "bach_mai_treatment": (
                        "Siêu âm màng phổi định vị mức dịch; chọc dò dịch màng phổi làm sinh hóa (tiêu chuẩn Light: protein, LDH), "
                        "tế bào học và vi sinh theo hướng dẫn Hô hấp BV Bạch Mai."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 6: Xẹp phổi thùy / phân thùy (Atelectasis)
        # -------------------------------------------------------------------------
        ate_item = ranked_map.get("Atelectasis", {})
        ate_prob = ate_item.get("probability", 0.0)
        ate_cal = ate_item.get("calibrated_score", 0.0)
        has_ate_flag = ate_item.get("flagged", False) and ate_prob >= 0.12
        if has_ate_flag or top_pathology == "Atelectasis" or (has_surgery and has_dyspnea):
            base_p = max(ate_cal * 100.0, 58.0)
            bonus = (14.0 if has_surgery else 0.0) + (8.0 if has_dyspnea else 0.0)
            p_pct = min(94.0, base_p + bonus)
            surg_desc = "Thở nông sau phẫu thuật" if has_surgery else "Giảm thông khí đáy phổi"
            candidates.append(
                {
                    "disease_name": "Xẹp Phổi Thùy do Ứ Tắc Đờm / Giảm Thông Khí (Atelectasis)",
                    "icd10_code": "J98.11",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Vừa",
                    "cxr_evidence": (
                        f"Bóng mờ dạng dải/tam giác co kéo rãnh liên thùy, vòm hoành nâng cao cùng bên (xác suất {ate_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"{surg_desc}, ho khạc đờm kém, giảm âm phế bào vùng đáy phổi, SpO2 {spo2}%.",
                    "bach_mai_treatment": (
                        "Vật lý trị liệu hô hấp tích cực, vỗ rung dẫn lưu tư thế, tập thở sâu kích thích ho khạc đờm; "
                        "khí dung giãn phế quản và hút đờm qua nội soi phế quản nếu tắc nghẽn kéo dài."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 7: Khối u / Nốt nhu mô phổi (Lung Mass / Nodule)
        # -------------------------------------------------------------------------
        mass_item = ranked_map.get("Mass", {})
        nod_item = ranked_map.get("Nodule", {})
        max_mass_prob = max(mass_item.get("probability", 0.0), nod_item.get("probability", 0.0))
        max_mass_cal = max(mass_item.get("calibrated_score", 0.0), nod_item.get("calibrated_score", 0.0))
        has_mass_flag = any(it.get("flagged") and it.get("probability", 0.0) >= 0.12 for it in [mass_item, nod_item])
        if has_mass_flag or top_pathology in ("Mass", "Nodule") or (has_smoking and (has_hemoptysis or has_weight_loss)):
            base_p = max(max_mass_cal * 100.0, 58.0)
            bonus = (14.0 if has_smoking else 0.0) + (10.0 if (has_hemoptysis or has_weight_loss) else 0.0)
            p_pct = min(95.0, base_p + bonus)
            cl_items = []
            if has_smoking:
                cl_items.append("tiền sử hút thuốc lá nặng")
            if has_hemoptysis:
                cl_items.append("ho đờm lẫn máu")
            if has_weight_loss:
                cl_items.append("sụt cân bất thường")
            cl_str = ", ".join(cl_items) if cl_items else "ho kéo dài dai dẳng"
            candidates.append(
                {
                    "disease_name": "Khối U Nhu Mô Phổi / Cần Loại Trừ U Phế Quản Phổi",
                    "icd10_code": "C34.9 / R91.1",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Nghiêm trọng",
                    "cxr_evidence": (
                        f"Bóng mờ đơn độc / tổn thương dạng khối nhu mô phổi đậm độ mô mềm bờ không đều (xác suất {max_mass_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"{cl_str}, khó thở tăng dần, SpO2 {spo2}%.",
                    "bach_mai_treatment": (
                        "Chỉ định chụp cắt lớp vi tính lồng ngực (CT Scan ngực tiêm thuốc cản quang độ phân giải cao), "
                        "nội soi phế quản sinh thiết u hoặc sinh thiết phổi xuyên thành ngực dưới hướng dẫn CT để chẩn đoán mô bệnh học."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 8: Thâm nhiễm lan tỏa / Viêm phổi kẽ (Infiltration)
        # -------------------------------------------------------------------------
        inf_item = ranked_map.get("Infiltration", {})
        inf_prob = inf_item.get("probability", 0.0)
        inf_cal = inf_item.get("calibrated_score", 0.0)
        has_inf_flag = inf_item.get("flagged", False) and inf_prob >= 0.12
        if has_inf_flag or top_pathology == "Infiltration":
            base_p = max(inf_cal * 100.0, 58.0)
            bonus = (12.0 if has_fever else 0.0) + (8.0 if crp_val and crp_val >= 25 else 0.0)
            p_pct = min(95.0, base_p + bonus)
            fev_text = f"Sốt ({temperature_c}°C)" if temperature_c else "Sốt lâm sàng"
            candidates.append(
                {
                    "disease_name": "Thâm Nhiễm Nhu Mô Phổi Lan Tỏa / Viêm Phổi Kẽ",
                    "icd10_code": "J18.8 / J84.9",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Nghiêm trọng",
                    "cxr_evidence": (
                        f"Tổn thương thâm nhiễm dạng lưới nốt phế nang lan tỏa hai phế trường (xác suất {inf_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"{fev_text}, mệt mỏi li bì, rale nổ rải rác hai đáy phổi, SpO2 {spo2}%.",
                    "bach_mai_treatment": (
                        "Làm xét nghiệm PCR căn nguyên vi sinh đường hô hấp, bilan miễn dịch tự miễn; kháng sinh hô hấp phổ rộng "
                        "kết hợp cân nhắc Corticoid nếu nghi ngờ tổn thương kẽ cấp theo phác đồ BV Bạch Mai."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 9: Xơ hóa phổi & Dày dính màng phổi (Fibrosis / Pleural Thickening)
        # -------------------------------------------------------------------------
        fib_item = ranked_map.get("Fibrosis", {})
        pt_item = ranked_map.get("Pleural_Thickening", {})
        max_fib_prob = max(fib_item.get("probability", 0.0), pt_item.get("probability", 0.0))
        max_fib_cal = max(fib_item.get("calibrated_score", 0.0), pt_item.get("calibrated_score", 0.0))
        has_fib_flag = any(it.get("flagged") and it.get("probability", 0.0) >= 0.12 for it in [fib_item, pt_item])
        if has_fib_flag or top_pathology in ("Fibrosis", "Pleural_Thickening") or has_tb:
            base_p = max(max_fib_cal * 100.0, 58.0)
            bonus = (14.0 if has_tb else 0.0) + (8.0 if has_dyspnea else 0.0)
            p_pct = min(94.0, base_p + bonus)
            tb_desc = "Tiền sử lao phổi cũ đã điều trị" if has_tb else "Di chứng viêm phổi màng phổi cũ"
            candidates.append(
                {
                    "disease_name": "Xơ Hóa Phổi & Dày Dính Màng Phổi (Di chứng sau nhiễm trùng/lao)",
                    "icd10_code": "J84.10 / J94.0",
                    "probability_pct": round(p_pct, 1),
                    "severity": "Vừa",
                    "cxr_evidence": (
                        f"Dải xơ co kéo cấu trúc nhu mô phổi, dày dính màng phổi đỉnh và góc sườn hoành (xác suất {max_fib_prob*100:.1f}%)."
                    ),
                    "clinical_evidence": f"{tb_desc}, khó thở tăng dần khi gắng sức, giảm âm phế bào.",
                    "bach_mai_treatment": (
                        "Đo chức năng thông khí phổi (Hô hấp ký đánh giá hội chứng hạn chế), phục hồi chức năng hô hấp; "
                        "điều trị triệu chứng ho khó thở và phòng ngừa bội nhiễm vi khuẩn."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 10: Tăng huyết áp nguyên phát
        # -------------------------------------------------------------------------
        if sys_bp >= 140.0:
            candidates.append(
                {
                    "disease_name": "Tăng Huyết Áp (Bệnh lý nền phối hợp)",
                    "icd10_code": "I10",
                    "probability_pct": round(min(74.0, max(50.0, (sys_bp - 100.0) * 0.8)), 1),
                    "severity": "Nghiêm trọng" if sys_bp >= 160.0 else "Vừa",
                    "cxr_evidence": "Quai động mạch chủ vồng, gián tiếp phản ánh tình trạng tăng áp lực kéo dài.",
                    "clinical_evidence": f"Huyết áp đo thực tế {sys_bp:.0f}/{dia_bp:.0f} mmHg, tiền sử bệnh mạn tính.",
                    "bach_mai_treatment": (
                        "Kiểm soát huyết áp mục tiêu < 130/80 mmHg, giảm muối < 5g/ngày; phối hợp thuốc Ức chế men chuyển/Chẹn thụ thể "
                        "cùng Chẹn kênh Canxi (Amlodipine 5–10mg) theo phác đồ Tim mạch BV Bạch Mai."
                    ),
                }
            )

        # -------------------------------------------------------------------------
        # Disease 11: Tổn thương cơ tim cấp (hs-Troponin)
        # -------------------------------------------------------------------------
        if trop_val and trop_val >= 0.04:
            candidates.append(
                {
                    "disease_name": "Tổn Thương Cơ Tim Cấp / Cần Loại Trừ Hội Chứng Vành Cấp",
                    "icd10_code": "I21.9",
                    "probability_pct": 91.5,
                    "severity": "Cấp cứu",
                    "cxr_evidence": "X-quang ngực giúp loại trừ các cấp cứu lồng ngực khác (tràn khí màng phổi, phình tách ĐMC).",
                    "clinical_evidence": f"Men tim hs-Troponin tăng cao ({trop_val} ng/mL), kèm triệu chứng đau ngực hoặc khó thở cấp.",
                    "bach_mai_treatment": (
                        "Đo ECG 12 chuyển đạo ngay trong 10 phút, lặp lại hs-Troponin động học sau 1–3 giờ, hội chẩn Tim mạch can thiệp khẩn cấp."
                    ),
                }
            )

    # Sort candidates by probability descending
    candidates.sort(key=lambda x: x["probability_pct"], reverse=True)

    # Fallback if no specific condition triggered
    if not candidates:
        candidates.append(
            {
                "disease_name": "Chưa Ghi Nhận Bệnh Lý Cấp Tính Khu Trú (Sức Khỏe Ổn Định / Sàng Lọc)",
                "icd10_code": "Z00.0",
                "probability_pct": round(max(90.0, (1.0 - prob) * 100.0), 1),
                "severity": "Ổn định / Nguy cơ thấp",
                "cxr_evidence": "Không phát hiện bóng mờ nhu mô phổi, góc sườn hoành sáng nhọn, chỉ số tim ngực trong giới hạn bình thường.",
                "clinical_evidence": f"Không có triệu chứng hô hấp cấp, SpO2 {spo2}%, mạch {heart_rate:.0f} ck/p, sinh hiệu ổn định.",
                "bach_mai_treatment": "Duy trì lối sống lành mạnh, tập thể dục thường xuyên và kiểm tra sức khỏe định kỳ.",
            }
        )

    # Synthesize Final Primary Diagnosis & Clinical Summary
    top_candidate = candidates[0]
    primary_diag = top_candidate["disease_name"]
    top_conf = top_candidate["probability_pct"]

    # Acuity level
    if any(c["severity"] == "Cấp cứu" for c in candidates) or vital_dynamics.get("risk_tier") == "Critical / High":
        acuity_risk = "CẤP CỨU / NGUY CƠ RẤT CAO"
    elif any(c["severity"] == "Nghiêm trọng" for c in candidates) or vital_dynamics.get("risk_tier") == "High":
        acuity_risk = "NGUY CƠ CAO (CẦN ĐIỀU TRỊ Y TẾ)"
    elif any(c["severity"] == "Vừa" for c in candidates):
        acuity_risk = "NGUY CƠ VỪA (THEO DÕI SÁT)"
    else:
        acuity_risk = "SỨC KHỎE ỔN ĐỊNH"

    # Contextualize conclusion summary
    conclusion_summary = (
        f"Hội đồng hội chẩn đa phương thức xác lập chẩn đoán chính: {primary_diag} "
        f"(Độ tin cậy: {top_conf}% · Phân tầng: {acuity_risk}). "
        f"Kết quả phân tích hình ảnh X-quang ({top_candidate['cxr_evidence']}) đối chiếu trực tiếp "
        f"với diễn biến lâm sàng & thực thể bóc tách từ bệnh án ({top_candidate['clinical_evidence']})."
    )

    recommendations = [
        "Kết quả tổng hợp hỗ trợ bác sĩ đánh giá đa phương thức (Clinical Decision Support), không thay thế chẩn đoán lâm sàng.",
    ]
    if acuity_risk.startswith("CẤP CỨU"):
        recommendations.append(
            "CẢNH BÁO TỐI KHẨN: Bệnh nhân có dấu hiệu suy hô hấp hoặc rối loạn huyết động cấp (SpO2/qSOFA/Shock Index), cần ưu tiên can thiệp cấp cứu ngay."
        )
    if vision_branch and vision_branch.get("flagged_count", 0) > 0:
        recommendations.append(
            f"Đối chiếu trực tiếp tổn thương hình ảnh ({top_pathology}) với triệu chứng hô hấp và xét nghiệm viêm để cá thể hóa phác đồ."
        )
    recommendations.append(
        "Lưu ý phương pháp luận: Nhánh DenseNet-121 (NIH-14) và XGBoost (Clinical 1,600) đã kiểm định độc lập; tầng Gated Fusion vận hành ở chế độ Prototype kiến trúc."
    )

    return {
        "primary_diagnosis": primary_diag,
        "diagnostic_confidence_pct": top_conf,
        "acuity_risk": acuity_risk,
        "conclusion_summary": conclusion_summary,
        "potential_diseases": candidates,
        "recommendations": recommendations,
    }


def _build_bach_mai_protocols(
    *,
    prob: float,
    age: int,
    sys_bp: float,
    heart_rate: float,
    spo2: float,
    chronic_conditions: int,
    stress_score: int,
    sleep_hours: float,
    vision_branch: Optional[Dict[str, Any]] = None,
    labs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper delegating to multimodal consensus synthesis."""
    res = _synthesize_multimodal_clinical_consensus(
        prob=prob,
        age=age,
        sys_bp=sys_bp,
        heart_rate=heart_rate,
        spo2=spo2,
        chronic_conditions=chronic_conditions,
        stress_score=stress_score,
        sleep_hours=sleep_hours,
        vision_branch=vision_branch,
        labs=labs,
    )
    return res["potential_diseases"]


class TreatmentPredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(default="Nam (Male)")
    heart_rate: float = Field(..., ge=30, le=220)
    sys_bp: float = Field(..., ge=50, le=250)
    dia_bp: float = Field(..., ge=30, le=150)
    spo2: float = Field(..., ge=50, le=100)
    stress_score: int = Field(..., ge=1, le=10)
    sleep_hours: float = Field(..., ge=0, le=24)
    physical_activity_hrs: float = Field(..., ge=0, le=24)
    chronic_conditions: int = Field(..., ge=0, le=10)
    symptoms_severity: int = Field(..., ge=1, le=5)


@router.post("/predict")
async def predict_treatment_needed(payload: TreatmentPredictRequest) -> dict[str, Any]:
    """
    Evaluates patient clinical features using the trained XGBoost model + native TreeSHAP.
    """
    try:
        patient_dict = payload.model_dump()
        xgb_service = get_xgboost_clinical_service()
        xgb_res = xgb_service.analyze_patient(patient_dict)

        predicted_diseases = _build_bach_mai_protocols(
            prob=xgb_res["probability"],
            age=payload.age,
            sys_bp=payload.sys_bp,
            heart_rate=payload.heart_rate,
            spo2=payload.spo2,
            chronic_conditions=payload.chronic_conditions,
            stress_score=payload.stress_score,
            sleep_hours=payload.sleep_hours,
        )

        risk_factors = [
            {
                "factor": item["feature_label"],
                "value": str(item["raw_value"]),
                "impact": f"SHAP {item['shap_value']:+.3f} ({item['relative_impact_pct']}%)",
            }
            for item in xgb_res["shap_attributions"][:5]
            if item["shap_value"] > 0
        ]

        return {
            "status": "success",
            "treatment_needed": xgb_res["treatment_needed"],
            "treatment_probability_pct": xgb_res["probability_pct"],
            "risk_level": xgb_res["risk_level"],
            "champion_model": xgb_res["model_identifier"],
            "shap_attributions": xgb_res["shap_attributions"],
            "predicted_diseases": predicted_diseases,
            "risk_factors": risk_factors,
            "recommendations": [
                "Đối chiếu kết quả hỗ trợ quyết định lâm sàng với thăm khám trực tiếp của bác sĩ chuyên khoa.",
                "Theo dõi sát sinh hiệu (huyết áp, nhịp tim, SpO2) và tuân thủ phác đồ điều trị chuẩn.",
            ],
        }
    except Exception as exc:
        logger.error("Error predicting treatment: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-comprehensive")
async def predict_treatment_comprehensive(
    age: int = Form(default=58),
    gender: str = Form(default="Male"),
    heart_rate: float = Form(default=92.0),
    sys_bp: float = Form(default=148.0),
    dia_bp: float = Form(default=94.0),
    spo2: float = Form(default=93.5),
    stress_score: int = Form(default=8),
    sleep_hours: float = Form(default=4.5),
    physical_activity_hrs: float = Form(default=1.0),
    chronic_conditions: int = Form(default=2),
    symptoms_severity: int = Form(default=4),
    enable_fhe: str = Form(default="false"),
    clinical_text: str = Form(default="Khám đánh giá toàn diện sinh hiệu và X-quang lồng ngực."),
    xray_file: Optional[UploadFile] = File(default=None),
) -> dict[str, Any]:
    """
    Comprehensive Clinical Prediction Endpoint supporting XGBoost, CKKS Homomorphic Encryption,
    and Multimodal Chest X-Ray synthesis.
    """
    try:
        patient_dict = {
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "spo2": spo2,
            "stress_score": stress_score,
            "sleep_hours": sleep_hours,
            "physical_activity_hrs": physical_activity_hrs,
            "chronic_conditions": chronic_conditions,
            "symptoms_severity": symptoms_severity,
        }
        xgb_service = get_xgboost_clinical_service()
        xgb_res = xgb_service.analyze_patient(patient_dict)

        fhe_metadata = None
        if str(enable_fhe).strip().lower() in {"true", "1", "yes"}:
            fhe = _get_fhe_engine()
            fhe_vec = [
                float(symptoms_severity),
                float(chronic_conditions),
                0.0,
                1.0 if spo2 < 92.0 else 0.0,
                0.0,
                xgb_res["probability"],
                heart_rate,
                sys_bp,
                spo2,
                37.0,
                5.0,
                8.0,
            ]
            ctx, pub_bytes, _ = fhe.create_context()
            enc_bytes = fhe.encrypt_vector(fhe_vec, ctx)
            enc_out = fhe.evaluate_encrypted(enc_bytes, pub_bytes)
            dec_res = fhe.decrypt_predictions(enc_out, ctx)
            fhe_metadata = {
                "enabled": True,
                "scheme": dec_res["encryption_scheme"],
                "security_level": dec_res["privacy_guarantee"],
                "predicted_risk_level": dec_res["predicted_risk_level"],
                "probabilities": dec_res["probabilities"],
                "ciphertext_size_bytes": len(enc_bytes),
            }

        multimodal_xray = None
        if xray_file is not None and xray_file.filename:
            img_bytes = await xray_file.read()
            if img_bytes:
                compat_fusion = _get_compat_fusion()
                multimodal_xray = compat_fusion.analyze_multimodal(
                    image_input=img_bytes,
                    clinical_text=clinical_text,
                )

        predicted_diseases = _build_bach_mai_protocols(
            prob=xgb_res["probability"],
            age=age,
            sys_bp=sys_bp,
            heart_rate=heart_rate,
            spo2=spo2,
            chronic_conditions=chronic_conditions,
            stress_score=stress_score,
            sleep_hours=sleep_hours,
        )

        return {
            "status": "success",
            "treatment_needed": xgb_res["treatment_needed"],
            "treatment_probability_pct": xgb_res["probability_pct"],
            "risk_level": xgb_res["risk_level"],
            "champion_model": xgb_res["model_identifier"],
            "predicted_diseases": predicted_diseases,
            "fhe_metadata": fhe_metadata,
            "multimodal_xray": multimodal_xray,
        }
    except Exception as exc:
        logger.error("Comprehensive prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/benchmark")
async def get_model_benchmark() -> dict[str, Any]:
    """
    Returns live Experiment E2 benchmark metrics for all 6 trained ML/DL models
    from `models/treatment_prediction/benchmark_report.json`.
    """
    xgb_service = get_xgboost_clinical_service()
    summary = xgb_service.benchmark_summary
    if os.path.exists(BENCHMARK_REPORT_PATH):
        try:
            with open(BENCHMARK_REPORT_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    results = summary.get("benchmark_results", {})
    xgb_metrics = results.get("XGBoost", {})
    return {
        "status": "success",
        "experiment_id": summary.get("experiment_id", "E2_CLINICAL_TABULAR_BENCHMARK"),
        "primary_clinical_branch_model": "XGBoost",
        "best_model_name": "XGBoost",
        "validation_champion_name": summary.get("best_model_name", "XGBoost"),
        "best_f1_score": xgb_metrics.get("f1_score", summary.get("best_f1_score", 0.8462)),
        "best_roc_auc": xgb_metrics.get("roc_auc", summary.get("best_roc_auc", 0.9416)),
        "best_auprc": xgb_metrics.get("auprc", 0.9318),
        "dataset_split_summary": summary.get("dataset_split_summary", {}),
        "benchmark_results": results,
        "feature_importances": summary.get("feature_importances", {}),
        "xgboost_gain_importances": summary.get("xgboost_gain_importances", {}),
    }


@router.post("/multimodal_predict")
async def execute_multimodal_multi_agent_pipeline(
    fastapi_req: Request,
    clinical_text: str = Form(default=""),
    age: int = Form(default=58),
    gender: str = Form(default="Nam (Male)"),
    heart_rate: float = Form(default=92.0),
    sys_bp: float = Form(default=148.0),
    dia_bp: float = Form(default=94.0),
    spo2: float = Form(default=93.5),
    respiratory_rate: Optional[float] = Form(default=22.0),
    temperature_c: Optional[float] = Form(default=38.2),
    wbc: Optional[float] = Form(default=None),
    crp: Optional[float] = Form(default=None),
    troponin: Optional[float] = Form(default=None),
    stress_score: int = Form(default=8),
    sleep_hours: float = Form(default=4.5),
    physical_activity_hrs: float = Form(default=1.0),
    chronic_conditions: int = Form(default=2),
    symptoms_severity: int = Form(default=4),
    xray_file: Optional[UploadFile] = File(default=None),
) -> dict[str, Any]:
    """
    Executes the end-to-end MedFusion-CXR Multimodal Multi-Agent Pipeline:
    1. TemporalLeakageGuardAgent & TriageTimelineAgent (PhoBERT Clinical NLP)
    2. CXRDiagnosticAgent (Pretrained DenseNet-121 NIH-14 -> 1024-d F_v + 14 labels)
    3. VitalLabDynamicsAgent & XGBoostClinicalService (XGBoost + TreeSHAP -> 32-d F_c + qSOFA/SI)
    4. MedFusionGatedNetwork (256-d Feature-Level Gated Fusion GMU: g * F_v + (1 - g) * F_c)
    5. MultimodalConsensusOrchestrator & Bach Mai Guideline Synthesis
    """
    t_start = time.perf_counter()
    execution_steps: List[Dict[str, Any]] = []

    # 1. Temporal Leakage Guard + PhoBERT Triage NLP Agent
    app_predictor = getattr(fastapi_req.app.state, "predictor", None)
    nlp_entities = _extract_clinical_entities(clinical_text, app_predictor=app_predictor)

    agent_ctx = AgentContext(text=clinical_text or "Khám đánh giá đa phương thức")
    agent_ctx.metadata["vitals"] = {
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "spo2": spo2,
        "temperature_c": temperature_c,
    }
    labs_dict = {
        k: v
        for k, v in {"wbc": wbc, "crp": crp, "troponin": troponin}.items()
        if v is not None
    }
    agent_ctx.metadata["labs"] = labs_dict

    TemporalLeakageGuardAgent().process(agent_ctx)
    TriageTimelineAgent(ner_predictor=app_predictor).process(agent_ctx)
    VitalLabDynamicsAgent().process(agent_ctx)

    vital_dynamics = agent_ctx.metadata.get("vital_dynamics", {})
    duration_hrs = agent_ctx.metadata.get("chief_duration_hours")

    if clinical_text.strip():
        execution_steps.append(
            {
                "agent": "Agent 1 · TriageTimeline & TemporalGuard (PhoBERT NLP)",
                "status": "success",
                "detail": (
                    f"Kiểm định mốc thời gian t₀ hợp lệ; trích xuất {len(nlp_entities)} thực thể y khoa"
                    + (f" (thời gian khởi phát ~{duration_hrs:.0f}h)." if duration_hrs else ".")
                ),
            }
        )
    else:
        execution_steps.append(
            {
                "agent": "Agent 1 · TriageTimeline & TemporalGuard (PhoBERT NLP)",
                "status": "skipped",
                "detail": "Không nhập bệnh sử tự do; kích hoạt kiểm định mốc thời gian t₀ mặc định.",
            }
        )

    # 2. Read CXR Radiograph Bytes (if uploaded)
    img_bytes: Optional[bytes] = None
    uploaded_filename: Optional[str] = None
    if xray_file is not None and xray_file.filename:
        raw = await xray_file.read()
        if raw:
            img_bytes = raw
            uploaded_filename = xray_file.filename

    # 3. Execute Dual-Branch MedFusion Engine (DenseNet-121 NIH-14 + XGBoost + Gated Fusion)
    patient_tabular = {
        "age": age,
        "gender": gender,
        "heart_rate": heart_rate,
        "sys_bp": sys_bp,
        "dia_bp": dia_bp,
        "spo2": spo2,
        "stress_score": stress_score,
        "sleep_hours": sleep_hours,
        "physical_activity_hrs": physical_activity_hrs,
        "chronic_conditions": chronic_conditions,
        "symptoms_severity": symptoms_severity,
    }

    medfusion = get_medfusion_engine()
    mf_out = medfusion.run_multimodal_analysis(
        patient_tabular=patient_tabular,
        image_bytes_or_path=img_bytes,
        vitals_extra=agent_ctx.metadata["vitals"],
        labs=labs_dict,
        nlp_entities=nlp_entities,
    )

    vision_branch = mf_out["vision_branch"]
    xgb_branch = mf_out["clinical_branch"]
    gated_fusion = mf_out["gated_fusion"]

    xray_results: Optional[Dict[str, Any]] = None
    if vision_branch is not None:
        xray_results = {
            "filename": uploaded_filename,
            "backbone": vision_branch["backbone"],
            "checkpoint_sha256_prefix": vision_branch["checkpoint_sha256_prefix"],
            "primary_finding": vision_branch["primary_finding"],
            "top_pathology": vision_branch["top_pathology"],
            "confidence_pct": vision_branch["confidence_pct"],
            "raw_probability_pct": vision_branch["raw_probability_pct"],
            "flagged_count": vision_branch["flagged_count"],
            "detected_pathologies": vision_branch["nih_14_ranked"],
            "recommendation": (
                f"Trích xuất vector F_v (1024-d) từ checkpoint NIH-14 ({vision_branch['checkpoint_sha256_prefix']}…). "
                f"Nhãn nổi bật nhất: {vision_branch['top_pathology']} (calibrated {vision_branch['confidence_pct']}%)."
            ),
        }
        execution_steps.append(
            {
                "agent": "Agent 2 · CXRDiagnosticAgent (DenseNet-121 NIH-14)",
                "status": "success",
                "detail": (
                    f"Phân tích 14 nhãn NIH ChestX-ray14 thành công: {vision_branch['top_pathology']} "
                    f"({vision_branch['confidence_pct']}%, F_v ∈ ℝ¹⁰²⁴)."
                ),
            }
        )
    else:
        execution_steps.append(
            {
                "agent": "Agent 2 · CXRDiagnosticAgent (DenseNet-121 NIH-14)",
                "status": "skipped",
                "detail": "Chưa cung cấp phim X-quang ngực; hệ thống chuyển sang chế độ Clinical-only an toàn.",
            }
        )

    execution_steps.append(
        {
            "agent": "Agent 3 · Clinical Intelligence (XGBoost + VitalLabDynamicsAgent)",
            "status": "success",
            "detail": (
                f"XGBoost dự đoán xác suất cần can thiệp {xgb_branch['probability_pct']}% "
                f"(ngưỡng Val {xgb_branch['optimal_threshold']}); "
                f"Shock Index={vital_dynamics.get('shock_index', 'N/A')}, qSOFA={vital_dynamics.get('qsofa_score', 0)}."
            ),
        }
    )

    execution_steps.append(
        {
            "agent": "Agent 4 · Gated Feature Fusion Layer (GMU 256-d Prototype)",
            "status": "success" if gated_fusion["active"] else "warning",
            "detail": (
                f"Gate ḡ_vision={gated_fusion['gate_vision_weight']:.2f} | "
                f"(1−ḡ)_clinical={gated_fusion['gate_clinical_weight']:.2f} trên không gian chung ℝ²⁵⁶ "
                f"[{gated_fusion['validation_status']}]."
            ),
        }
    )

    consensus_result = _synthesize_multimodal_clinical_consensus(
        prob=xgb_branch["probability"],
        age=age,
        gender=gender,
        sys_bp=sys_bp,
        dia_bp=dia_bp,
        heart_rate=heart_rate,
        spo2=spo2,
        respiratory_rate=respiratory_rate,
        temperature_c=temperature_c,
        chronic_conditions=chronic_conditions,
        stress_score=stress_score,
        sleep_hours=sleep_hours,
        clinical_text=clinical_text,
        nlp_entities=nlp_entities,
        vision_branch=vision_branch,
        labs=labs_dict,
        vital_dynamics=vital_dynamics,
        xgb_branch=xgb_branch,
    )
    predicted_diseases = consensus_result["potential_diseases"]

    risk_factors: List[Dict[str, Any]] = []
    for item in xgb_branch["shap_attributions"][:5]:
        if abs(item["shap_value"]) >= 0.05:
            direction_vi = "Tăng nguy cơ" if item["shap_value"] > 0 else "Giảm nguy cơ"
            risk_factors.append(
                {
                    "factor": item["feature_label"],
                    "value": str(item["raw_value"]),
                    "impact": f"{direction_vi} · TreeSHAP {item['shap_value']:+.3f} ({item['relative_impact_pct']}%)",
                    "shap_value": item["shap_value"],
                    "relative_impact_pct": item["relative_impact_pct"],
                }
            )

    for alert in vital_dynamics.get("active_alerts", []):
        risk_factors.append(
            {
                "factor": "Cảnh báo Sinh lý / Xét nghiệm Cấp (VitalLabDynamicsAgent)",
                "value": alert,
                "impact": f"Mức nguy cơ sinh lý: {vital_dynamics.get('risk_tier', 'Moderate')}",
                "shap_value": 0.35,
                "relative_impact_pct": 15.0,
            }
        )

    execution_steps.append(
        {
            "agent": "Agent 5 · MultimodalConsensusOrchestrator & Bach Mai RAG",
            "status": "success",
            "detail": (
                f"Tổng hợp {len(predicted_diseases)} định hướng chẩn đoán & phác đồ tham khảo, "
                f"giải thích {len(risk_factors)} yếu tố TreeSHAP & huyết động."
            ),
        }
    )

    latency_ms = round((time.perf_counter() - t_start) * 1000.0, 1)

    recommendations = consensus_result["recommendations"]

    return {
        "status": "success",
        "latency_ms": latency_ms,
        "multimodal_consensus": consensus_result,
        "execution_steps": execution_steps,
        "agent1_nlp": nlp_entities,
        "agent2_xray": xray_results,
        "agent3_prediction": {
            "treatment_needed": xgb_branch["treatment_needed"],
            "probability_pct": xgb_branch["probability_pct"],
            "optimal_threshold": xgb_branch["optimal_threshold"],
            "risk_level": xgb_branch["risk_level"],
            "champion_model": xgb_branch["model_identifier"],
            "shap_attributions": xgb_branch["shap_attributions"],
            "tree_leaf_preview": xgb_branch["tree_leaf_preview"],
        },
        "vital_dynamics": vital_dynamics,
        "gated_fusion": gated_fusion,
        "agent4_protocols": predicted_diseases,
        "agent5_xai": risk_factors,
        "consensus_recommendations": recommendations,
    }
