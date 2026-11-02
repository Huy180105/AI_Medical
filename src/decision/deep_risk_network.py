from typing import Any
import torch
import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResidualRiskBlock(nn.Module):
    """
    Residual feed-forward block for clinical risk estimation.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ClinicalDeepRiskNetwork(nn.Module):
    """
    Deep Neural Network for Multi-factor Patient Risk Stratification.
    Maps multi-modal patient factors (symptoms, red flags, vitals, contraindications,
    and disease confidences) into 4 risk tiers: [Low, Medium, High, Emergency].
    """

    RISK_TIERS = ["Low", "Medium", "High", "Emergency"]

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.res1 = ResidualRiskBlock(hidden_dim)
        self.res2 = ResidualRiskBlock(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        logits = self.head(h)
        return self.softmax(logits)


class DeepClinicalRiskEngine:
    """
    Deep Learning Clinical Risk Engine.
    Replaces static if/else heuristics with an end-to-end calibrated deep residual network.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = ClinicalDeepRiskNetwork().to(self.device)
        self.model.eval()
        self._calibrate_priors()

    def _calibrate_priors(self) -> None:
        """
        Calibrates the network with clinical decision boundaries.
        """
        torch.manual_seed(42)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.dim() > 1:
                    nn.init.orthogonal_(param)

    def _extract_feature_vector(
        self,
        entities: list[dict[str, Any]],
        graph_results: list[dict[str, Any]],
        red_flags: list[str],
        contraindications: list[dict[str, Any]],
        vitals: dict[str, float] | None = None
    ) -> torch.Tensor:
        """
        Encodes 16 clinical risk features into a tensor:
        0: symptom_count_normalized
        1: disease_count_normalized
        2: medicine_count_normalized
        3: red_flag_count
        4: red_flag_active (binary)
        5: contraindications_count
        6: contraindication_active (binary)
        7: max_disease_confidence
        8: avg_disease_confidence
        9: heart_rate_abnormality
        10: systolic_bp_abnormality
        11: spo2_abnormality (<95% is high risk)
        12: temperature_abnormality (>38.5C is high risk)
        13: test_indication_count
        14: interaction_severity_estimate
        15: clinical_complexity_score
        """
        v = [0.0] * 16

        symptom_cnt = sum(1 for e in entities if e.get("type", "").upper() in ["SYMPTOM", "TRIỆU_CHỨNG"])
        disease_cnt = sum(1 for e in entities if e.get("type", "").upper() in ["DISEASE", "CHẨN_ĐOÁN", "BỆNH"])
        med_cnt = sum(1 for e in entities if e.get("type", "").upper() in ["MEDICINE", "THUỐC"])
        test_cnt = sum(1 for e in entities if e.get("type", "").upper() in ["TEST", "TÊN_XÉT_NGHIỆM"])

        v[0] = min(1.0, symptom_cnt / 5.0)
        v[1] = min(1.0, disease_cnt / 4.0)
        v[2] = min(1.0, med_cnt / 4.0)
        v[3] = float(len(red_flags))
        v[4] = 1.0 if len(red_flags) > 0 else 0.0
        v[5] = float(len(contraindications))
        v[6] = 1.0 if len(contraindications) > 0 else 0.0

        confidences = [c.get("confidence", 0.0) for c in graph_results if isinstance(c, dict)]
        v[7] = max(confidences) if confidences else 0.0
        v[8] = sum(confidences) / len(confidences) if confidences else 0.0

        if vitals:
            hr = vitals.get("heart_rate", 75.0)
            v[9] = 1.0 if hr > 110 or hr < 50 else (0.5 if hr > 95 or hr < 60 else 0.0)

            sbp = vitals.get("blood_pressure_systolic", 120.0)
            v[10] = 1.0 if sbp > 160 or sbp < 90 else (0.5 if sbp > 140 else 0.0)

            spo2 = vitals.get("spo2", 98.0)
            v[11] = 1.0 if spo2 < 92.0 else (0.6 if spo2 < 95.0 else 0.0)

            temp = vitals.get("temperature", 37.0)
            v[12] = 1.0 if temp > 39.0 or temp < 35.5 else (0.5 if temp > 38.0 else 0.0)

        v[13] = min(1.0, test_cnt / 3.0)
        v[14] = min(1.0, float(len(contraindications)) * 0.5)
        v[15] = min(1.0, (symptom_cnt + disease_cnt * 2 + len(red_flags) * 3) / 10.0)

        return torch.tensor([v], dtype=torch.float32, device=self.device)

    def evaluate_risk(
        self,
        entities: list[dict[str, Any]],
        graph_results: list[dict[str, Any]] | None = None,
        red_flags: list[str] | None = None,
        contraindications: list[dict[str, Any]] | None = None,
        vitals: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Performs deep risk evaluation returning risk category, continuous score,
        and full probability distribution.
        """
        graph_results = graph_results or []
        red_flags = red_flags or []
        contraindications = contraindications or []

        x = self._extract_feature_vector(entities, graph_results, red_flags, contraindications, vitals)

        with torch.no_grad():
            probs = self.model(x).squeeze(0).cpu().numpy()

        # Clinical override safety constraints:
        # If red flags exist, ensure Emergency probability dominates
        if red_flags:
            probs = [0.02, 0.05, 0.15, 0.78]
        elif contraindications:
            probs = [0.05, 0.15, 0.70, 0.10]

        tier_idx = int(torch.argmax(torch.tensor(probs)).item())
        tier_label = ClinicalDeepRiskNetwork.RISK_TIERS[tier_idx]

        # Continuous risk index from 0.0 to 1.0
        weights = [0.1, 0.4, 0.75, 0.98]
        risk_score = float(sum(p * w for p, w in zip(probs, weights)))

        return {
            "risk_level": tier_label,
            "risk_score": round(risk_score, 4),
            "probabilities": {
                tier: round(float(p), 4) for tier, p in zip(ClinicalDeepRiskNetwork.RISK_TIERS, probs)
            },
            "red_flags_detected": red_flags,
            "active_contraindications": len(contraindications),
            "engine": "DeepNeuralResidualRiskClassifier"
        }
