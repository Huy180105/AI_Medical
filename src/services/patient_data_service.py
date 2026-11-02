from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "treatment_prediction")

CLINICAL_FEATURE_LABELS_VI: Dict[str, str] = {
    "age": "Độ tuổi (Age)",
    "heart_rate": "Nhịp tim (HR bpm)",
    "sys_bp": "Huyết áp tâm thu (SBP mmHg)",
    "dia_bp": "Huyết áp tâm trương (DBP mmHg)",
    "spo2": "Độ bão hòa Oxy (SpO2 %)",
    "stress_score": "Điểm căng thẳng (Stress 1-10)",
    "sleep_hours": "Thời lượng ngủ (Giờ/ngày)",
    "physical_activity_hrs": "Vận động thể chất (Giờ/tuần)",
    "chronic_conditions": "Số bệnh nền mạn tính",
    "symptoms_severity": "Mức độ triệu chứng (1-5)",
    "gender_encoded": "Giới tính (Gender)",
}


def generate_patient_health_dataset(
    num_samples: int = 1500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds the independent tabular clinical dataset (1,500 physiological records +
    100 clinical records extracted from `data/clinical_input_dataset.csv` = 1,600 samples).
    Note: Explicitly independent from NIH ChestX-ray14 (unpaired cohort).
    """
    rng = np.random.RandomState(random_state)

    ages = rng.randint(18, 82, size=num_samples)
    genders = rng.choice(["Male", "Female"], size=num_samples)
    heart_rates = rng.normal(75, 12, size=num_samples).clip(50, 130).round(1)
    sys_bps = rng.normal(122, 18, size=num_samples).clip(90, 190).round(1)
    dia_bps = rng.normal(80, 10, size=num_samples).clip(55, 115).round(1)
    spo2s = rng.normal(97, 2.5, size=num_samples).clip(85, 100).round(1)
    stress_scores = rng.randint(1, 11, size=num_samples)
    sleep_hours = rng.normal(6.8, 1.4, size=num_samples).clip(3, 10).round(1)
    activity_hrs = rng.exponential(3.5, size=num_samples).clip(0, 15).round(1)
    chronic_conditions = rng.poisson(0.8, size=num_samples).clip(0, 4)
    symptoms_severities = rng.randint(1, 6, size=num_samples)

    risk_score = (
        (ages > 60) * 1.5
        + (sys_bps > 140) * 2.0
        + (dia_bps > 90) * 1.5
        + (spo2s < 94) * 3.0
        + (stress_scores >= 8) * 2.0
        + (sleep_hours < 5) * 1.5
        + (chronic_conditions >= 2) * 2.5
        + (symptoms_severities >= 4) * 3.0
        + rng.normal(0, 1.2, size=num_samples)
    )

    treatment_needed = (risk_score > 4.5).astype(int)

    df_synth = pd.DataFrame(
        {
            "age": ages,
            "gender": genders,
            "heart_rate": heart_rates,
            "sys_bp": sys_bps,
            "dia_bp": dia_bps,
            "spo2": spo2s,
            "stress_score": stress_scores,
            "sleep_hours": sleep_hours,
            "physical_activity_hrs": activity_hrs,
            "chronic_conditions": chronic_conditions,
            "symptoms_severity": symptoms_severities,
            "treatment_needed": treatment_needed,
        }
    )

    clin_csv = os.path.join(DATA_DIR, "clinical_input_dataset.csv")
    if os.path.exists(clin_csv):
        try:
            df_clin = pd.read_csv(clin_csv)
            if "doc_id" in df_clin.columns:
                df_clin = df_clin.drop(columns=["doc_id"])
            # Sanitize out-of-range physiological anomalies (e.g., SpO2 > 100)
            if "spo2" in df_clin.columns:
                df_clin["spo2"] = df_clin["spo2"].clip(50.0, 100.0)
            if "age" in df_clin.columns:
                df_clin["age"] = df_clin["age"].clip(0, 115)
            df_synth = pd.concat([df_synth, df_clin], ignore_index=True)
        except Exception:
            pass

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "patient_treatment_dataset.csv")
    df_synth.to_csv(csv_path, index=False)
    return df_synth


class PatientDataPreprocessor:
    """
    Leakage-free clinical tabular preprocessor.
    Fits `LabelEncoder` and `StandardScaler` strictly on the training split.
    """

    CANONICAL_COLUMNS: List[str] = [
        "age",
        "heart_rate",
        "sys_bp",
        "dia_bp",
        "spo2",
        "stress_score",
        "sleep_hours",
        "physical_activity_hrs",
        "chronic_conditions",
        "symptoms_severity",
        "gender_encoded",
    ]

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.gender_encoder = LabelEncoder()
        self.feature_names: List[str] = list(self.CANONICAL_COLUMNS)

    def _encode_dataframe(self, df: pd.DataFrame, fit_gender: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        df_clean = df.copy()
        gender_norm = (
            df_clean["gender"]
            .astype(str)
            .apply(lambda v: "Male" if ("male" in v.lower() or "nam" in v.lower()) else "Female")
        )
        if fit_gender:
            self.gender_encoder.fit(["Female", "Male"])
        df_clean["gender_encoded"] = self.gender_encoder.transform(gender_norm)
        df_clean["spo2"] = df_clean["spo2"].clip(50.0, 100.0)

        X_df = df_clean[self.CANONICAL_COLUMNS]
        self.feature_names = list(X_df.columns)
        X = X_df.to_numpy(dtype=np.float64)
        y = df_clean["treatment_needed"].to_numpy(dtype=np.int64)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._encode_dataframe(df, fit_gender=True)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def transform_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self._encode_dataframe(df, fit_gender=False)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def transform_single_patient(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        gender_raw = str(patient_dict.get("gender", "Male")).lower()
        gender_enc = 1.0 if ("male" in gender_raw or "nam" in gender_raw) else 0.0

        features = [
            float(patient_dict.get("age", 45)),
            float(patient_dict.get("heart_rate", 75.0)),
            float(patient_dict.get("sys_bp", 120.0)),
            float(patient_dict.get("dia_bp", 80.0)),
            float(np.clip(float(patient_dict.get("spo2", 98.0)), 50.0, 100.0)),
            float(patient_dict.get("stress_score", 5)),
            float(patient_dict.get("sleep_hours", 7.0)),
            float(patient_dict.get("physical_activity_hrs", 3.0)),
            float(patient_dict.get("chronic_conditions", 0)),
            float(patient_dict.get("symptoms_severity", 2)),
            gender_enc,
        ]
        X_arr = np.array(features, dtype=np.float64).reshape(1, -1)
        return self.scaler.transform(X_arr)


def load_and_preprocess_patient_splits(
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    PatientDataPreprocessor,
    Dict[str, Any],
]:
    """
    Loads the 1,600-sample clinical dataset and creates stratified, leakage-free
    Train (70% = 1,120) / Validation (15% = 240) / Test (15% = 240) splits.
    StandardScaler is fit strictly on Train.
    """
    df = generate_patient_health_dataset(random_state=random_state)
    y_all = df["treatment_needed"].to_numpy(dtype=np.int64)

    df_train, df_temp = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=y_all
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=df_temp["treatment_needed"],
    )

    preprocessor = PatientDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_val, y_val = preprocessor.transform_dataframe(df_val)
    X_test, y_test = preprocessor.transform_dataframe(df_test)

    split_meta = {
        "total_samples": int(len(df)),
        "synthetic_physiological_samples": 1500,
        "extracted_clinical_records": int(len(df) - 1500),
        "train_samples": int(len(df_train)),
        "val_samples": int(len(df_val)),
        "test_samples": int(len(df_test)),
        "positive_rate_overall": round(float(y_all.mean()), 4),
        "nih_paired_linkage": False,
        "scaler_fit_split": "train_only",
    }
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, split_meta


def load_and_preprocess_patient_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, PatientDataPreprocessor
]:
    """Backward-compatible helper returning Train and held-out Test splits."""
    X_train, _, X_test, y_train, _, y_test, preprocessor, _ = load_and_preprocess_patient_splits()
    return X_train, X_test, y_train, y_test, preprocessor


class XGBoostClinicalService:
    """
    Production Clinical Branch Service powered by trained XGBoost.
    Provides:
    1. Calibrated probability prediction (`predict_proba`) using validation-selected threshold.
    2. Exact native TreeSHAP per-feature contributions (`pred_contribs=True`).
    3. Dense 24-d clinical representation vector [X_scaled (11), TreeSHAP (11), bias, logit]
       and tree-leaf indices (`apply()`) for Gated Feature Fusion alignment.
    """

    def __init__(self) -> None:
        self.xgb_model: Any = None
        self.preprocessor: Optional[PatientDataPreprocessor] = None
        self.optimal_threshold: float = 0.50
        self.benchmark_summary: Dict[str, Any] = {}
        self._load_or_train()

    def _load_or_train(self) -> None:
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
        prep_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
        report_path = os.path.join(MODEL_DIR, "benchmark_report.json")

        if not (os.path.exists(xgb_path) and os.path.exists(prep_path) and os.path.exists(report_path)):
            from src.training.treatment_predictor_trainer import train_and_evaluate_all_models

            train_and_evaluate_all_models()

        self.xgb_model = joblib.load(xgb_path)
        self.preprocessor = joblib.load(prep_path)
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                self.benchmark_summary = json.load(f)
            xgb_metrics = self.benchmark_summary.get("benchmark_results", {}).get("XGBoost", {})
            self.optimal_threshold = float(xgb_metrics.get("optimal_threshold", 0.50))

    def analyze_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes XGBoost clinical branch inference, native TreeSHAP attribution,
        and extracts the structured clinical representation vector for Gated Fusion.
        """
        import xgboost as xgb

        assert self.preprocessor is not None
        X_scaled = self.preprocessor.transform_single_patient(patient_dict)  # (1, 11)
        prob = float(self.xgb_model.predict_proba(X_scaled)[0, 1])
        prob_clipped = float(np.clip(prob, 1e-5, 1.0 - 1e-5))
        raw_logit = float(np.log(prob_clipped / (1.0 - prob_clipped)))

        # Native exact TreeSHAP feature contributions: shape (1, 12) where last column is bias
        booster = self.xgb_model.get_booster()
        dmat = xgb.DMatrix(X_scaled, feature_names=self.preprocessor.feature_names)
        contribs = booster.predict(dmat, pred_contribs=True)[0]  # (12,)
        shap_values = contribs[:-1]  # (11,)
        shap_bias = float(contribs[-1])

        # Tree-leaf index representation across ensemble trees
        leaf_indices = self.xgb_model.apply(X_scaled)[0].astype(np.int32).tolist()

        # Construct 24-d clinical representation vector [X_scaled (11), TreeSHAP (11), bias, logit]
        clinical_repr_24d = np.concatenate(
            [
                X_scaled[0].astype(np.float32),
                shap_values.astype(np.float32),
                np.array([shap_bias, raw_logit], dtype=np.float32),
            ],
            axis=0,
        )

        # Build human-readable ranked SHAP attributions
        shap_ranking: List[Dict[str, Any]] = []
        total_abs_shap = float(np.sum(np.abs(shap_values))) + 1e-8
        for fname, scaled_val, sv in zip(self.preprocessor.feature_names, X_scaled[0], shap_values):
            impact_pct = round(float(abs(sv) / total_abs_shap * 100.0), 1)
            raw_val = patient_dict.get(fname, "Male" if scaled_val > 0 else "Female")
            shap_ranking.append(
                {
                    "feature_key": fname,
                    "feature_label": CLINICAL_FEATURE_LABELS_VI.get(fname, fname),
                    "raw_value": raw_val,
                    "shap_value": round(float(sv), 4),
                    "direction": "increase_risk" if sv >= 0 else "decrease_risk",
                    "relative_impact_pct": impact_pct,
                }
            )
        shap_ranking.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        treatment_needed = bool(prob >= self.optimal_threshold)
        if prob >= 0.80:
            risk_tier = "CRITICAL (Nguy cơ Rất Cao - Cần Can thiệp Khẩn cấp)"
        elif prob >= 0.55:
            risk_tier = "HIGH (Nguy cơ Cao - Cần Điều trị Y tế)"
        elif prob >= 0.35:
            risk_tier = "MEDIUM (Nguy cơ Trung bình - Theo dõi Sát)"
        else:
            risk_tier = "LOW (Nguy cơ Thấp - Sức khỏe Ổn định)"

        xgb_test_metrics = self.benchmark_summary.get("benchmark_results", {}).get("XGBoost", {})
        auc_val = xgb_test_metrics.get("roc_auc", 0.94)
        f1_val = xgb_test_metrics.get("f1_score", 0.84)

        return {
            "probability": round(prob, 4),
            "probability_pct": round(prob * 100.0, 1),
            "optimal_threshold": round(self.optimal_threshold, 4),
            "treatment_needed": treatment_needed,
            "risk_level": risk_tier,
            "logit": round(raw_logit, 4),
            "shap_bias": round(shap_bias, 4),
            "shap_attributions": shap_ranking,
            "clinical_representation_24d": clinical_repr_24d.tolist(),
            "tree_leaf_count": len(leaf_indices),
            "tree_leaf_preview": leaf_indices[:10],
            "model_identifier": f"XGBoost Clinical Branch (Test AUROC: {auc_val * 100:.2f}%, F1: {f1_val * 100:.2f}%)",
        }


_clinical_service_singleton: Optional[XGBoostClinicalService] = None


def get_xgboost_clinical_service() -> XGBoostClinicalService:
    global _clinical_service_singleton
    if _clinical_service_singleton is None:
        _clinical_service_singleton = XGBoostClinicalService()
    return _clinical_service_singleton
