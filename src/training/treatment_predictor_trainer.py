from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.services.patient_data_service import (
    CLINICAL_FEATURE_LABELS_VI,
    load_and_preprocess_patient_splits,
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "treatment_prediction",
)


def _select_validation_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Selects optimal F1 threshold strictly on the validation split."""
    best_t = 0.50
    best_f1 = -1.0
    for t in np.linspace(0.15, 0.85, 71):
        preds = (p_val >= t).astype(int)
        score = float(f1_score(y_val, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 4)


def train_and_evaluate_all_models() -> Dict[str, Any]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
        split_meta,
    ) = load_and_preprocess_patient_splits(random_state=42)

    models: Dict[str, Any] = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.88,
            colsample_bytree=0.90,
            reg_lambda=1.2,
            random_state=42,
            eval_metric="logloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.06,
            random_state=42,
            verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
        ),
        "Deep Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
        ),
    }

    benchmark_results: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}
    best_model_name = "XGBoost"
    best_composite = -1.0

    print("\n=====================================================================")
    print("  MEDFUSION-CXR E2 BENCHMARK: CLINICAL TABULAR BRANCH (TRAIN/VAL/TEST)")
    print("=====================================================================\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        val_proba = model.predict_proba(X_val)[:, 1]
        opt_thresh = _select_validation_threshold(y_val, val_proba)
        val_f1 = float(f1_score(y_val, (val_proba >= opt_thresh).astype(int), zero_division=0))
        val_auc = float(roc_auc_score(y_val, val_proba))

        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= opt_thresh).astype(int)

        acc = float(accuracy_score(y_test, test_pred))
        prec = float(precision_score(y_test, test_pred, zero_division=0))
        rec = float(recall_score(y_test, test_pred, zero_division=0))
        f1 = float(f1_score(y_test, test_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, test_proba))
        auprc = float(average_precision_score(y_test, test_proba))
        brier = float(brier_score_loss(y_test, test_proba))
        cm = confusion_matrix(y_test, test_pred).tolist()

        benchmark_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "auprc": round(auprc, 4),
            "brier_score": round(brier, 4),
            "optimal_threshold": opt_thresh,
            "val_f1_score": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4),
            "confusion_matrix": cm,
        }

        print(
            f"[{name:<26}] Thresh(Val): {opt_thresh:.2f} | "
            f"Test F1: {f1:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f}"
        )

        # Select champion strictly by validation performance (AUROC + F1)
        val_composite = 0.5 * val_auc + 0.5 * val_f1
        if val_composite > best_composite:
            best_composite = val_composite
            best_model_name = name

    os.makedirs(MODEL_DIR, exist_ok=True)
    xgb_model = trained_models["XGBoost"]
    best_model = trained_models[best_model_name]

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_treatment_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    # Compute exact global TreeSHAP feature importances on held-out Test split from XGBoost
    dtest = xgb.DMatrix(X_test, feature_names=preprocessor.feature_names)
    shap_matrix = xgb_model.get_booster().predict(dtest, pred_contribs=True)[:, :-1]
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    norm_shap = mean_abs_shap / (np.sum(mean_abs_shap) + 1e-8)

    shap_importances: Dict[str, float] = {}
    shap_importances_vi: Dict[str, float] = {}
    for fname, val in zip(preprocessor.feature_names, norm_shap):
        rounded = round(float(val), 4)
        shap_importances[fname] = rounded
        shap_importances_vi[CLINICAL_FEATURE_LABELS_VI.get(fname, fname)] = rounded

    gain_importances: Dict[str, float] = {}
    for fname, imp in zip(preprocessor.feature_names, xgb_model.feature_importances_):
        gain_importances[fname] = round(float(imp), 4)

    summary_report: Dict[str, Any] = {
        "experiment_id": "E2_CLINICAL_TABULAR_BENCHMARK",
        "primary_clinical_branch_model": "XGBoost",
        "best_model_name": best_model_name,
        "best_f1_score": benchmark_results[best_model_name]["f1_score"],
        "best_roc_auc": benchmark_results[best_model_name]["roc_auc"],
        "xgboost_metrics": benchmark_results["XGBoost"],
        "dataset_split_summary": split_meta,
        "benchmark_results": benchmark_results,
        "feature_importances": shap_importances_vi,
        "xgboost_shap_importances": shap_importances,
        "xgboost_gain_importances": gain_importances,
        "feature_names": preprocessor.feature_names,
    }

    report_path = os.path.join(MODEL_DIR, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)

    print(
        f"\nPRIMARY CLINICAL MODEL: XGBoost | Test AUROC = {benchmark_results['XGBoost']['roc_auc']:.4f}, "
        f"AUPRC = {benchmark_results['XGBoost']['auprc']:.4f}, F1 = {benchmark_results['XGBoost']['f1_score']:.4f}"
    )
    print(f"VALIDATION CHAMPION: '{best_model_name}' saved to '{MODEL_DIR}'")
    return summary_report


if __name__ == "__main__":
    train_and_evaluate_all_models()
