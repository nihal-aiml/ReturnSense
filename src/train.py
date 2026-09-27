"""
ReturnSense -- Model Training Module
=====================================
Trains a LightGBM classifier to predict return probability per transaction.

Pipeline
--------
1. Load processed_returns.csv
2. Extract 7 features + target
3. Train/test split (80/20, stratified)
4. Apply SMOTE on training set only
5. Train LightGBM with tuned hyperparameters
6. Evaluate: AUC, Accuracy, Precision, Recall, F1
7. Log everything to MLflow
8. Save model to data/return_model.pkl
9. Generate SHAP feature importance and save

Expected AUC: 0.70 - 0.85
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import shap
import joblib
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from src.feature_engineering import load_processed_data, prepare_features, FEATURE_COLUMNS
except ImportError:
    from feature_engineering import load_processed_data, prepare_features, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "return_model.pkl"
SHAP_PATH = PROJECT_ROOT / "data" / "shap_importance.json"
METRICS_PATH = PROJECT_ROOT / "data" / "model_metrics.json"


def train_model():
    """
    Full training pipeline for the LightGBM return classifier.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    # Step 1: Load data
    df = load_processed_data()
    X, y = prepare_features(df)

    # Step 2: Train/test split -- 80/20, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[Train] Train set: {X_train.shape[0]} samples")
    print(f"[Train] Test set:  {X_test.shape[0]} samples")
    print(f"[Train] Train return rate: {y_train.mean():.4f}")
    print(f"[Train] Test return rate:  {y_test.mean():.4f}")

    # Step 3: SMOTE on training set only
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"\n[Train] After SMOTE -- Train set: {X_train_smote.shape[0]} samples")
    print(f"[Train] After SMOTE -- Class distribution: "
          f"0={int((y_train_smote == 0).sum())}, 1={int((y_train_smote == 1).sum())}")

    # Step 4: Train LightGBM
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train_smote, y_train_smote)
    print("\n[Train] LightGBM training complete.")

    # Step 5: Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "auc": round(auc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }

    print(f"\n{'='*50}")
    print(f"  MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # MLflow 3.x requires a database backend (file store is deprecated)
    mlflow_db = PROJECT_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db.as_posix()}")
    mlflow.set_experiment("ReturnSense-ReturnClassifier")

    with mlflow.start_run(run_name="lgbm_return_classifier"):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("features", ",".join(FEATURE_COLUMNS))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            "lgbm_return_model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
            ],
        )

    print(f"\n[Train] Logged to MLflow experiment: ReturnSense-ReturnClassifier")

    # Step 7: Save model to disk
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[Train] Model saved to: {MODEL_PATH}")

    # Step 8: SHAP feature importance
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # class 1 (returned)
        else:
            shap_vals = shap_values

        importance = np.abs(shap_vals).mean(axis=0)
        shap_dict = {
            feat: round(float(imp), 6)
            for feat, imp in zip(FEATURE_COLUMNS, importance)
        }
        shap_dict = dict(sorted(shap_dict.items(), key=lambda x: x[1], reverse=True))

        with open(SHAP_PATH, "w") as f:
            json.dump(shap_dict, f, indent=2)
        print(f"[Train] SHAP importance saved to: {SHAP_PATH}")
        print(f"[Train] Top features: {shap_dict}")
    except Exception as e:
        print(f"[Train] SHAP computation failed (non-critical): {e}")
        shap_dict = {feat: 0.0 for feat in FEATURE_COLUMNS}
        with open(SHAP_PATH, "w") as f:
            json.dump(shap_dict, f, indent=2)

    # Step 9: Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Train] Metrics saved to: {METRICS_PATH}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    metrics = train_model()
    print(f"\n[OK] Training complete. AUC: {metrics['auc']}")
