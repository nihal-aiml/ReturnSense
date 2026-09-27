"""
ReturnSense -- Prediction Module
=================================
Core business logic: computes Net Demand for every product.

Formula: Net Demand = Predicted Gross Sales x (1 - Return Probability)

This module:
1. Takes a raw DataFrame (from user upload or CSV)
2. Engineers all features automatically (no manual input)
3. Loads the trained LightGBM model
4. Predicts return probability for every transaction row
5. Aggregates results by StockCode
6. Returns a clean result DataFrame

Risk Levels:
  - HIGH:   return_probability > 0.30
  - MEDIUM: return_probability > 0.15
  - LOW:    return_probability <= 0.15
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

try:
    from src.feature_engineering import (
        engineer_features_from_raw,
        FEATURE_COLUMNS,
        prepare_features_for_prediction,
    )
except ImportError:
    from feature_engineering import (
        engineer_features_from_raw,
        FEATURE_COLUMNS,
        prepare_features_for_prediction,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "return_model.pkl"
PROPHET_MODEL_PATH = PROJECT_ROOT / "data" / "prophet_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained LightGBM return classifier."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run `python -m src.train` first to train the model."
        )
    model = joblib.load(model_path)
    print(f"[Predict] Loaded model from: {model_path}")
    return model


def classify_risk(prob: float) -> str:
    """Assign a risk level based on return probability threshold."""
    if prob > 0.30:
        return "HIGH"
    elif prob > 0.15:
        return "MEDIUM"
    else:
        return "LOW"


def compute_net_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main prediction function -- takes raw data and returns net demand results.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data with columns: InvoiceNo, StockCode,
        Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.

    Returns
    -------
    pd.DataFrame
        Aggregated results by StockCode with columns:
        stock_code, description, gross_sales, return_probability,
        net_demand, units_saved, risk_level.
    """
    required_cols = ["InvoiceNo", "StockCode", "Description", "Quantity",
                     "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {required_cols}"
        )

    # Step 1: Clean basic issues
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df = df[df["UnitPrice"] > 0]

    # Step 2: Engineer features from raw data
    featured = engineer_features_from_raw(df)

    # Step 3: Load model and predict
    model = load_model()
    X = prepare_features_for_prediction(featured)
    return_probs = model.predict_proba(X)[:, 1]
    featured["return_probability"] = return_probs

    # Step 4: Aggregate by StockCode
    agg = featured.groupby("StockCode").agg(
        description=("Description", "first"),
        gross_sales=("Quantity", "sum"),
        return_probability=("return_probability", "mean"),
    ).reset_index()

    # Step 5: Compute net demand
    # Net Demand = Gross Sales x (1 - Return Probability)
    agg["net_demand"] = (agg["gross_sales"] * (1 - agg["return_probability"])).round(0).astype(int)
    agg["units_saved"] = (agg["gross_sales"] - agg["net_demand"]).astype(int)
    agg["risk_level"] = agg["return_probability"].apply(classify_risk)

    agg["return_probability"] = agg["return_probability"].round(4)
    agg = agg.rename(columns={"StockCode": "stock_code"})
    agg = agg.sort_values("return_probability", ascending=False).reset_index(drop=True)

    print(f"\n[Predict] Results:")
    print(f"  Total products: {len(agg)}")
    print(f"  HIGH risk:   {(agg['risk_level'] == 'HIGH').sum()}")
    print(f"  MEDIUM risk: {(agg['risk_level'] == 'MEDIUM').sum()}")
    print(f"  LOW risk:    {(agg['risk_level'] == 'LOW').sum()}")
    print(f"  Total units saved: {agg['units_saved'].sum()}")
    print(f"  Avg return probability: {agg['return_probability'].mean():.4f}")

    return agg


def compute_what_if(product_row: dict, reduction_pcts: list = None) -> list:
    """
    Compute what-if scenarios for a selected product.
    Shows what happens to net demand if the return rate drops by various percentages.
    """
    if reduction_pcts is None:
        reduction_pcts = [10, 20, 30]

    gross = product_row["gross_sales"]
    base_prob = product_row["return_probability"]
    base_net = int(gross * (1 - base_prob))

    scenarios = [
        {
            "scenario": "Current",
            "return_probability": round(base_prob, 4),
            "net_demand": base_net,
            "units_saved": int(gross - base_net),
        }
    ]

    for pct in reduction_pcts:
        new_prob = base_prob * (1 - pct / 100)
        new_net = int(gross * (1 - new_prob))
        scenarios.append({
            "scenario": f"Return rate -{pct}%",
            "return_probability": round(new_prob, 4),
            "net_demand": new_net,
            "units_saved": int(gross - new_net),
        })

    return scenarios


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        ext = filepath.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(filepath, encoding="latin-1")
        else:
            df = pd.read_excel(filepath, engine="openpyxl")
    else:
        try:
            from src.data_ingestion import ingest
        except ImportError:
            from data_ingestion import ingest
        df = ingest()

    results = compute_net_demand(df)
    print(f"\n[OK] Prediction complete. {len(results)} products scored.")
    print(results.head(10).to_string(index=False))
