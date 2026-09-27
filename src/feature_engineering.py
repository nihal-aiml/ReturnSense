"""
ReturnSense -- Feature Engineering Module
==========================================
Prepares the feature matrix X and target vector y for model training.

Features (exact order used by the model)
----------------------------------------
1. UnitPrice
2. price_vs_category_avg
3. customer_past_return_rate
4. customer_order_count
5. day_of_week
6. month
7. Quantity
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"

# Canonical feature order -- used everywhere
FEATURE_COLUMNS = [
    "UnitPrice",
    "price_vs_category_avg",
    "customer_past_return_rate",
    "customer_order_count",
    "day_of_week",
    "month",
    "Quantity",
]

TARGET_COLUMN = "target"


def load_processed_data(filepath: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed returns CSV."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            f"Run `python -m src.preprocessing` first."
        )
    df = pd.read_csv(filepath)
    print(f"[FeatureEng] Loaded {len(df)} rows from {filepath}")
    return df


def validate_features(df: pd.DataFrame) -> None:
    """Check that all required feature columns exist."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def prepare_features(df: pd.DataFrame):
    """
    Extract the feature matrix X and target vector y from a processed DataFrame.

    Returns
    -------
    X : pd.DataFrame   -- Feature matrix with shape (n_samples, 7).
    y : pd.Series       -- Binary target vector.
    """
    validate_features(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Handle any remaining NaN / inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"[FeatureEng] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[FeatureEng] Target distribution: 0={int((y == 0).sum())}, 1={int((y == 1).sum())}")
    print(f"[FeatureEng] Return rate: {y.mean():.4f} ({y.mean()*100:.2f}%)")

    return X, y


def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the feature matrix X for inference (no target needed)."""
    validate_features(df)

    X = df[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X


def engineer_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all 7 features from a RAW uploaded DataFrame.

    This is the key function used by predict.py and the API when a user
    uploads a new file. It does NOT require preprocessing.py to have
    been run -- it computes everything on the fly.
    """
    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Keep only purchases (Quantity > 0) for prediction
    purchases = df[df["Quantity"] > 0].copy()
    returns = df[df["Quantity"] < 0].copy()

    if purchases.empty:
        raise ValueError("No purchase rows found (Quantity > 0). Cannot generate predictions.")

    # price_vs_category_avg
    cat_avg = purchases.groupby("Description")["UnitPrice"].transform("mean")
    purchases["price_vs_category_avg"] = purchases["UnitPrice"] / cat_avg
    purchases["price_vs_category_avg"] = purchases["price_vs_category_avg"].replace(
        [np.inf, -np.inf], 1.0
    ).fillna(1.0)

    # customer_past_return_rate
    if not returns.empty:
        return_pairs = returns[["CustomerID", "StockCode"]].drop_duplicates()
        return_pairs["was_returned"] = 1
        purchases = purchases.merge(return_pairs, on=["CustomerID", "StockCode"], how="left")
        purchases["was_returned"] = purchases["was_returned"].fillna(0)

        # 1. Sort strictly by Customer and Date to ensure chronologic order
        purchases = purchases.sort_values(by=["CustomerID", "InvoiceDate"])

        # 2. Calculate expanding sum and count, SHIFTED by 1 so the current row is excluded
        past_returns = purchases.groupby("CustomerID")["was_returned"].transform(
            lambda x: x.shift(1).expanding().sum()
        )
        past_orders = purchases.groupby("CustomerID")["was_returned"].transform(
            lambda x: x.shift(1).expanding().count()
        )

        # 3. Calculate rate and fill NaNs (which happen on the customer's very first order) with 0.0
        purchases["customer_past_return_rate"] = (past_returns / past_orders).fillna(0.0)

        # Clean up the temporary column and sort back to original index
        purchases.drop(columns=["was_returned"], inplace=True, errors="ignore")
        purchases = purchases.sort_index()
    else:
        purchases["customer_past_return_rate"] = 0.0

    purchases["customer_past_return_rate"] = purchases["customer_past_return_rate"].fillna(0.0)

    # customer_order_count
    order_counts = purchases.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    order_counts.columns = ["CustomerID", "customer_order_count"]
    purchases = purchases.merge(order_counts, on="CustomerID", how="left")
    purchases["customer_order_count"] = purchases["customer_order_count"].fillna(1).astype(int)

    # day_of_week
    purchases["day_of_week"] = purchases["InvoiceDate"].dt.dayofweek

    # month
    purchases["month"] = purchases["InvoiceDate"].dt.month

    return purchases


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_processed_data()
    X, y = prepare_features(df)
    print(f"\n[OK] Feature engineering complete.")
    print(f"  Features: {list(X.columns)}")
    print(f"  Samples:  {len(X)}")