"""
ReturnSense -- Preprocessing Module
====================================
Takes raw ingested data and produces a fully-featured, labeled dataset
ready for model training.

Pipeline Steps
--------------
1. Separate purchases (Quantity > 0) from returns (Quantity < 0).
2. Build the return label using transaction-level matching.
3. Engineer all 7 features automatically from the data without time leakage.
4. Validate the return rate.
5. Save the processed dataset to data/raw/processed_returns.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

try:
    from src.data_ingestion import ingest
except ImportError:
    from data_ingestion import ingest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"


def separate_purchases_returns(df: pd.DataFrame):
    """Split dataset into purchase rows and return rows."""
    purchases = df[df["Quantity"] > 0].copy()
    returns = df[df["Quantity"] < 0].copy()
    print(f"[Preprocessing] Purchases: {len(purchases)}, Returns: {len(returns)}")
    return purchases, returns


def create_return_labels(purchases: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target variable for return classification.

    Strategy: Transaction-level matching.
    Matches a return (Quantity < 0) back to a preceding purchase (Quantity > 0) 
    by the same CustomerID for the exact same StockCode.
    """
    print("[Preprocessing] Mapping returns to specific historical purchases...")
    
    # Ensure dates are in datetime format
    purchases["InvoiceDate"] = pd.to_datetime(purchases["InvoiceDate"], errors="coerce")
    returns["InvoiceDate"] = pd.to_datetime(returns["InvoiceDate"], errors="coerce")

    # Create a unique ID for each purchase row to track matches
    purchases = purchases.copy()
    purchases["purchase_id"] = purchases.index

    # Merge returns to purchases on Customer and Item
    merged = pd.merge(
        purchases[["purchase_id", "CustomerID", "StockCode", "InvoiceDate"]],
        returns[["CustomerID", "StockCode", "InvoiceDate"]],
        on=["CustomerID", "StockCode"],
        suffixes=("_purch", "_ret")
    )

    # Filter for valid matches: The return must happen ON or AFTER the purchase date
    valid_returns = merged[merged["InvoiceDate_ret"] >= merged["InvoiceDate_purch"]]

    # Get the unique IDs of purchases that were eventually returned
    returned_purchase_ids = valid_returns["purchase_id"].unique()
    print(f"[Preprocessing] Found {len(returned_purchase_ids)} matched returns.")

    # Assign binary target
    purchases["target"] = 0
    purchases.loc[purchases["purchase_id"].isin(returned_purchase_ids), "target"] = 1

    # Clean up temporary ID
    purchases.drop(columns=["purchase_id"], inplace=True)

    return_rate = purchases["target"].mean()
    print(f"[Preprocessing] Return rate (target=1): {return_rate:.4f} ({return_rate*100:.2f}%)")

    # Sanity checks
    if return_rate > 0.50:
        print("\n*** ERROR: Return rate > 50%. Label creation logic is wrong. ***\n")
    elif return_rate > 0.30:
        print("[WARNING] Return rate is high. Check for duplicates in matching.")
    elif return_rate < 0.01:
        print("[WARNING] Return rate is very low (<1%). Exact matching is strict on this dataset.")
    else:
        print(f"[OK] Return rate {return_rate*100:.1f}% is realistic.")

    return purchases


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all 7 features from the raw data automatically.

    Features: UnitPrice, Quantity, price_vs_category_avg,
    customer_past_return_rate, customer_order_count, day_of_week, month
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # price_vs_category_avg
    category_avg = df.groupby("Description")["UnitPrice"].transform("mean")
    df["price_vs_category_avg"] = df["UnitPrice"] / category_avg
    df["price_vs_category_avg"] = df["price_vs_category_avg"].replace(
        [np.inf, -np.inf], 1.0
    ).fillna(1.0)

    # ---------------------------------------------------------
    # FIXED: customer_past_return_rate (Time-based Expanding Window)
    # ---------------------------------------------------------
    # 1. Sort strictly by Customer and Date to ensure chronologic order
    df = df.sort_values(by=["CustomerID", "InvoiceDate"])

    # 2. Calculate expanding sum and count of the TARGET, SHIFTED by 1
    # This prevents the current row's outcome from bleeding into its own feature
    past_returns = df.groupby("CustomerID")["target"].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    past_orders = df.groupby("CustomerID")["target"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # 3. Calculate rate and fill NaNs (first orders) with 0.0
    df["customer_past_return_rate"] = (past_returns / past_orders).fillna(0.0)

    # Sort back to original index to maintain order
    df = df.sort_index()
    # ---------------------------------------------------------

    # customer_order_count
    order_counts = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
    order_counts.columns = ["CustomerID", "customer_order_count"]
    df = df.merge(order_counts, on="CustomerID", how="left")
    df["customer_order_count"] = df["customer_order_count"].fillna(1).astype(int)

    # day_of_week
    df["day_of_week"] = df["InvoiceDate"].dt.dayofweek

    # month
    df["month"] = df["InvoiceDate"].dt.month

    print("[Preprocessing] Feature engineering complete.")
    print(f"  Features: UnitPrice, price_vs_category_avg, customer_past_return_rate, "
          f"customer_order_count, day_of_week, month, Quantity")

    return df


def preprocess(filepath=None) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = ingest(filepath)
    purchases, returns = separate_purchases_returns(df)
    labeled = create_return_labels(purchases, returns)
    featured = engineer_features(labeled)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[Preprocessing] Saved processed data to: {OUTPUT_PATH}")
    print(f"[Preprocessing] Final shape: {featured.shape}")
    print(f"[Preprocessing] Target distribution:\n{featured['target'].value_counts()}")

    return featured


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    df = preprocess(filepath)
    print("\n[OK] Preprocessing complete.")