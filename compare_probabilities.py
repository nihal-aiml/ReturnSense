"""
ReturnSense — Manual vs Model Return Probability Comparison
===========================================================
Calculates return probability manually from raw data
and compares it against what LightGBM model predicts.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "raw"
MODEL_PATH   = PROJECT_ROOT / "data" / "return_model.pkl"

FEATURES = [
    "UnitPrice",
    "price_vs_category_avg",
    "customer_past_return_rate",
    "customer_order_count",
    "day_of_week",
    "month",
    "Quantity",
]

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Loading dataset...")

# Try synthetic first, then original
for fname in ["ReturnSense_Synthetic.xlsx", "Online Retail.xlsx"]:
    fpath = DATA_DIR / fname
    if fpath.exists():
        print(f"Using: {fname}")
        df = pd.read_excel(fpath)
        break
else:
    # Try processed CSV
    csv = DATA_DIR / "processed_returns.csv"
    if csv.exists():
        print("Using: processed_returns.csv")
        df = None
        processed = pd.read_csv(csv)
    else:
        print("ERROR: No dataset found in data/raw/")
        exit()

# ── Feature engineering ───────────────────────────────────────────────────────
if df is not None:
    df = df.dropna(subset=["CustomerID"])
    df["is_returned"] = (df["Quantity"] < 0).astype(int)
    purchases = df[df["Quantity"] > 0].copy()

    # Customer-level features
    cust_return_rate  = df.groupby("CustomerID")["is_returned"].mean()
    cust_order_count  = df.groupby("CustomerID")["InvoiceNo"].nunique()

    purchases["customer_past_return_rate"] = purchases["CustomerID"].map(cust_return_rate)
    purchases["customer_order_count"]      = purchases["CustomerID"].map(cust_order_count)

    # Price vs category average
    cat_avg = purchases.groupby("Description")["UnitPrice"].transform("mean")
    purchases["price_vs_category_avg"] = purchases["UnitPrice"] / cat_avg.replace(0, 1)

    # Date features
    purchases["InvoiceDate"] = pd.to_datetime(purchases["InvoiceDate"])
    purchases["day_of_week"] = purchases["InvoiceDate"].dt.dayofweek
    purchases["month"]       = purchases["InvoiceDate"].dt.month

    # Target label
    returns = df[df["Quantity"] < 0][["CustomerID", "StockCode"]].drop_duplicates()
    returns["was_returned"] = 1
    purchases = purchases.merge(returns, on=["CustomerID", "StockCode"], how="left")
    purchases["target"] = purchases["was_returned"].fillna(0).astype(int)

    processed = purchases.copy()

# ── Manual probability calculation ───────────────────────────────────────────
print("Calculating manual return probabilities...")

# Per product: actual return rate from data
product_actual = (
    processed.groupby("StockCode")
    .agg(
        Description=("Description", "first"),
        total_purchases=("target", "count"),
        actual_returns=("target", "sum"),
        avg_price=("UnitPrice", "mean"),
        avg_cust_return_rate=("customer_past_return_rate", "mean"),
        avg_order_count=("customer_order_count", "mean"),
        avg_price_vs_avg=("price_vs_category_avg", "mean"),
    )
    .reset_index()
)

product_actual["manual_return_prob"] = (
    product_actual["actual_returns"] / product_actual["total_purchases"]
)

# Filter products with enough data
product_actual = product_actual[product_actual["total_purchases"] >= 10].copy()

# ── Model predictions ─────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print("ERROR: Model not found. Run python src/train.py first.")
    exit()

print("Loading model and predicting...")
model = joblib.load(MODEL_PATH)

# Build feature rows using average values per product
feature_rows = []
for _, row in product_actual.iterrows():
    feature_rows.append({
        "UnitPrice":               row["avg_price"],
        "price_vs_category_avg":   row["avg_price_vs_avg"],
        "customer_past_return_rate": row["avg_cust_return_rate"],
        "customer_order_count":    row["avg_order_count"],
        "day_of_week":             1,    # Tuesday (mid-week average)
        "month":                   6,    # June (mid-year average)
        "Quantity":                1,
    })

X_pred = pd.DataFrame(feature_rows)
model_probs = model.predict_proba(X_pred)[:, 1]
product_actual["model_return_prob"] = model_probs

# ── Compute difference ────────────────────────────────────────────────────────
product_actual["difference"] = (
    product_actual["model_return_prob"] - product_actual["manual_return_prob"]
)
product_actual["abs_diff"] = product_actual["difference"].abs()

# Sort by manual return prob descending
product_actual = product_actual.sort_values("manual_return_prob", ascending=False)

# ── Print comparison table ────────────────────────────────────────────────────
print()
print("=" * 95)
print("   MANUAL vs MODEL RETURN PROBABILITY COMPARISON")
print("=" * 95)
print(f"  {'Product':<35} {'Purchases':>9} {'Manual%':>9} {'Model%':>9} {'Diff%':>8} {'Match?':>8}")
print("  " + "-" * 83)

matches     = 0
close_count = 0
total       = len(product_actual)

for _, row in product_actual.iterrows():
    desc     = str(row["Description"])[:33]
    manual   = row["manual_return_prob"] * 100
    model_p  = row["model_return_prob"]  * 100
    diff     = row["difference"] * 100
    purchases = int(row["total_purchases"])

    # Match = within 10 percentage points
    if abs(diff) <= 10:
        match = "CLOSE"
        close_count += 1
    elif abs(diff) <= 20:
        match = "OK"
        matches += 1
    else:
        match = "OFF"

    print(f"  {desc:<35} {purchases:>9,} {manual:>8.1f}% {model_p:>8.1f}% {diff:>+7.1f}% {match:>8}")

print("  " + "-" * 83)
print()

# ── Summary stats ─────────────────────────────────────────────────────────────
mae  = product_actual["abs_diff"].mean() * 100
corr = product_actual[["manual_return_prob","model_return_prob"]].corr().iloc[0,1]

print("=" * 60)
print("   SUMMARY")
print("=" * 60)
print(f"  Total products compared      : {total}")
print(f"  CLOSE match (within 10%)     : {close_count} ({close_count/total*100:.0f}%)")
print(f"  OK match (within 20%)        : {matches} ({matches/total*100:.0f}%)")
print(f"  Mean Absolute Error (MAE)    : {mae:.2f}%")
print(f"  Correlation (manual vs model): {corr:.3f}")
print()

if corr > 0.7:
    print("  VERDICT: Model tracks manual probability well")
elif corr > 0.4:
    print("  VERDICT: Moderate alignment — model captures general trend")
else:
    print("  VERDICT: Low correlation — model may need more training data")

print()

# ── Top 5 highest return risk products ───────────────────────────────────────
print("=" * 60)
print("  TOP 5 HIGH RETURN RISK PRODUCTS (by manual rate)")
print("=" * 60)
top5 = product_actual.nlargest(5, "manual_return_prob")
for _, row in top5.iterrows():
    print(f"  {str(row['Description'])[:40]:<40}")
    print(f"    Manual: {row['manual_return_prob']*100:.1f}%  |  Model: {row['model_return_prob']*100:.1f}%  |  Diff: {row['difference']*100:+.1f}%")

print()

# ── Top 5 lowest return risk products ────────────────────────────────────────
print("=" * 60)
print("  TOP 5 LOW RETURN RISK PRODUCTS (by manual rate)")
print("=" * 60)
bot5 = product_actual.nsmallest(5, "manual_return_prob")
for _, row in bot5.iterrows():
    print(f"  {str(row['Description'])[:40]:<40}")
    print(f"    Manual: {row['manual_return_prob']*100:.1f}%  |  Model: {row['model_return_prob']*100:.1f}%  |  Diff: {row['difference']*100:+.1f}%")

print()
print("=" * 60)
print("  HOW TO READ THIS:")
print("  Manual%  = actual return rate calculated from raw data")
print("  Model%   = what LightGBM predicts using learned patterns")
print("  Diff%    = Model - Manual (+ means model overestimates)")
print("  CLOSE    = within 10% — good match")
print("  OK       = within 20% — acceptable")
print("  OFF      = more than 20% difference — investigate")
print("=" * 60)

# ── Save to CSV ───────────────────────────────────────────────────────────────
save_path = PROJECT_ROOT / "data" / "prob_comparison.csv"
product_actual[[
    "StockCode", "Description", "total_purchases",
    "manual_return_prob", "model_return_prob", "difference"
]].to_csv(save_path, index=False)
print(f"\n  Full comparison saved to: {save_path}")