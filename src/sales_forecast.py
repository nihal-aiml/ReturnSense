"""
ReturnSense -- Sales Forecast Module
=====================================
Uses Facebook Prophet to forecast daily gross sales.

Pipeline
--------
1. Load dataset -- tries UCI Online Retail data aggregated by date.
2. Aggregate to daily revenue: sum(Quantity * UnitPrice) per day.
3. Train Prophet model on historical daily revenue.
4. Generate a 30-day forecast.
5. Save Prophet model to data/prophet_model.pkl.
6. Save forecast CSV to data/sales_forecast.csv.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

try:
    from src.data_ingestion import ingest
except ImportError:
    from data_ingestion import ingest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROPHET_MODEL_PATH = PROJECT_ROOT / "data" / "prophet_model.pkl"
FORECAST_CSV_PATH = PROJECT_ROOT / "data" / "sales_forecast.csv"


def prepare_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transaction data into daily revenue."""
    purchases = df[df["Quantity"] > 0].copy()
    purchases["revenue"] = purchases["Quantity"] * purchases["UnitPrice"]

    daily = purchases.groupby(purchases["InvoiceDate"].dt.date)["revenue"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    # Remove extreme outliers (> 3 std from mean)
    mean_y = daily["y"].mean()
    std_y = daily["y"].std()
    daily = daily[(daily["y"] > 0) & (daily["y"] < mean_y + 3 * std_y)]

    print(f"[SalesForecast] Daily sales data: {len(daily)} days")
    print(f"[SalesForecast] Date range: {daily['ds'].min()} -> {daily['ds'].max()}")
    print(f"[SalesForecast] Avg daily revenue: ${daily['y'].mean():,.2f}")

    return daily.reset_index(drop=True)


def train_prophet(daily_sales: pd.DataFrame, forecast_days: int = 30):
    """Train a Prophet model and generate a forecast."""
    print(f"\n[SalesForecast] Training Prophet model...")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    print(f"[SalesForecast] Forecast generated for {forecast_days} days ahead.")
    return model, forecast


def run_sales_forecast(filepath=None, forecast_days: int = 30):
    """Full sales forecasting pipeline."""
    df = ingest(filepath)
    daily_sales = prepare_daily_sales(df)

    if len(daily_sales) < 30:
        print("[SalesForecast] WARNING: Less than 30 days of data.")

    model, forecast = train_prophet(daily_sales, forecast_days)

    # Save model
    PROPHET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PROPHET_MODEL_PATH)
    print(f"[SalesForecast] Model saved to: {PROPHET_MODEL_PATH}")

    # Save forecast CSV
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_output.columns = ["date", "predicted_revenue", "lower_bound", "upper_bound"]
    forecast_output.to_csv(FORECAST_CSV_PATH, index=False)
    print(f"[SalesForecast] Forecast saved to: {FORECAST_CSV_PATH}")

    future_only = forecast_output.tail(forecast_days)
    print(f"\n{'='*50}")
    print(f"  30-DAY FORECAST SUMMARY")
    print(f"{'='*50}")
    print(f"  Total predicted revenue: ${future_only['predicted_revenue'].sum():,.2f}")
    print(f"  Avg daily revenue:       ${future_only['predicted_revenue'].mean():,.2f}")
    print(f"{'='*50}")

    return forecast_output


if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    forecast = run_sales_forecast(filepath)
    print(f"\n[OK] Sales forecast complete. {len(forecast)} forecast rows generated.")
