"""
ReturnSense — FastAPI Backend
==============================
REST API that serves predictions and integrates all pipeline components.

Endpoints
---------
GET  /          → Health check
POST /upload    → Upload CSV/Excel, run prediction, return results
GET  /summary   → Return model metrics from MLflow/saved metrics
GET  /drift     → Run Evidently drift check
POST /retrain   → Trigger full retraining pipeline
GET  /shap      → Return SHAP feature importance values
GET  /forecast  → Return Prophet sales forecast
POST /forecast/train → Train the Prophet model
GET  /validate  → Validate the processed dataset
POST /what-if   → Compute what-if scenarios

CORS is enabled so the HTML frontend can call the API directly
from a browser (file:// or any origin).
"""

import pandas as pd
import numpy as np
import json
import io
import traceback
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path so we can import src modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import compute_net_demand, compute_what_if
from src.monitor import run_drift_check
from src.validate import run_validation

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ReturnSense API",
    description="Return-Aware Net Demand Forecasting for E-commerce",
    version="1.1.0",
)

# Enable CORS — allow all origins so index.html can call from file:// or localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = PROJECT_ROOT / "data" / "return_model.pkl"
METRICS_PATH = PROJECT_ROOT / "data" / "model_metrics.json"
SHAP_PATH = PROJECT_ROOT / "data" / "shap_importance.json"
FORECAST_CSV_PATH = PROJECT_ROOT / "data" / "sales_forecast.csv"
PROPHET_MODEL_PATH = PROJECT_ROOT / "data" / "prophet_model.pkl"


# ---------------------------------------------------------------------------
# GET / — Health check
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ReturnSense API running"}


# ---------------------------------------------------------------------------
# POST /upload — Main prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accept a CSV or Excel file upload, run the full prediction pipeline,
    and return net demand results.

    Flow:
    1. Read uploaded file into pandas DataFrame
    2. Run data validation
    3. Run preprocessing + feature engineering automatically
    4. Run LightGBM prediction on all rows
    5. Aggregate results by StockCode
    6. Return JSON with summary + per-product predictions + validation
    """
    # Validate file extension
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Please upload a .csv or .xlsx file."
        )

    try:
        # Read file contents
        contents = await file.read()

        # Parse into DataFrame
        if ext == ".csv":
            for encoding in ("utf-8", "latin-1", "iso-8859-1"):
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise HTTPException(status_code=400, detail="Could not decode CSV file.")
        else:
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")

        # Validate required columns
        required = ["InvoiceNo", "StockCode", "Description", "Quantity",
                     "InvoiceDate", "UnitPrice", "CustomerID"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing}. "
                       f"Expected: {required}"
            )

        # Run data validation
        validation_report = run_validation(df)

        # Check model exists
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run `python src/train.py` first."
            )

        # Run prediction pipeline
        results = compute_net_demand(df)

        # Build response
        products = results.to_dict(orient="records")

        # Ensure JSON-serializable types
        for p in products:
            for key, val in p.items():
                if isinstance(val, (np.integer,)):
                    p[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    p[key] = float(val)
                elif isinstance(val, (np.bool_,)):
                    p[key] = bool(val)

        response = {
            "total_products": len(results),
            "high_risk_count": int((results["risk_level"] == "HIGH").sum()),
            "medium_risk_count": int((results["risk_level"] == "MEDIUM").sum()),
            "low_risk_count": int((results["risk_level"] == "LOW").sum()),
            "total_units_saved": int(results["units_saved"].sum()),
            "average_return_probability": round(float(results["return_probability"].mean()), 4),
            "products": products,
            "validation": validation_report,
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during prediction: {str(e)}"
        )


# ---------------------------------------------------------------------------
# GET /summary — Model metrics
# ---------------------------------------------------------------------------
@app.get("/summary")
async def get_summary():
    """Return model evaluation metrics from the last training run."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No model metrics found. Train the model first."
        )

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    return {
        "status": "success",
        "model": "LightGBM Return Classifier",
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# GET /shap — SHAP feature importance
# ---------------------------------------------------------------------------
@app.get("/shap")
async def get_shap():
    """Return SHAP feature importance values."""
    if not SHAP_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No SHAP data found. Train the model first."
        )

    with open(SHAP_PATH, "r") as f:
        shap_data = json.load(f)

    return {
        "status": "success",
        "feature_importance": shap_data,
    }


# ---------------------------------------------------------------------------
# GET /drift — Drift monitoring
# ---------------------------------------------------------------------------
@app.get("/drift")
async def check_drift():
    """Run Evidently drift check and return results."""
    try:
        result = run_drift_check()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drift check failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# POST /retrain — Retrain pipeline
# ---------------------------------------------------------------------------
@app.post("/retrain")
async def retrain():
    """
    Trigger full retraining pipeline:
    1. Run preprocessing
    2. Train LightGBM
    3. Return new metrics
    """
    try:
        from src.preprocessing import preprocess
        from src.train import train_model

        print("[API] Starting retraining pipeline...")

        # Step 1: Preprocess
        preprocess()

        # Step 2: Train
        metrics = train_model()

        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Retraining failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# POST /what-if — What-if scenario analysis
# ---------------------------------------------------------------------------
@app.post("/what-if")
async def what_if_analysis(product: dict):
    """
    Compute what-if scenarios for a specific product.

    Expects JSON body with: gross_sales, return_probability
    """
    try:
        required = ["gross_sales", "return_probability"]
        missing = [k for k in required if k not in product]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing fields: {missing}"
            )

        scenarios = compute_what_if(product)
        return {"status": "success", "scenarios": scenarios}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"What-if computation failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# GET /forecast — Sales forecast data
# ---------------------------------------------------------------------------
@app.get("/forecast")
async def get_forecast():
    """
    Return Prophet sales forecast data.
    Reads from the saved forecast CSV if available.
    """
    if not FORECAST_CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No forecast data found. Train the Prophet model first "
                   "via POST /forecast/train or run `python src/sales_forecast.py`."
        )

    try:
        forecast_df = pd.read_csv(FORECAST_CSV_PATH)
        forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.strftime("%Y-%m-%d")

        records = forecast_df.to_dict(orient="records")

        # Ensure JSON-serializable
        for r in records:
            for key, val in r.items():
                if isinstance(val, (np.integer,)):
                    r[key] = int(val)
                elif isinstance(val, (np.floating,)):
                    r[key] = float(val)

        # Summary stats for the forecast portion (last 30 days)
        forecast_only = forecast_df.tail(30)
        summary = {
            "total_predicted_revenue": round(float(forecast_only["predicted_revenue"].sum()), 2),
            "avg_daily_revenue": round(float(forecast_only["predicted_revenue"].mean()), 2),
            "forecast_days": len(forecast_only),
            "total_data_points": len(forecast_df),
        }

        return {
            "status": "success",
            "summary": summary,
            "data": records,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load forecast data: {str(e)}"
        )


# ---------------------------------------------------------------------------
# POST /forecast/train — Train Prophet model
# ---------------------------------------------------------------------------
@app.post("/forecast/train")
async def train_forecast():
    """
    Train the Prophet sales forecasting model.
    Uses the raw dataset to generate a 30-day forecast.
    """
    try:
        from src.sales_forecast import run_sales_forecast

        print("[API] Starting Prophet training pipeline...")
        forecast = run_sales_forecast()

        return {
            "status": "success",
            "message": "Prophet model trained and forecast generated",
            "forecast_rows": len(forecast),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Prophet is not installed: {str(e)}. "
                   f"Install with: pip install prophet"
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Forecast training failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# GET /validate — Validate processed data
# ---------------------------------------------------------------------------
@app.get("/validate")
async def validate_data():
    """
    Validate the processed dataset using Great Expectations or pandas fallback.
    """
    processed_path = PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"

    if not processed_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No processed data found. Run preprocessing first."
        )

    try:
        df = pd.read_csv(processed_path)
        result = run_validation(df)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Run with: uvicorn api.app:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
