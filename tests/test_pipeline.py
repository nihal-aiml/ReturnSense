"""
ReturnSense — Automated Test Suite
====================================
Comprehensive pytest tests validating the entire ML pipeline.

Tests cover:
  1. Model file existence after training
  2. Prediction output range (0–1)
  3. Net demand <= gross sales invariant
  4. Feature engineering column correctness
  5. Return rate sanity check (1–30%)
  6. FastAPI /upload returns 200
  7. API root health check
  8. What-if computation
  9. Risk level classification
  10. Data validation module
  11. API /validate endpoint
  12. API /forecast endpoint
  13. API /shap endpoint
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_raw_data():
    """Create a minimal synthetic dataset mimicking UCI Online Retail."""
    np.random.seed(42)
    n = 500

    # Generate purchases
    data = {
        "InvoiceNo": [f"INV{i:04d}" for i in range(n)],
        "StockCode": np.random.choice(["85123A", "71053", "84406B", "22752", "21730"], n),
        "Description": np.random.choice(
            [
                "WHITE HANGING HEART T-LIGHT HOLDER",
                "WHITE METAL LANTERN",
                "CREAM CUPID HEARTS COAT HANGER",
                "SET 7 BABUSHKA NESTING BOXES",
                "GLASS STAR FROSTED T-LIGHT HOLDER",
            ],
            n,
        ),
        "Quantity": np.random.randint(1, 20, n),
        "InvoiceDate": pd.date_range("2011-01-01", periods=n, freq="H"),
        "UnitPrice": np.round(np.random.uniform(0.5, 50.0, n), 2),
        "CustomerID": np.random.randint(12000, 12050, n),
        "Country": "United Kingdom",
    }
    df = pd.DataFrame(data)

    # Add some return rows (Quantity < 0) — about 5% of rows
    n_returns = int(n * 0.05)
    return_rows = df.sample(n_returns, random_state=42).copy()
    return_rows["Quantity"] = -return_rows["Quantity"]
    return_rows["InvoiceNo"] = [f"C{inv}" for inv in return_rows["InvoiceNo"]]

    df = pd.concat([df, return_rows], ignore_index=True)
    return df


@pytest.fixture
def processed_data_path():
    """Path to the processed returns CSV."""
    return PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"


@pytest.fixture
def model_path():
    """Path to the trained model."""
    return PROJECT_ROOT / "data" / "return_model.pkl"


@pytest.fixture
def metrics_path():
    """Path to saved model metrics."""
    return PROJECT_ROOT / "data" / "model_metrics.json"


# ---------------------------------------------------------------------------
# Test 1: Model file exists after training
# ---------------------------------------------------------------------------
class TestModelExists:
    def test_model_file_exists(self, model_path):
        """After training, data/return_model.pkl must exist."""
        if not model_path.exists():
            pytest.skip("Model not yet trained — run `python src/train.py` first.")
        assert model_path.exists(), f"Model file not found at {model_path}"
        assert model_path.stat().st_size > 0, "Model file is empty"

    def test_metrics_file_exists(self, metrics_path):
        """After training, data/model_metrics.json must exist."""
        if not metrics_path.exists():
            pytest.skip("Metrics not found — run `python src/train.py` first.")
        assert metrics_path.exists()
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "auc" in metrics
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Test 2: Prediction output is between 0 and 1
# ---------------------------------------------------------------------------
class TestPredictionRange:
    def test_return_probability_range(self, sample_raw_data, model_path):
        """All predicted return probabilities must be in [0, 1]."""
        if not model_path.exists():
            pytest.skip("Model not trained yet.")

        from src.predict import compute_net_demand

        results = compute_net_demand(sample_raw_data)
        assert (results["return_probability"] >= 0).all(), \
            "Some return probabilities are negative"
        assert (results["return_probability"] <= 1).all(), \
            "Some return probabilities exceed 1.0"


# ---------------------------------------------------------------------------
# Test 3: Net demand <= gross sales
# ---------------------------------------------------------------------------
class TestNetDemandInvariant:
    def test_net_demand_le_gross_sales(self, sample_raw_data, model_path):
        """Net demand must always be <= gross sales (can't sell more than predicted)."""
        if not model_path.exists():
            pytest.skip("Model not trained yet.")

        from src.predict import compute_net_demand

        results = compute_net_demand(sample_raw_data)
        assert (results["net_demand"] <= results["gross_sales"]).all(), \
            "Net demand exceeds gross sales for some products"


# ---------------------------------------------------------------------------
# Test 4: Feature engineering produces correct columns
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_feature_columns_present(self, sample_raw_data):
        """Feature engineering must produce all 7 required columns."""
        from src.feature_engineering import engineer_features_from_raw

        featured = engineer_features_from_raw(sample_raw_data)

        for col in FEATURE_COLUMNS:
            assert col in featured.columns, \
                f"Missing feature column: {col}"

    def test_no_nan_in_features(self, sample_raw_data):
        """Engineered features must not contain NaN values."""
        from src.feature_engineering import engineer_features_from_raw

        featured = engineer_features_from_raw(sample_raw_data)

        for col in FEATURE_COLUMNS:
            nan_count = featured[col].isna().sum()
            assert nan_count == 0, \
                f"Feature {col} has {nan_count} NaN values"

    def test_day_of_week_range(self, sample_raw_data):
        """day_of_week must be 0–6."""
        from src.feature_engineering import engineer_features_from_raw

        featured = engineer_features_from_raw(sample_raw_data)
        assert featured["day_of_week"].min() >= 0
        assert featured["day_of_week"].max() <= 6

    def test_month_range(self, sample_raw_data):
        """month must be 1–12."""
        from src.feature_engineering import engineer_features_from_raw

        featured = engineer_features_from_raw(sample_raw_data)
        assert featured["month"].min() >= 1
        assert featured["month"].max() <= 12


# ---------------------------------------------------------------------------
# Test 5: Return rate sanity check
# ---------------------------------------------------------------------------
class TestReturnRate:
    def test_return_rate_in_range(self, processed_data_path):
        """Return rate in processed CSV must be between 1% and 30%."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not found — run preprocessing first.")

        df = pd.read_csv(processed_data_path)
        return_rate = df["target"].mean()

        assert 0.01 <= return_rate <= 0.30, \
            f"Return rate {return_rate:.4f} is outside expected range [0.01, 0.30]"

    def test_target_is_binary(self, processed_data_path):
        """Target column must only contain 0 and 1."""
        if not processed_data_path.exists():
            pytest.skip("Processed data not found.")

        df = pd.read_csv(processed_data_path)
        unique_vals = set(df["target"].unique())
        assert unique_vals.issubset({0, 1}), \
            f"Target contains non-binary values: {unique_vals}"


# ---------------------------------------------------------------------------
# Test 6: FastAPI /upload returns 200
# ---------------------------------------------------------------------------
class TestAPIUpload:
    def test_upload_valid_csv(self, sample_raw_data, model_path, tmp_path):
        """POST /upload with a valid CSV should return 200."""
        if not model_path.exists():
            pytest.skip("Model not trained yet.")

        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)

        # Save sample data as CSV
        csv_path = tmp_path / "test_upload.csv"
        sample_raw_data.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test_upload.csv", f, "text/csv")},
            )

        assert response.status_code == 200, \
            f"Upload failed with status {response.status_code}: {response.text}"

        data = response.json()
        assert "total_products" in data
        assert "products" in data
        assert data["total_products"] > 0
        # Check validation is included
        assert "validation" in data

    def test_upload_invalid_format(self):
        """POST /upload with a .txt file should return 400."""
        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)

        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"some text", "text/plain")},
        )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test 7: API root health check
# ---------------------------------------------------------------------------
class TestAPIRoot:
    def test_root_endpoint(self):
        """GET / should return the health check message."""
        from fastapi.testclient import TestClient
        from api.app import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ReturnSense API running"


# ---------------------------------------------------------------------------
# Test 8: What-if computation
# ---------------------------------------------------------------------------
class TestWhatIf:
    def test_what_if_scenarios(self):
        """What-if scenarios must return correct number of results."""
        from src.predict import compute_what_if

        product = {"gross_sales": 100, "return_probability": 0.20}
        scenarios = compute_what_if(product)

        # Should have current + 3 scenarios (10%, 20%, 30%)
        assert len(scenarios) == 4
        assert scenarios[0]["scenario"] == "Current"

        # Net demand must decrease as return rate drops
        # (i.e., net demand should increase when return probability drops)
        for s in scenarios[1:]:
            assert s["net_demand"] >= scenarios[0]["net_demand"]


# ---------------------------------------------------------------------------
# Test 9: Risk level classification
# ---------------------------------------------------------------------------
class TestRiskLevel:
    def test_risk_classification(self):
        """Risk levels must follow the defined thresholds."""
        from src.predict import classify_risk

        assert classify_risk(0.35) == "HIGH"
        assert classify_risk(0.31) == "HIGH"
        assert classify_risk(0.20) == "MEDIUM"
        assert classify_risk(0.16) == "MEDIUM"
        assert classify_risk(0.15) == "LOW"
        assert classify_risk(0.05) == "LOW"
        assert classify_risk(0.0) == "LOW"


# ---------------------------------------------------------------------------
# Test 10: Data validation module
# ---------------------------------------------------------------------------
class TestValidation:
    def test_validation_passes_good_data(self, sample_raw_data):
        """Validation should pass on well-formed synthetic data."""
        from src.validate import validate_dataframe

        result = validate_dataframe(sample_raw_data)
        assert result["status"] in ("PASS", "WARNING")
        assert result["total_checks"] > 0
        assert result["passed"] > 0

    def test_validation_catches_missing_columns(self):
        """Validation should fail when required columns are missing."""
        from src.validate import validate_dataframe

        df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        result = validate_dataframe(df)
        assert result["status"] == "FAIL"

    def test_validation_report_structure(self, sample_raw_data):
        """Validation report must have the correct structure."""
        from src.validate import validate_dataframe

        result = validate_dataframe(sample_raw_data)
        assert "status" in result
        assert "total_checks" in result
        assert "passed" in result
        assert "failed" in result
        assert "checks" in result
        assert isinstance(result["checks"], list)

    def test_validation_checks_have_required_fields(self, sample_raw_data):
        """Each individual check must have name, passed, and detail."""
        from src.validate import validate_dataframe

        result = validate_dataframe(sample_raw_data)
        for check in result["checks"]:
            assert "name" in check
            assert "passed" in check
            assert isinstance(check["passed"], bool)


# ---------------------------------------------------------------------------
# Test 11: API /validate endpoint
# ---------------------------------------------------------------------------
class TestAPIValidate:
    def test_validate_endpoint(self):
        """GET /validate should return validation results if processed data exists."""
        from fastapi.testclient import TestClient
        from api.app import app

        processed_path = PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"
        if not processed_path.exists():
            pytest.skip("Processed data not found.")

        client = TestClient(app)
        response = client.get("/validate")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "total_checks" in data


# ---------------------------------------------------------------------------
# Test 12: API /forecast endpoint
# ---------------------------------------------------------------------------
class TestAPIForecast:
    def test_forecast_endpoint_structure(self):
        """GET /forecast should return proper structure if forecast data exists."""
        from fastapi.testclient import TestClient
        from api.app import app

        forecast_path = PROJECT_ROOT / "data" / "sales_forecast.csv"
        if not forecast_path.exists():
            pytest.skip("Forecast data not found — run `python src/sales_forecast.py` first.")

        client = TestClient(app)
        response = client.get("/forecast")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "summary" in data
        assert "data" in data
        assert len(data["data"]) > 0


# ---------------------------------------------------------------------------
# Test 13: API /shap endpoint
# ---------------------------------------------------------------------------
class TestAPISHAP:
    def test_shap_endpoint(self):
        """GET /shap should return SHAP feature importance."""
        from fastapi.testclient import TestClient
        from api.app import app

        shap_path = PROJECT_ROOT / "data" / "shap_importance.json"
        if not shap_path.exists():
            pytest.skip("SHAP data not found.")

        client = TestClient(app)
        response = client.get("/shap")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "feature_importance" in data
