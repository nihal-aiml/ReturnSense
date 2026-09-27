# 📦 ReturnSense — Return-Aware Net Demand Forecasting System

> **Predict not just what sells — but what stays sold.**

ReturnSense is a production-ready MLOps system that solves a critical e-commerce problem: demand forecasting systems predict gross sales but ignore returns. With an industry average return rate of **17.6%**, if you predict 100 sales but 22 get returned, your real demand is 78 — but you ordered 100.

## Core Formula

```
Net Demand = Predicted Gross Sales × (1 − Return Probability)
```

## Two Models

| Model | Purpose | Algorithm |
|-------|---------|-----------|
| **Return Classifier** | Predicts return probability per transaction | LightGBM |
| **Sales Forecaster** | Forecasts daily gross sales | Prophet |

---

## 🏗️ Architecture

```
User uploads CSV → FastAPI preprocesses & engineers features →
LightGBM predicts return probability → Net Demand computed →
Beautiful dashboard renders results with charts, tables, what-if simulator
```

## 🛠️ Tech Stack

- **ML Models**: LightGBM, Prophet
- **Explainability**: SHAP
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Frontend**: Pure HTML + CSS + JavaScript (no frameworks)
- **Data Validation**: Great Expectations (with pandas fallback)
- **Drift Monitoring**: Evidently AI
- **CI/CD**: GitHub Actions
- **Data Versioning**: DVC
- **Testing**: Pytest
- **Class Imbalance**: SMOTE (imbalanced-learn)

---

## 📁 Folder Structure

```
ReturnSense/
├── data/
│   ├── .gitignore                     # Git-ignore large files (DVC-tracked)
│   └── raw/                           # Place Online Retail.xlsx here
├── src/
│   ├── data_ingestion.py              # Load and clean dataset
│   ├── preprocessing.py               # Feature engineering + return labels
│   ├── feature_engineering.py         # Prepare X and y for training
│   ├── train.py                       # Train LightGBM + log to MLflow
│   ├── sales_forecast.py              # Train Prophet model
│   ├── predict.py                     # Net demand computation
│   ├── monitor.py                     # Evidently AI drift monitoring
│   └── validate.py                    # Great Expectations data validation
├── api/
│   └── app.py                         # FastAPI backend (10 endpoints)
├── frontend/
│   ├── index.html                     # Main dashboard
│   ├── style.css                      # Styling
│   └── app.js                         # JavaScript logic
├── tests/
│   └── test_pipeline.py               # Pytest test suite (13+ tests)
├── scripts/
│   └── dev.js                         # Concurrent dev launcher
├── .github/
│   └── workflows/
│       └── ci_cd.yml                  # GitHub Actions CI/CD
├── server.js                          # Frontend dev server (Node.js)
├── package.json                       # npm scripts
├── start.ps1                          # Windows launcher
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the dev server)
- pip

### Step 1: Install Dependencies

```bash
cd ReturnSense
pip install -r requirements.txt
```

### Step 2: Prepare Data

Download the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) and place `Online Retail.xlsx` in the `data/raw/` directory:

```
ReturnSense/data/raw/Online Retail.xlsx
```

**Or if using DVC:**
```bash
dvc pull   # Restores data from DVC remote
```

### Step 3: Run Preprocessing

```bash
python src/preprocessing.py
```

This will:
- Load and clean the dataset
- Engineer all 7 features automatically
- Create the return label (target) using CustomerID + StockCode matching
- Save processed data to `data/raw/processed_returns.csv`
- Print the return rate (expected: 4–6%)

### Step 4: Train the Model

```bash
python src/train.py
```

This will:
- Load processed data
- Split 80/20 with stratification
- Apply SMOTE on training set
- Train LightGBM (n_estimators=300, lr=0.05, max_depth=6)
- Log all metrics to MLflow
- Save model to `data/return_model.pkl`
- Generate SHAP feature importance
- Print AUC, Accuracy, Precision, Recall, F1

### Step 5: (Optional) Train Sales Forecaster

```bash
python src/sales_forecast.py
```

Trains a Prophet model on daily revenue data and generates a 30-day forecast.

### Step 6: Start the App

**Option A: npm dev server (recommended)**
```bash
npm run dev
```
This starts both the **frontend** (http://localhost:5173) and **FastAPI backend** (http://localhost:8000) concurrently.

**Option B: Start servers separately**
```bash
# Terminal 1: API
uvicorn api.app:app --reload --port 8000

# Terminal 2: Frontend
node server.js
```

**Option C: Windows PowerShell**
```bash
.\start.ps1
```

### Step 7: Open the Dashboard

Open **http://localhost:5173** in your browser.

Then:
1. Drag and drop your CSV/Excel file onto the upload area
2. Click "Analyze Now"
3. View results: data quality report, summary cards, risk chart, SHAP importance, sales forecast, product table, what-if simulator

### Step 8: Run Tests

```bash
npm test
# or
python -m pytest tests/ -v
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload CSV/Excel and get predictions + validation |
| `GET` | `/summary` | Get model evaluation metrics |
| `GET` | `/shap` | Get SHAP feature importance |
| `GET` | `/drift` | Run drift monitoring check |
| `POST` | `/retrain` | Trigger model retraining |
| `POST` | `/what-if` | Compute what-if scenarios |
| `GET` | `/forecast` | Get Prophet sales forecast data |
| `POST` | `/forecast/train` | Train Prophet forecasting model |
| `GET` | `/validate` | Validate processed dataset |

**Interactive API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🔬 Features Engineered

All features are computed **automatically** from the uploaded data:

| # | Feature | Source |
|---|---------|--------|
| 1 | `UnitPrice` | Direct from dataset |
| 2 | `price_vs_category_avg` | UnitPrice / mean(UnitPrice) per Description |
| 3 | `customer_past_return_rate` | Returned items / total items per CustomerID |
| 4 | `customer_order_count` | Unique InvoiceNo count per CustomerID |
| 5 | `day_of_week` | From InvoiceDate (0=Monday) |
| 6 | `month` | From InvoiceDate (1–12) |
| 7 | `Quantity` | Direct from dataset |

---

## ✅ Data Validation

ReturnSense validates uploaded data automatically using **Great Expectations** (or a pandas-based fallback). The validation checks include:

| Check | Description |
|-------|-------------|
| Required Columns | All 7 required columns must exist |
| Non-null CustomerID | CustomerID must not have null values |
| Positive UnitPrice | Prices must be > 0 |
| Numeric Quantity | Quantities must be numeric |
| Parseable Dates | InvoiceDate must be datetime-compatible |
| Price Range | UnitPrice within reasonable bounds |
| Quantity Range | Quantities within reasonable bounds |
| Return Rate | Between 0–30% |
| Minimum Rows | At least 100 rows |
| Customer Diversity | Multiple unique customers |

---

## 📈 Sales Forecasting

The **Prophet** integration provides:
- Daily revenue aggregation from transaction data
- 30-day ahead forecast with confidence intervals
- Historical trend visualization
- Weekly and yearly seasonality detection

Train the model:
```bash
python src/sales_forecast.py
```

The forecast is then available via `GET /forecast` and displayed as a line chart in the dashboard.

---

## 🎯 How the Flow Works

1. User opens `http://localhost:5173` in browser
2. User drags and drops their Online Retail CSV/Excel file
3. Frontend sends file to FastAPI `POST /upload`
4. FastAPI runs **data validation** (10 quality checks)
5. FastAPI runs preprocessing automatically (engineers all features, no manual input)
6. FastAPI runs LightGBM prediction on all rows
7. FastAPI aggregates results by StockCode
8. FastAPI returns JSON with all product predictions + validation report
9. Frontend renders: data quality bar, summary cards, risk chart, SHAP importance, **sales forecast**, products table, what-if simulator
10. User can search, sort, and paginate the products table
11. User selects a product in the what-if simulator to see impact scenarios

---

## 🔄 DVC — Data Version Control

Large data files are tracked with DVC instead of git:

```bash
# Initialize DVC (already done)
dvc init

# Track the raw dataset
dvc add data/raw/Online\ Retail.xlsx

# Push to remote (if configured)
dvc push

# Pull data on a new machine
dvc pull
```

---

## 🌍 SDG Alignment

- **SDG 9**: Industry, Innovation and Infrastructure — AI-powered supply chain optimization
- **SDG 12**: Responsible Consumption and Production — reducing overstock waste through accurate demand forecasting

---

## 📄 License

This project is for educational and research purposes.
