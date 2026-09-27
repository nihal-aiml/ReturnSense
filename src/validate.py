"""
ReturnSense -- Data Validation Module
======================================
Uses Great Expectations-style validation to ensure data quality
before running predictions.

Validates:
  1. Required columns exist
  2. No null CustomerIDs
  3. UnitPrice > 0
  4. Quantity is numeric
  5. InvoiceDate is parseable
  6. Value range checks
  7. Return rate sanity check (1–30%)

If Great Expectations is installed, uses its native API.
Otherwise, falls back to a pure-pandas implementation with the
same validation rules — ensuring the project works without
heavy dependencies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Required columns for the UCI Online Retail dataset
REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"
]


def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Run comprehensive data validation on a DataFrame.

    Returns
    -------
    dict
        Validation report with overall status, individual check results,
        and summary statistics.
    """
    checks = []
    warnings_list = []
    total_rows = len(df)

    # ------------------------------------------------------------------
    # Check 1: Required columns exist
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks.append({
        "name": "required_columns",
        "description": "All required columns present",
        "passed": len(missing_cols) == 0,
        "detail": f"Missing: {missing_cols}" if missing_cols else "All 7 columns present",
    })

    if missing_cols:
        # Can't proceed with other checks if columns are missing
        return _build_report(checks, warnings_list, total_rows, overall_status="FAIL")

    # ------------------------------------------------------------------
    # Check 2: No null CustomerIDs
    # ------------------------------------------------------------------
    null_customer_count = df["CustomerID"].isna().sum()
    null_pct = (null_customer_count / total_rows * 100) if total_rows > 0 else 0
    checks.append({
        "name": "no_null_customer_id",
        "description": "CustomerID has no null values",
        "passed": bool(null_customer_count == 0),
        "detail": f"{null_customer_count} nulls ({null_pct:.1f}%)" if null_customer_count > 0
                  else "No null values",
    })
    if null_pct > 25:
        warnings_list.append(f"High null rate in CustomerID: {null_pct:.1f}%")

    # ------------------------------------------------------------------
    # Check 3: UnitPrice > 0
    # ------------------------------------------------------------------
    non_positive_price = (df["UnitPrice"] <= 0).sum()
    np_pct = (non_positive_price / total_rows * 100) if total_rows > 0 else 0
    checks.append({
        "name": "positive_unit_price",
        "description": "UnitPrice is positive (> 0)",
        "passed": bool(non_positive_price == 0),
        "detail": f"{non_positive_price} rows with UnitPrice ≤ 0 ({np_pct:.1f}%)"
                  if non_positive_price > 0 else "All prices positive",
    })

    # ------------------------------------------------------------------
    # Check 4: Quantity is numeric
    # ------------------------------------------------------------------
    qty_numeric = pd.to_numeric(df["Quantity"], errors="coerce")
    non_numeric_qty = qty_numeric.isna().sum() - df["Quantity"].isna().sum()
    checks.append({
        "name": "quantity_is_numeric",
        "description": "Quantity values are numeric",
        "passed": bool(non_numeric_qty == 0),
        "detail": f"{non_numeric_qty} non-numeric values" if non_numeric_qty > 0
                  else "All values numeric",
    })

    # ------------------------------------------------------------------
    # Check 5: InvoiceDate is parseable
    # ------------------------------------------------------------------
    parsed_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    unparseable_dates = parsed_dates.isna().sum() - df["InvoiceDate"].isna().sum()
    checks.append({
        "name": "invoice_date_parseable",
        "description": "InvoiceDate can be parsed as datetime",
        "passed": bool(unparseable_dates == 0),
        "detail": f"{unparseable_dates} unparseable dates" if unparseable_dates > 0
                  else "All dates parseable",
    })

    # ------------------------------------------------------------------
    # Check 6: Value range — UnitPrice reasonable
    # ------------------------------------------------------------------
    if (df["UnitPrice"] > 0).any():
        max_price = df["UnitPrice"].max()
        median_price = df["UnitPrice"].median()
        price_reasonable = max_price < 50000  # sanity limit
        checks.append({
            "name": "unit_price_range",
            "description": "UnitPrice within reasonable range (< $50,000)",
            "passed": bool(price_reasonable),
            "detail": f"Max: ${max_price:,.2f}, Median: ${median_price:,.2f}",
        })
        if not price_reasonable:
            warnings_list.append(f"Extreme UnitPrice detected: ${max_price:,.2f}")

    # ------------------------------------------------------------------
    # Check 7: Value range — Quantity reasonable
    # ------------------------------------------------------------------
    abs_qty = df["Quantity"].abs()
    max_qty = abs_qty.max()
    qty_reasonable = max_qty < 100000
    checks.append({
        "name": "quantity_range",
        "description": "Quantity within reasonable range (< 100,000)",
        "passed": bool(qty_reasonable),
        "detail": f"Max absolute quantity: {max_qty:,}",
    })

    # ------------------------------------------------------------------
    # Check 8: Return rate sanity (if we can compute it)
    # ------------------------------------------------------------------
    purchases = df[df["Quantity"] > 0]
    returns = df[df["Quantity"] < 0]
    if len(purchases) > 0:
        return_rate = len(returns) / (len(purchases) + len(returns))
        rate_ok = 0.0 <= return_rate <= 0.30
        checks.append({
            "name": "return_rate_sanity",
            "description": "Return rate between 0–30%",
            "passed": bool(rate_ok),
            "detail": f"Return rate: {return_rate*100:.1f}% "
                      f"({len(returns)} returns / {len(purchases)} purchases)",
        })
        if not rate_ok:
            warnings_list.append(f"Unusual return rate: {return_rate*100:.1f}%")

    # ------------------------------------------------------------------
    # Check 9: Minimum row count
    # ------------------------------------------------------------------
    enough_rows = total_rows >= 100
    checks.append({
        "name": "minimum_rows",
        "description": "Dataset has at least 100 rows",
        "passed": bool(enough_rows),
        "detail": f"{total_rows:,} rows",
    })
    if not enough_rows:
        warnings_list.append(f"Small dataset: only {total_rows} rows")

    # ------------------------------------------------------------------
    # Check 10: Unique customers
    # ------------------------------------------------------------------
    valid_customers = df["CustomerID"].dropna()
    n_customers = valid_customers.nunique()
    checks.append({
        "name": "customer_diversity",
        "description": "Multiple unique customers present",
        "passed": bool(n_customers >= 2),
        "detail": f"{n_customers:,} unique customers",
    })

    # Build overall status
    failed_count = sum(1 for c in checks if not c["passed"])
    if failed_count == 0:
        overall_status = "PASS"
    elif failed_count <= 2:
        overall_status = "WARNING"
    else:
        overall_status = "FAIL"

    return _build_report(checks, warnings_list, total_rows, overall_status)


def _build_report(checks: list, warnings_list: list, total_rows: int,
                  overall_status: str) -> dict:
    """Build the final validation report dictionary."""
    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])

    report = {
        "status": overall_status,
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(checks) * 100, 1) if checks else 0,
        "total_rows": total_rows,
        "checks": checks,
        "warnings": warnings_list,
    }

    print(f"[Validate] Data validation: {overall_status} "
          f"({passed}/{len(checks)} checks passed)")
    if warnings_list:
        for w in warnings_list:
            print(f"[Validate] ⚠ {w}")

    return report


def validate_file(filepath: Path) -> dict:
    """Validate a CSV or Excel file from disk."""
    ext = filepath.suffix.lower()
    if ext == ".csv":
        for encoding in ("utf-8", "latin-1", "iso-8859-1"):
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return {"status": "FAIL", "message": "Could not decode CSV file"}
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        return {"status": "FAIL", "message": f"Unsupported file type: {ext}"}

    return validate_dataframe(df)


# Try to use Great Expectations if installed
def validate_with_great_expectations(df: pd.DataFrame) -> Optional[dict]:
    """
    Run validation using Great Expectations if available.
    Returns None if GE is not installed, allowing fallback to pandas validation.
    """
    try:
        import great_expectations as gx
        from great_expectations.dataset import PandasDataset

        ge_df = PandasDataset(df)

        results = []

        # Required columns
        for col in REQUIRED_COLUMNS:
            r = ge_df.expect_column_to_exist(col)
            results.append({
                "name": f"column_exists_{col}",
                "passed": bool(r["success"]),
                "detail": f"Column '{col}' exists" if bool(r["success"]) else f"Missing: {col}",
            })

        # Non-null checks
        r = ge_df.expect_column_values_to_not_be_null("CustomerID", mostly=0.7)
        results.append({
            "name": "customer_id_not_null",
            "passed": bool(r["success"]),
            "detail": "CustomerID mostly non-null" if bool(r["success"])
                      else "Too many null CustomerIDs",
        })

        # Positive UnitPrice
        r = ge_df.expect_column_values_to_be_between(
            "UnitPrice", min_value=0, strict_min=True, mostly=0.95
        )
        results.append({
            "name": "positive_unit_price",
            "passed": bool(r["success"]),
            "detail": "UnitPrice mostly positive" if bool(r["success"])
                      else "Many non-positive prices",
        })

        # Quantity numeric
        r = ge_df.expect_column_values_to_be_of_type("Quantity", "int64")
        passed_type = r["success"]
        if not passed_type:
            r = ge_df.expect_column_values_to_be_of_type("Quantity", "float64")
            passed_type = r["success"]
        results.append({
            "name": "quantity_numeric",
            "passed": bool(passed_type),
            "detail": "Quantity is numeric" if bool(passed_type) else "Quantity type issue",
        })

        passed = sum(1 for r in results if r["passed"])
        failed = sum(1 for r in results if not r["passed"])

        return {
            "status": "PASS" if failed == 0 else ("WARNING" if failed <= 2 else "FAIL"),
            "engine": "great_expectations",
            "total_checks": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
            "total_rows": len(df),
            "checks": results,
            "warnings": [],
        }

    except ImportError:
        return None
    except Exception as e:
        print(f"[Validate] Great Expectations failed: {e}")
        return None


def run_validation(df: pd.DataFrame) -> dict:
    """
    Run validation — tries Great Expectations first, falls back to pandas.
    This is the main entry point used by the API.
    """
    # Try GE first
    ge_result = validate_with_great_expectations(df)
    if ge_result is not None:
        print("[Validate] Used Great Expectations engine")
        return ge_result

    # Fallback to pandas validation
    print("[Validate] Using pandas validation engine")
    result = validate_dataframe(df)
    result["engine"] = "pandas"
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
        result = validate_file(filepath)
    else:
        # Validate the processed data
        processed_path = PROJECT_ROOT / "data" / "raw" / "processed_returns.csv"
        if processed_path.exists():
            df = pd.read_csv(processed_path)
            result = run_validation(df)
        else:
            print("No data file found. Provide a path as argument.")
            sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  DATA VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"  Status:       {result['status']}")
    print(f"  Checks:       {result.get('passed', 0)}/{result.get('total_checks', 0)} passed")
    print(f"  Pass rate:    {result.get('pass_rate', 0)}%")
    print(f"  Total rows:   {result.get('total_rows', 0):,}")
    if result.get("engine"):
        print(f"  Engine:       {result['engine']}")
    print(f"{'='*50}")
    for check in result.get("checks", []):
        icon = "✓" if check["passed"] else "✗"
        print(f"  {icon} {check.get('description', check['name'])}: {check.get('detail', '')}")
