"""
ReturnSense -- Data Ingestion Module
====================================
Loads the UCI Online Retail dataset (Online Retail.xlsx) from the data/raw/ directory.
Handles both .xlsx and .csv formats. Performs initial cleaning:
  - Drops rows with missing CustomerID
  - Converts InvoiceDate to datetime
  - Casts CustomerID to int
  - Removes rows with zero or missing Quantity/UnitPrice

This module is the FIRST step in the pipeline. It outputs a clean DataFrame
ready for preprocessing.py to engineer features and labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


def find_dataset(directory: Path = RAW_DATA_DIR) -> Path:
    """
    Scans the raw data directory for the first file matching a supported
    extension. Prefers 'Online Retail.xlsx' if it exists.
    """
    canonical = directory / "Online Retail.xlsx"
    if canonical.exists():
        return canonical

    for f in sorted(directory.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            return f

    raise FileNotFoundError(
        f"No dataset found in {directory}. "
        f"Please place 'Online Retail.xlsx' (UCI Online Retail dataset) in {directory}"
    )


def load_data(filepath=None) -> pd.DataFrame:
    """
    Load the raw dataset into a pandas DataFrame.
    """
    if filepath is None:
        filepath = find_dataset()
    else:
        filepath = Path(filepath)

    print(f"[DataIngestion] Loading dataset from: {filepath}")

    ext = filepath.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    elif ext == ".csv":
        for encoding in ("utf-8", "latin-1", "iso-8859-1"):
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode CSV file: {filepath}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"[DataIngestion] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning.
    """
    initial_rows = len(df)

    # Drop missing CustomerID -- critical for return matching
    df = df.dropna(subset=["CustomerID"]).copy()
    print(f"[DataIngestion] Dropped {initial_rows - len(df)} rows with missing CustomerID")

    # Convert InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Cast CustomerID to int
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove zero-quantity and non-positive price rows
    df = df[df["Quantity"] != 0]
    df = df[df["UnitPrice"] > 0]

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    print(f"[DataIngestion] Cleaned shape: {df.shape}")
    return df.reset_index(drop=True)


def ingest(filepath=None) -> pd.DataFrame:
    """
    Full ingestion pipeline: load -> clean -> return DataFrame.
    This is the main entry point for other modules.
    """
    df = load_data(filepath)
    df = clean_data(df)
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    df = ingest(filepath)
    print(f"\n[DataIngestion] Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[DataIngestion] Columns: {list(df.columns)}")
    print(f"[DataIngestion] Date range: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    print(f"[DataIngestion] Unique customers: {df['CustomerID'].nunique()}")
    print(f"[DataIngestion] Unique products: {df['StockCode'].nunique()}")
