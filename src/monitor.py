"""
ReturnSense -- Drift Monitoring Module
=======================================
Uses Evidently AI (or SciPy fallback) to detect data drift.
Automatically dispatches a GitHub Action to retrain the model when drift occurs.
"""

import os
import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    from src.feature_engineering import (
        load_processed_data,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
except ImportError:
    from feature_engineering import (
        load_processed_data,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIFT_REPORT_PATH = PROJECT_ROOT / "data" / "drift_report.json"

# Configuration for automatic retraining trigger
GITHUB_REPO = os.getenv("GITHUB_REPOSITORY", "your-username/ReturnSense")
GITHUB_PAT = os.getenv("GITHUB_PAT")


def trigger_github_retrain(repo_slug: str = GITHUB_REPO, token: str = GITHUB_PAT) -> bool:
    """Send a repository_dispatch event to trigger GitHub Actions retraining."""
    if not token:
        print("[Trigger] Skipping auto-retrain trigger: GITHUB_PAT environment variable not set.")
        return False

    url = f"https://api.github.com/repos/{repo_slug}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "event_type": "data-drift-detected",
        "client_payload": {"source": "drift_monitoring_script"}
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 204:
            print(f"[Trigger] Successfully dispatched retraining workflow to {repo_slug}.")
            return True
        else:
            print(f"[Trigger] Failed to trigger workflow: {response.status_code} - {response.text}")
            return False
    except requests.RequestException as e:
        print(f"[Trigger] Network error while contacting GitHub: {e}")
        return False


def run_drift_check(current_data: pd.DataFrame = None, auto_trigger: bool = True) -> dict:
    """
    Run an Evidently AI drift check comparing reference vs. current data.
    If no current_data is provided, splits the training data 50/50 for demonstration.
    """
    try:
        ref_df = load_processed_data()
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "No reference data found. Run preprocessing first.",
            "drift_detected": False,
        }

    feature_cols = [c for c in FEATURE_COLUMNS if c in ref_df.columns]

    if current_data is not None and not current_data.empty:
        curr_cols = [c for c in feature_cols if c in current_data.columns]
        reference = ref_df[feature_cols].copy()
        current = current_data[curr_cols].copy()
    else:
        midpoint = len(ref_df) // 2
        reference = ref_df.iloc[:midpoint][feature_cols].copy()
        current = ref_df.iloc[midpoint:][feature_cols].copy()

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)

        report_dict = report.as_dict()
        metrics = report_dict.get("metrics", [])

        drift_detected = False
        feature_drift = {}

        for metric in metrics:
            result = metric.get("result", {})
            if "drift_by_columns" in result:
                for col_name, col_data in result["drift_by_columns"].items():
                    is_drifted = col_data.get("drift_detected", False)
                    feature_drift[col_name] = {
                        "drift_detected": is_drifted,
                        "drift_score": round(col_data.get("drift_score", 0), 4),
                        "stattest_name": col_data.get("stattest_name", "unknown"),
                    }
                    if is_drifted:
                        drift_detected = True
            if "dataset_drift" in result:
                drift_detected = result["dataset_drift"]

        drift_result = {
            "status": "success",
            "drift_detected": bool(drift_detected),
            "n_features_analyzed": len(feature_drift),
            "n_features_drifted": sum(
                1 for v in feature_drift.values() if v["drift_detected"]
            ),
            "feature_drift": feature_drift,
            "reference_size": len(reference),
            "current_size": len(current),
            "method": "evidently_ai",
        }

    except ImportError:
        from scipy import stats

        feature_drift = {}
        drift_detected = False

        for col in feature_cols:
            if col in reference.columns and col in current.columns:
                stat, p_value = stats.ks_2samp(
                    reference[col].dropna(),
                    current[col].dropna()
                )
                is_drifted = bool(p_value < 0.05)
                feature_drift[col] = {
                    "drift_detected": is_drifted,
                    "drift_score": float(round(p_value, 4)),
                    "stattest_name": "ks_2samp",
                }
                if is_drifted:
                    drift_detected = True

        drift_result = {
            "status": "success",
            "drift_detected": bool(drift_detected),
            "n_features_analyzed": len(feature_drift),
            "n_features_drifted": sum(
                1 for v in feature_drift.values() if v["drift_detected"]
            ),
            "feature_drift": feature_drift,
            "reference_size": len(reference),
            "current_size": len(current),
            "method": "scipy_ks_test_fallback",
        }
    except Exception as e:
        drift_result = {
            "status": "error",
            "message": str(e),
            "drift_detected": False,
        }

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_REPORT_PATH = DRIFT_REPORT_PATH.with_suffix(".html")

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(drift_result, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"[Monitor] JSON report saved to: {DRIFT_REPORT_PATH}")

    if drift_result.get("status") == "success":
        rows_html = "".join([
            f"""<tr>
                <td><strong>{feat}</strong></td>
                <td><span class="status-badge {'drifted' if info['drift_detected'] else 'ok'}">{'DRIFTED' if info['drift_detected'] else 'OK'}</span></td>
                <td><code>{info['drift_score']}</code></td>
                <td>{info['stattest_name']}</td>
            </tr>"""
            for feat, info in drift_result["feature_drift"].items()
        ])

        alert_class = "alert-danger" if drift_result["drift_detected"] else "alert-success"
        alert_msg = "⚠️ ALERT: Dataset drift detected! Triggering automatic retrain..." if drift_result["drift_detected"] else "✅ Dataset stable. No drift detected."

        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>ReturnSense - Drift Monitoring Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 40px; }}
                .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
                h1 {{ color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 15px; margin-top: 0; }}
                .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; background: #f1f5f9; padding: 15px; border-radius: 6px; }}
                .meta-item {{ font-size: 14px; }}
                .meta-label {{ color: #64748b; font-weight: 600; display: block; }}
                .meta-value {{ font-size: 16px; font-weight: bold; color: #0f172a; }}
                .alert {{ padding: 15px; border-radius: 6px; font-weight: bold; margin-bottom: 25px; border: 1px solid transparent; }}
                .alert-success {{ background-color: #d1e7dd; color: #0f5132; border-color: #badbcc; }}
                .alert-danger {{ background-color: #f8d7da; color: #842029; border-color: #f5c2c7; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
                th {{ background-color: #f8f9fa; color: #475569; font-weight: 600; }}
                tr:hover {{ background-color: #f8fafc; }}
                .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; display: inline-block; }}
                .status-badge.ok {{ background-color: #d1e7dd; color: #0f5132; }}
                .status-badge.drifted {{ background-color: #f8d7da; color: #842029; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 ReturnSense Drift Monitoring Report</h1>
                <div class="alert {alert_class}">{alert_msg}</div>
                <div class="meta-grid">
                    <div class="meta-item"><span class="meta-label">Method</span><span class="meta-value">{drift_result.get('method', 'Evidently AI')}</span></div>
                    <div class="meta-item"><span class="meta-label">Features Checked</span><span class="meta-value">{drift_result['n_features_analyzed']}</span></div>
                    <div class="meta-item"><span class="meta-label">Features Drifted</span><span class="meta-value">{drift_result['n_features_drifted']}</span></div>
                    <div class="meta-item"><span class="meta-label">Reference / Current Size</span><span class="meta-value">{drift_result['reference_size']} / {drift_result['current_size']}</span></div>
                </div>
                <h3>Feature-Level Analysis</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Status</th>
                            <th>Drift Score (p-value)</th>
                            <th>Statistical Test</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[Monitor] HTML report saved to: {HTML_REPORT_PATH}")

    # Fire retraining webhook if drift is confirmed
    if auto_trigger and drift_result.get("drift_detected", False):
        print("\n[Monitor] Drift threshold breached. Dispatching retraining event...")
        trigger_github_retrain()

    return drift_result


if __name__ == "__main__":
    result = run_drift_check()
    print(f"\n{'='*50}")
    print(f"  DRIFT MONITORING REPORT")
    print(f"{'='*50}")
    print(f"  Status:          {result['status']}")
    print(f"  Drift detected:  {result['drift_detected']}")
    if "n_features_analyzed" in result:
        print(f"  Features analyzed: {result['n_features_analyzed']}")
        print(f"  Features drifted:  {result['n_features_drifted']}")
    print(f"{'='*50}")
    if "feature_drift" in result:
        for feat, info in result["feature_drift"].items():
            status = "[!] DRIFTED" if info["drift_detected"] else "[OK]"
            print(f"  {feat}: {status} (score: {info['drift_score']})")