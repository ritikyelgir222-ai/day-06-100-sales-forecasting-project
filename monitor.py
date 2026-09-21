"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index) specifically: PSI is the standard,
interview-recognizable metric for feature drift in industry because it
gives a single interpretable number per feature with well-established
severity thresholds (< 0.1 stable, 0.1-0.25 moderate shift, > 0.25
significant shift needing action).

WHY THIS SCRIPT'S "NEW BATCH" IS GENUINELY DIFFERENT FROM THE OTHER
PROJECTS IN THIS SERIES: the churn and house-price/attrition projects'
monitor.py scripts compare train against a same-snapshot held-out test
set as a stand-in for "new data," since those datasets have no real time
axis. This project's test set is the actual most recent 12 weeks of the
dataset (see split.py) — so any drift detected here (e.g. in Fuel_Price
or CPI, which genuinely move over a 2.7-year window) reflects real
macroeconomic change over time, not just an artifact of random sampling.
"""

import numpy as np
import pandas as pd
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import weighted_mae, pct_within_tolerance

WMAE_INCREASE_THRESHOLD = 0.15  # per SDLC doc Phase 13: retrain if test WMAE rises more than 15% vs. baseline
DRIFT_FEATURES = ["Fuel_Price", "CPI", "Unemployment", "Temperature", "Lag_1"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/sales_forecast_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    train_df, val_df, test_df = split_data(engineered)

    print("=== Feature Drift (PSI): train period vs. most-recent-12-weeks batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(train_df[feat], test_df[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # UNLIKE THE OTHER PROJECTS' MONITOR.PY: because train and test here
    # are genuinely different time periods (~2.5 years apart at the
    # midpoint), don't be surprised if Fuel_Price or CPI show real drift
    # — that's the monitoring script doing its job, not a bug.

    X_test = test_df[feature_cols]
    y_test = test_df["Weekly_Sales"]
    preds = model.predict(X_test)
    wmae = weighted_mae(y_test, preds, X_test["IsHoliday"])
    pct_within = pct_within_tolerance(y_test, preds)

    print(f"\n=== Performance on held-out batch ===")
    print(f"WMAE: {wmae:.2f}")
    print(f"% within +/-15%: {pct_within:.4f}")

    import json
    with open("outputs/business_validation.json") as f:
        baseline_wmae = json.load(f)["test_wmae"]

    pct_increase = (wmae - baseline_wmae) / baseline_wmae
    if pct_increase > WMAE_INCREASE_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: WMAE increased {pct_increase:.1%} vs. baseline (threshold: {WMAE_INCREASE_THRESHOLD:.0%})")
    else:
        print(f"\n✅ No retrain needed (WMAE change: {pct_increase:+.1%}, threshold: {WMAE_INCREASE_THRESHOLD:.0%})")


if __name__ == "__main__":
    run_monitoring_check()
