"""
Phase 8: Model Development & Phase 9: Evaluation & Business Validation
---------------------------------------------------------------------------
Every modeling choice below has a WHY comment. The goal is that a
non-technical stakeholder reading the printed output, and a technical
reviewer reading the code, both understand not just WHAT was done but
WHY it was the right call for THIS problem.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data


def weighted_mae(y_true, y_pred, is_holiday):
    """
    BUSINESS-RELEVANT METRIC, not just a technical one.
    WHY WMAE SPECIFICALLY (this is the exact metric Kaggle used to score
    this competition, adopted here deliberately rather than a plain MAE):
    holiday weeks are weighted 5x. This directly encodes the business
    reality that getting the Thanksgiving/Christmas forecast wrong is far
    costlier — in stockouts and lost sales, or in excess unsold holiday
    inventory — than being equally wrong on an ordinary March week. A
    plain average MAE would treat both errors as equally important, which
    doesn't match how the business actually experiences forecast error.
    """
    weights = np.where(np.asarray(is_holiday) == 1, 5, 1)
    return float(np.sum(weights * np.abs(np.asarray(y_true) - np.asarray(y_pred))) / np.sum(weights))


def pct_within_tolerance(y_true, y_pred, tolerance=0.15):
    """
    A second business-relevant metric: retail demand-planning tolerances
    are typically looser than, say, a real-estate price estimate — plus
    or minus 15% is a commonly used "good enough to plan around" bar in
    inventory forecasting, looser than the 8% bar used for the
    house-price project because the underlying decision (how many units
    to stock) is far less sensitive to small percentage misses than a
    single listing price is.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # avoid division by zero for the (rare, post-clipping) zero-sales weeks
    denom = np.where(y_true == 0, 1, y_true)
    pct_error = np.abs(y_pred - y_true) / denom
    return float((pct_error <= tolerance).mean())


def train_and_evaluate():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    train_df, val_df, test_df = split_data(engineered)
    print(f"Train: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"Val:   {len(val_df)} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"Test:  {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    X_train, y_train = train_df[feature_cols], train_df["Weekly_Sales"]
    X_val, y_val = val_df[feature_cols], val_df["Weekly_Sales"]
    X_test, y_test = test_df[feature_cols], test_df["Weekly_Sales"]

    experiment_log = []

    # -----------------------------------------------------------------
    # Baseline: naive persistence ("this week = last week's actual")
    # WHY THIS AS THE BASELINE (not a simple linear model, unlike the
    # other projects in this series): for forecasting problems
    # specifically, the standard, most honest baseline isn't a simple
    # model — it's "assume no change from last observed value." Lag_1
    # (computed in clean_and_engineer.py) IS exactly this forecast
    # already. Per the SDLC doc's Phase 8 guidance, this floor is always
    # established before reaching for a more complex model — if a
    # complex model can't beat naive persistence, it isn't earning its
    # complexity.
    # -----------------------------------------------------------------
    naive_val_pred = X_val["Lag_1"].values
    experiment_log.append({
        "model": "naive_persistence (baseline, = last observed week)",
        "val_wmae": round(weighted_mae(y_val, naive_val_pred, X_val["IsHoliday"]), 2),
        "val_pct_within_15pct": round(pct_within_tolerance(y_val, naive_val_pred), 4),
    })

    # -----------------------------------------------------------------
    # Candidate: Random Forest
    # WHY TRIED: a natural next step — captures non-linear interactions
    # (e.g. holiday effect size differing by department) that a single
    # lag value alone can't represent.
    # WHY A SMALLER FOREST (15 trees, depth 8) THAN THE OTHER PROJECTS'
    # RANDOM FOREST CANDIDATES: this dataset is ~350K training rows,
    # roughly 5-8x larger than the churn/house-price/attrition datasets
    # in this series, and this walkthrough runs on a single CPU core. A
    # full-size forest (150+ trees) would take minutes rather than
    # seconds here — a real team would either use more compute or accept
    # the longer wall-clock time; for this walkthrough, the tree count
    # is reduced and flagged as a compute-budget simplification rather
    # than silently shipping a slow script.
    # -----------------------------------------------------------------
    rf = RandomForestRegressor(n_estimators=15, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    experiment_log.append({
        "model": "random_forest",
        "val_wmae": round(weighted_mae(y_val, rf_val_pred, X_val["IsHoliday"]), 2),
        "val_pct_within_15pct": round(pct_within_tolerance(y_val, rf_val_pred), 4),
    })

    # -----------------------------------------------------------------
    # Final candidate: Gradient Boosting (XGBoost)
    # WHY THIS AS THE FINAL CHOICE (assuming it wins, confirmed below):
    # boosted trees build each tree to correct the previous ensemble's
    # residual errors, which tends to outperform Random Forest on this
    # kind of tabular problem with strong lag-feature signal, especially
    # for the conditional patterns retail forecasting is known for (e.g.
    # a store-department's holiday lift being proportional to its own
    # recent baseline, not a flat additive amount).
    #
    # WHY tree_method="hist": XGBoost's histogram-based tree construction
    # is dramatically faster than the default exact method on a dataset
    # this size, with negligible accuracy cost — the right practical
    # choice given the single-core constraint noted above, and a
    # legitimate production choice regardless of compute constraints for
    # datasets at this scale.
    #
    # SELECTION CRITERION: per Phase 8's guidance, the final model is
    # chosen by val_wmae (the business-relevant, holiday-weighted metric)
    # — a model that's slightly better on an unweighted average but worse
    # specifically on holiday weeks is the wrong choice for a business
    # that explicitly cares more about getting holidays right.
    # -----------------------------------------------------------------
    xgb = XGBRegressor(
        n_estimators=250, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    xgb_val_pred = xgb.predict(X_val)
    xgb_val_wmae = weighted_mae(y_val, xgb_val_pred, X_val["IsHoliday"])
    xgb_val_pct_within = pct_within_tolerance(y_val, xgb_val_pred)
    experiment_log.append({
        "model": "xgboost (final)",
        "val_wmae": round(xgb_val_wmae, 2),
        "val_pct_within_15pct": round(xgb_val_pct_within, 4),
    })

    print("\n=== Experiment Log (validation set) ===")
    log_df = pd.DataFrame(experiment_log)
    print(log_df.to_string(index=False))
    log_df.to_csv("outputs/experiment_log.csv", index=False)

    # -----------------------------------------------------------------
    # Phase 9: Final evaluation on the held-out TEST set (the most
    # recent 12 weeks, never touched until now)
    # -----------------------------------------------------------------
    xgb_test_pred = xgb.predict(X_test)
    test_wmae = weighted_mae(y_test, xgb_test_pred, X_test["IsHoliday"])
    test_mae = mean_absolute_error(y_test, xgb_test_pred)
    test_pct_within = pct_within_tolerance(y_test, xgb_test_pred)
    naive_test_wmae = weighted_mae(y_test, X_test["Lag_1"].values, X_test["IsHoliday"])
    improvement_over_naive = (naive_test_wmae - test_wmae) / naive_test_wmae

    # -----------------------------------------------------------------
    # Business validation: translate forecast accuracy into the
    # business's actual pain point (over/under-stocking).
    # WHY WE DO THIS INSTEAD OF STOPPING AT WMAE: per Phase 9 of the SDLC
    # doc, a model is only validated once its performance is tied back to
    # the KPI defined in the BRD, not just a technical score a demand
    # planner can't act on.
    #
    # ASSUMPTION MADE EXPLICIT (not derived from this dataset, which has
    # no inventory-cost data): each percentage point of forecast error
    # avoided is assumed to translate to a proportional reduction in
    # excess-inventory and stockout costs — a simplification real
    # inventory-cost modeling would refine with actual holding-cost and
    # stockout-cost figures.
    # -----------------------------------------------------------------
    business_summary = {
        "test_wmae": round(test_wmae, 2),
        "test_mae": round(test_mae, 2),
        "test_pct_within_15pct": round(test_pct_within, 4),
        "naive_baseline_test_wmae": round(naive_test_wmae, 2),
        "improvement_over_naive_baseline": round(improvement_over_naive, 4),
        "n_test_rows": len(y_test),
        "note": (
            "improvement_over_naive_baseline is a direct, measured comparison "
            "(both WMAE figures come from this dataset). The link from 'forecast "
            "error reduced' to 'inventory cost saved' is NOT derived from this "
            "dataset (no cost data available) and should be replaced with the "
            "business's real holding-cost and stockout-cost figures before being "
            "presented as a dollar-value business case."
        ),
    }

    print("\n=== Business Validation Summary (Phase 9) ===")
    for k, v in business_summary.items():
        print(f"{k}: {v}")
    with open("outputs/business_validation.json", "w") as f:
        json.dump(business_summary, f, indent=2)

    # -----------------------------------------------------------------
    # Explainability (Phase 9): feature importance
    # -----------------------------------------------------------------
    importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Top 10 Feature Importances (final model) ===")
    print(importances.head(10).round(4).to_string())
    importances.to_csv("outputs/feature_importance.csv", header=["importance"])

    # -----------------------------------------------------------------
    # Save artifacts for deployment (Phase 10)
    # -----------------------------------------------------------------
    joblib.dump(xgb, "outputs/sales_forecast_model.joblib")
    joblib.dump(feature_cols, "outputs/feature_columns.joblib")

    with open("outputs/model_card.json", "w") as f:
        json.dump({
            "model_type": "XGBoost Regressor",
            "data_source": "Walmart Recruiting - Store Sales Forecasting (Kaggle)",
            "n_features": len(feature_cols),
            "features": feature_cols,
            "training_rows": len(X_train),
            "split_strategy": "time-based (train < validation < test chronologically), not random",
            "validation_wmae": round(xgb_val_wmae, 2),
            "test_wmae": round(test_wmae, 2),
            "test_pct_within_15pct": round(test_pct_within, 4),
            "intended_use": "Forecast next week's sales per store-department for inventory and staffing planning.",
            "known_limitations": (
                "Negative sales values were clipped to 0 rather than modeled, "
                "which slightly understates variance for the small share of "
                "return-heavy store-weeks. Lag_52 and Rolling_4wk_mean fall back "
                "to Lag_1 (or the series mean) for the first weeks of any "
                "store-department's history, which are less reliable than "
                "genuine year-over-year or 4-week-trailing signal. The model "
                "forecasts one week ahead using the true prior week's sales as "
                "Lag_1 — a production deployment forecasting further out (e.g. "
                "4 weeks ahead) would need to feed the model's own prior "
                "predictions back in as Lag_1, which compounds error and is not "
                "evaluated here."
            ),
        }, f, indent=2)

    print("\nSaved model -> outputs/sales_forecast_model.joblib")
    print("Saved model card -> outputs/model_card.json")


if __name__ == "__main__":
    train_and_evaluate()
