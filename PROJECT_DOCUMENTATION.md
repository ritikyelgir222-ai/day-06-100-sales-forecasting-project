# Project Documentation: Sales Forecasting
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. It complements `README.md` (setup and run commands)
by explaining the architecture, data, decisions, and results in enough
depth that someone who did not build this project could maintain or extend
it. Every number below came from actually running the code in this
repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Forecast next week's sales per store-department to support inventory and staffing planning |
| **Client context** | Multi-store retail chain's demand-planning team |
| **Data source** | Walmart Recruiting - Store Sales Forecasting (Kaggle) |
| **Dataset size** | 421,570 store-department-week rows, 45 stores, 81 departments, 2010-02-05 to 2012-10-26 |
| **Final model** | XGBoost Regressor, 19 features |
| **Test WMAE** | 1,302.95 (holiday-weighted, vs. 1,653.16 for the naive baseline — a 21.2% improvement) |
| **Business framing** | Forecast accurately enough to reduce both stockouts (under-forecasting) and excess inventory (over-forecasting), with extra weight on getting holiday weeks right |

---

## 2. Architecture

```
data/train.csv, features.csv, stores.csv
        │
        ▼
data_loader.py ──────► merges the 3 raw files into one frame
        │
        ▼
clean_and_engineer.py ─► fixes markdown missingness & negative sales,
        │                extracts date features, builds lag/rolling
        │                sales-history features (the most important
        │                feature group — see Section 5)
        ▼
   split.py ───────────► CHRONOLOGICAL train/val/test split
        │                (train < validation < test in time)
        ▼
train_model.py ───────► naive-persistence baseline, Random Forest,
        │                XGBoost (final); WMAE business validation
        ▼
outputs/sales_forecast_model.joblib, feature_columns.joblib
        │
        ├──────────────► app.py ─── FastAPI service, /forecast endpoint,
        │                           reuses clean_and_engineer.py so
        │                           training and serving logic never
        │                           drift apart
        │
        └──────────────► monitor.py ─ PSI drift check on genuinely
                                       future data + WMAE-increase
                                       retrain trigger
```

The key design principle carried over from the rest of this series:
**cleaning and feature engineering logic lives in exactly one file**
(`clean_and_engineer.py`), imported by both `train_model.py` and `app.py`.

---

## 3. Data Dictionary (key columns, after merging train + features + stores)

| Column | Type | Description | Kept as-is / Transformed / Dropped |
|---|---|---|---|
| `Store`, `Dept` | int | Store and department identifiers | Kept as numeric features — tree models split on these meaningfully without one-hot encoding, unlike a linear model |
| `Weekly_Sales` | float | Sales for that store-department-week | **Target** — negative values (1,285 rows, 0.305%) clipped to 0 (see Section 8) |
| `Type`, `Size` | string, int | Store category (A/B/C) and square footage | `Type` one-hot encoded, `Size` kept as-is |
| `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` | float | Regional macroeconomic conditions | Kept as-is — see Section 4 for why their raw correlation with sales is nearly zero |
| `MarkDown1`-`5` | float | Anonymized promotional markdown amounts | 64-74% missing (program didn't exist before Nov 2011) — filled with 0, summarized into `TotalMarkdown`/`HasMarkdown` |
| `IsHoliday` | bool | Whether the week contains a major US holiday | Binary-encoded |

**Engineered features (not in the raw data):**

| Feature | Formula | Business rationale |
|---|---|---|
| `Year`, `Month`, `WeekOfYear` | From `Date` | Lets the model learn seasonality directly (e.g. the week-51 spike) rather than only from the 4-holiday `IsHoliday` flag |
| `TotalMarkdown`, `HasMarkdown` | Sum / presence of the 5 markdown columns | Compresses 5 sparse columns into one signal |
| `Lag_1` | This store-dept's actual sales, 1 week prior | The single strongest per-series signal (recent weeks resemble recent weeks) |
| `Lag_52` | This store-dept's actual sales, 52 weeks prior | Captures THIS series' own year-over-year seasonality — turned out to be the single most important feature in the model (Section 5) |
| `Rolling_4wk_mean` | This store-dept's trailing 4-week average, shifted to avoid leakage | Smooths single-week noise |

Full reasoning for every decision above is inline in `clean_and_engineer.py`.

---

## 4. Key EDA Findings

From `eda.py`, run against the real dataset:

| Cut | Finding |
|---|---|
| Holiday effect | Holiday weeks average **$17,036** vs. **$15,901** for non-holiday weeks |
| Seasonality (by ISO week) | Week 51 (the week before Christmas) is the actual peak at **$26,396** average — week 52 (Christmas week itself) drops to **$14,543**. Confirms shoppers buy BEFORE Christmas, not during Christmas week — an assumption worth checking rather than guessing |
| Store type | Type A stores average **$20,100**/week, Type B **$12,237**, Type C **$9,520** |
| Macro correlations | Temperature (r=-0.002), Fuel_Price (r=-0.0001), CPI (r=-0.021), Unemployment (r=-0.026) — **all essentially zero** at the raw store-week level. Store/department identity and seasonality dominate; these macro features are kept in the model but this finding sets honest expectations for how much they'll matter |
| Department variance | Dept 92 averages **$75,205**/week vs. Dept 43 at **$1**/week — departments are not remotely interchangeable, confirming `Dept` needed to be a first-class model feature |
| Data quality | 1,285 rows (0.305%) have negative sales — returns exceeding purchases in that store-week, a real event, not a data error |

These findings directly shaped the feature set: the near-zero macro correlations are exactly why `Lag_1`/`Lag_52`/`Rolling_4wk_mean` were built — a store-department's own history carries far more signal than any external economic indicator on its own.

---

## 5. Modeling Results

### Experiment log (validation set — 12 weeks, 2012-05-18 to 2012-08-03)

| Model | Val WMAE | Val % within ±15% |
|---|---|---|
| Naive persistence (baseline) | 1,591.01 | 0.6408 |
| Random Forest (15 trees, depth 8 — see Section 8) | 1,318.93 | **0.6730** |
| **XGBoost (final)** | **1,279.85** | 0.6701 |

**Note on model selection:** Random Forest actually had a marginally
higher `% within ±15%` than XGBoost (0.6730 vs. 0.6701), but XGBoost won
on `val_wmae` (1,279.85 vs. 1,318.93) — the metric that directly encodes
the business's stated priority (holiday weeks matter 5x more). This is
the same kind of close, honestly-reported model-selection call made
elsewhere in this series (see the attrition project's AUC-vs-recall
tradeoff) — the two metrics didn't fully agree, so the metric tied to
the actual business cost function won.

### Held-out test set (final, unbiased evaluation — most recent 12 weeks, 2012-08-10 to 2012-10-26)

| Metric | Value |
|---|---|
| Test WMAE | 1,302.95 |
| Test MAE (unweighted) | 1,239.03 |
| Test % within ±15% | 66.60% |
| Naive baseline test WMAE | 1,653.16 |
| **Improvement over naive baseline** | **21.18%** |
| N test rows | 35,563 |

### What drives the model (feature importance)

Top 5 by importance:

1. `Lag_52` (0.5582) — same week last year, by a wide margin the strongest driver
2. `Lag_1` (0.2637) — last week's actual sales
3. `Rolling_4wk_mean` (0.0736)
4. `IsHoliday` (0.0188)
5. `Type_A` (0.0139)

The three sales-history features (`Lag_52`, `Lag_1`, `Rolling_4wk_mean`)
together account for **89.6%** of the model's total decision weight —
confirming the EDA-driven hypothesis that a store-department's own
history dominates over any static attribute or external economic
indicator. This is worth calling out explicitly in the LinkedIn post:
it's a genuinely useful, somewhat counter-intuitive finding for a
learner who might assume weather or store size would matter more.

### Business validation

On the 35,563-row held-out test set (most recent 12 weeks):

- The model beat the naive "no change from last week" baseline by
  **21.18%** on WMAE — a real, measured improvement, not an assumption
- **66.60%** of forecasts landed within ±15% of actual sales
- Converting this into a dollar-value inventory-cost saving requires the
  business's actual holding-cost and stockout-cost figures, which this
  dataset does not contain — that conversion is explicitly NOT done here
  (see `business_validation.json`'s note field), unlike the other
  projects in this series where an illustrative dollar figure was at
  least estimated. This is a deliberate choice: unlike a plausible
  "35% retention success rate," there's no similarly reasonable industry
  rule-of-thumb to assume for inventory cost that wouldn't be
  misleading to state as a number.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser-based test form (every field wired) |
| `/docs` | GET | Auto-generated interactive API docs (Swagger UI) |
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/forecast` | POST | Forecast a single store-department-week — see request/response shape below |

**Important:** `Lag_1` is a **required** field, and `Lag_52`/
`Rolling_4wk_mean` are strongly recommended — these come from a real
sales-history lookup in production, not from the caller's general
knowledge (see the module docstring in `app.py`).

**Response (verified against two contrasting real scenarios):**
```json
{
  "forecast_sales": 33900.89,
  "top_factors": ["Lag_52", "Lag_1", "Rolling_4wk_mean"]
}
```
A regular (non-holiday) week at a smaller store with lower recent sales
scored **$8,806.67** on the same model — confirming the API responds to
genuinely different sales-history inputs, not a fixed value.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` implements, and was run to confirm works correctly — with a
genuinely different result from the other projects in this series:

- **Feature drift check** via PSI on `Fuel_Price`, `CPI`, `Unemployment`,
  `Temperature`, `Lag_1`. Unlike the churn/house-price/attrition
  projects (where train and "new batch" come from the same snapshot and
  PSI is expected to be near-zero), **this test set is genuinely 2.5+
  years later than the start of training**, so real drift shows up:
  `Fuel_Price` (PSI=4.07), `CPI` (PSI=2.87), `Unemployment` (PSI=0.55),
  and `Temperature` (PSI=1.31) all flagged as significant drift.
  `Lag_1` stayed stable (PSI=0.001).
  **Interpretation matters here, not just the number**: Fuel_Price and
  CPI genuinely trended over 2010-2012 (real economic drift a production
  system should care about), but Temperature's "drift" is largely
  because the test window (Aug-Oct) covers different seasons than the
  full-year training window — that's expected seasonal variation, not a
  data problem. A monitoring dashboard that flags all four identically
  as "drift" without this context would send a demand planner chasing
  the wrong signal.
- **Performance decay check**: compares current WMAE against the
  baseline in `business_validation.json`; triggers a retrain
  recommendation if it rises more than 15%. On this run: no increase (same
  test set), so no retrain triggered — confirming the check works before
  it's pointed at genuinely new data next quarter.

**In production**, this script should run monthly against the actual
next month's sales.

---

## 8. Known Limitations (stated for the handover record)

1. **Negative sales clipped to 0, not modeled.** Slightly understates
   variance for the 0.305% of return-heavy store-weeks.
2. **Lag_52/Rolling_4wk_mean fall back to Lag_1 (or the series mean)**
   for a store-department's first weeks of history — less reliable than
   genuine year-over-year or trailing-average signal for those rows.
3. **One-week-ahead only.** The model uses the TRUE prior week's actual
   sales as `Lag_1`. Forecasting further out (e.g. 4 weeks ahead) would
   require feeding the model's own predictions back in as `Lag_1`,
   compounding error in a way this evaluation doesn't measure.
4. **No dollar-value business case calculated** — only the measured WMAE
   improvement (21.18%) is reported; converting to a cost figure needs
   real holding-cost/stockout-cost data this dataset lacks.
5. **Random Forest and XGBoost were trained at reduced size** (15 trees;
   250 estimators at depth 7) due to a single-CPU-core compute
   constraint in this environment — a full-size forest would likely
   change the gap between the two candidates, though XGBoost's use of
   `tree_method="hist"` is a legitimate production choice regardless of
   compute constraints at this data scale.
6. **The API requires the caller to supply sales-history features
   directly** rather than looking them up automatically — a stated
   simplification for a demo endpoint (see `app.py`'s module docstring).

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load and merge the 3 raw files, document source |
| `eda.py` | 6 | Seasonality, store type, department variance, macro correlations |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, date features, lag/rolling sales-history features — fully commented |
| `split.py` | 7 | Chronological train/val/test split |
| `train_model.py` | 8-9 | Naive baseline, model training, WMAE business validation |
| `app.py` | 10 | FastAPI forecasting service + fully-wired test form |
| `monitor.py` | 13 | Drift detection (on genuinely future data), retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
