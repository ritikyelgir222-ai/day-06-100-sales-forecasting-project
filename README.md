# Sales Forecasting — Full Project Code (Real Dataset)

Companion code to Day 6 of the 100-day LinkedIn series. Uses the real
Walmart weekly sales dataset, run end to end — every number in
`PROJECT_DOCUMENTATION.md` came from executing this code, not from an
illustrative example.

## Data source

**Walmart Recruiting - Store Sales Forecasting** (Kaggle) — 421,570
store-department-week rows across 45 stores and 81 departments,
2010-02-05 to 2012-10-26. Three files: `train.csv`, `features.csv`
(temperature, fuel price, CPI, unemployment, markdowns), `stores.csv`
(store type and size).

All three CSVs are already included at `data/`. If you'd rather pull
them fresh from Kaggle: [the competition
page](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
— the schema is identical, so nothing else changes.

## What's genuinely different about this project vs. Days 3-4

This is the first **time-series** project in the series, and three
things change because of that — not stylistic choices, but real
methodology differences:

1. **The split is chronological, not random.** `split.py` trains on the
   earliest ~2.3 years, validates on the next 12 weeks, and tests on the
   most recent 12 weeks. A random split would let the model "see the
   future" during training — a much more serious leakage risk here than
   in the earlier cross-sectional projects.
2. **The baseline is naive persistence, not a simple regression.** For
   forecasting, "predict no change from last week" is the standard,
   honest floor a model has to beat — see `train_model.py`.
3. **The business metric is WMAE (holiday-weighted), not MAPE.** This is
   literally Kaggle's own competition metric — holiday weeks count 5x,
   because a bad Thanksgiving/Christmas forecast is far costlier to the
   business than a bad ordinary week.

## Why every script is commented the way it is

A few highlights beyond the three points above:

- **`Lag_1`, `Lag_52`, and `Rolling_4wk_mean`** (a store-department's own
  recent sales history) turned out to be the 3 most important features
  by a wide margin — `Lag_52` (same week last year) alone accounts for
  over half the model's decision weight. This is a genuinely useful,
  measured finding, not an assumption.
- **Negative sales (returns exceeding purchases) are clipped to 0, not
  dropped** — a documented modeling simplification, not silently
  ignored.
- **Model sizes were deliberately reduced** (`train_model.py`'s Random
  Forest is 15 trees, not 150+) because this walkthrough runs on a single
  CPU core against a ~350K-row training set — stated honestly as a
  compute-budget choice, not hidden.
- **`monitor.py`'s drift check is genuinely different from the other
  projects'** — because the test set here is real future data, not a
  same-snapshot stand-in, the PSI drift check actually found significant
  drift in Fuel_Price, CPI, Unemployment, and Temperature. That's the
  monitoring code working correctly, not a bug — see
  `PROJECT_DOCUMENTATION.md` Section 7 for why some of that "drift" is
  expected seasonal variation, not a problem.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & merges train + features + stores
python eda.py                  # Phase 6 — seasonality, store type, macro correlations + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — cleaning, lag/rolling features
python train_model.py          # Phase 8-9 — naive baseline + Random Forest + XGBoost, business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
        "Store": 1, "Dept": 1, "Date": "2012-12-21", "IsHoliday": true,
        "StoreType": "A", "StoreSize": 151315,
        "Lag_1": 25000, "Lag_52": 27000, "Rolling_4wk_mean": 24000
      }'
```

Expected: a forecast around **$33,900** — this profile (large store,
pre-Christmas holiday week, high recent sales) matches the seasonal spike
EDA identified around week 51.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads and merges the 3 raw files, documents source |
| `eda.py` | 6 | Seasonality, store type, department variance, macro-feature correlation, data quality checks |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, markdown handling, date features, lag/rolling features — **fully commented with reasoning** |
| `split.py` | 7 | Time-based (chronological) train/val/test split, with reasoning |
| `train_model.py` | 8-9 | Naive-persistence baseline → Random Forest → XGBoost, WMAE business-metric evaluation, feature importance |
| `app.py` | 10 | FastAPI forecasting service — every form field wired, none hardcoded |
| `monitor.py` | 13 | PSI-based drift check + WMAE-increase retrain trigger, on genuinely future data |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`
- `eda_summary.png`
- `experiment_log.csv` — naive baseline vs. candidate models
- `business_validation.json` — WMAE, % within ±15%, improvement over naive baseline
- `feature_importance.csv`
- `sales_forecast_model.joblib`, `feature_columns.joblib`
- `model_card.json` — includes explicit known limitations

## Known limitations (stated honestly, not hidden)

- **Negative sales values are clipped to 0**, slightly understating
  variance for the small share of return-heavy store-weeks.
- **Lag_52 and Rolling_4wk_mean fall back to Lag_1** (or the series mean)
  for the first weeks of any store-department's history — less reliable
  than genuine year-over-year or 4-week-trailing signal for those rows.
- **This model forecasts one week ahead using the TRUE prior week's
  sales as Lag_1.** A production deployment forecasting further out
  (e.g. 4 weeks ahead) would need to feed the model's own prior
  predictions back in as Lag_1, which compounds error — not evaluated
  here.
- **The dollar-value business case (inventory cost savings) is not
  calculated** — only the measured WMAE improvement over the naive
  baseline (21.2%) is real; converting that into a cost figure needs the
  business's actual holding-cost and stockout-cost data, which this
  dataset doesn't include.
- **Random Forest and XGBoost were trained with reduced size** for
  single-core compute budget reasons — a full-size forest would likely
  narrow or change the gap between the two candidates.
