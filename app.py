"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that forecasts next week's sales for a single
store-department. The cleaning and feature-engineering logic from
clean_and_engineer.py is reused here rather than duplicated, so a
transformation change only ever needs to happen in one place.

IMPORTANT DIFFERENCE FROM THE OTHER PROJECTS' APIs IN THIS SERIES: this
model's single most important features (Lag_1, Lag_52, Rolling_4wk_mean —
see the feature-importance table in PROJECT_DOCUMENTATION.md) are NOT
things a caller can meaningfully type in from scratch the way "garage
capacity" or "job level" can — they're derived from a specific
store-department's own recent sales HISTORY. A real production version of
this API would look these up from a sales-history database or feature
store at request time, keyed by (Store, Dept, Date), rather than
requiring the caller to supply them. This demo API accepts them as direct
inputs instead, which is a deliberate, stated simplification (see
PROJECT_DOCUMENTATION.md Section 8) — not an oversight.

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from clean_and_engineer import clean_data, engineer_features

app = FastAPI(title="Weekly Sales Forecast API", version="1.0")

MODEL = joblib.load("outputs/sales_forecast_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")


class ForecastRequest(BaseModel):
    Store: int
    Dept: int
    Date: str  # "YYYY-MM-DD", the week being forecast
    IsHoliday: bool = False
    StoreType: str = "A"  # "A" | "B" | "C"
    StoreSize: int = 150000
    Temperature: float = 60.0
    Fuel_Price: float = 3.5
    CPI: float = 215.0
    Unemployment: float = 7.5
    TotalMarkdown: float = 0.0
    # These three come from sales history — see the module docstring for why
    Lag_1: float  # actual sales for this store-dept, 1 week before Date
    Lag_52: float = None  # actual sales for this store-dept, 52 weeks before Date (falls back to Lag_1 if unknown)
    Rolling_4wk_mean: float = None  # trailing 4-week average sales for this store-dept (falls back to Lag_1 if unknown)


class ForecastResponse(BaseModel):
    forecast_sales: float
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Weekly Sales Forecast</title></head>
    <body style="font-family: sans-serif; max-width: 640px; margin: 40px auto;">
        <h2>Weekly Sales Forecast — Test Form</h2>
        <p>Every field below maps 1:1 to a field in <code>ForecastRequest</code>
           (app.py) — nothing here is hardcoded in the page itself.
           <code>Lag_1</code> is required — it's last week's actual sales for
           this store-department (see the note in app.py for why this can't
           be looked up automatically in this demo). Full API docs at
           <a href="/docs">/docs</a>.</p>
        <form id="forecastForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Store number<br><input name="Store" type="number" value="1" required></label>
            <label>Department number<br><input name="Dept" type="number" value="1" required></label>
            <label>Week to forecast (date)<br><input name="Date" type="date" value="2012-11-02" required></label>
            <label>Is holiday week?<br>
                <select name="IsHoliday">
                    <option value="false" selected>No</option><option value="true">Yes</option>
                </select>
            </label>
            <label>Store type<br>
                <select name="StoreType">
                    <option selected>A</option><option>B</option><option>C</option>
                </select>
            </label>
            <label>Store size (sq ft)<br><input name="StoreSize" type="number" value="151315"></label>
            <label>Last week's actual sales ($) — Lag_1<br><input name="Lag_1" type="number" step="0.01" value="20000" required></label>
            <label>Same week last year's sales ($) — Lag_52<br><input name="Lag_52" type="number" step="0.01" value="21000"></label>
            <label>Trailing 4-week avg sales ($)<br><input name="Rolling_4wk_mean" type="number" step="0.01" value="19500"></label>
            <label>Total active markdown ($)<br><input name="TotalMarkdown" type="number" step="0.01" value="0"></label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Forecast this week</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("forecastForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const form = new FormData(e.target);
            const asFloat = (name) => parseFloat(form.get(name));
            const payload = {
                Store: parseInt(form.get("Store"), 10),
                Dept: parseInt(form.get("Dept"), 10),
                Date: form.get("Date"),
                IsHoliday: form.get("IsHoliday") === "true",
                StoreType: form.get("StoreType"),
                StoreSize: parseInt(form.get("StoreSize"), 10),
                Lag_1: asFloat("Lag_1"),
                Lag_52: form.get("Lag_52") ? asFloat("Lag_52") : null,
                Rolling_4wk_mean: form.get("Rolling_4wk_mean") ? asFloat("Rolling_4wk_mean") : null,
                TotalMarkdown: asFloat("TotalMarkdown")
            };
            const res = await fetch("/forecast", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                document.getElementById("result").innerText =
                    "Error " + res.status + ": " + await res.text();
                return;
            }
            const data = await res.json();
            document.getElementById("result").innerText =
                "Forecast: $" + data.forecast_sales.toLocaleString() +
                " | Top factors: " + data.top_factors.join(", ");
        });
        </script>
    </body>
    </html>
    """


@app.post("/forecast", response_model=ForecastResponse)
def forecast_sales(record: ForecastRequest):
    raw = record.model_dump()
    date = pd.to_datetime(raw["Date"])

    lag_1 = raw["Lag_1"]
    lag_52 = raw["Lag_52"] if raw["Lag_52"] is not None else lag_1
    rolling_4wk = raw["Rolling_4wk_mean"] if raw["Rolling_4wk_mean"] is not None else lag_1

    raw_row = {
        "Store": raw["Store"], "Dept": raw["Dept"], "Date": date,
        "IsHoliday": 1 if raw["IsHoliday"] else 0,
        "Type": raw["StoreType"], "Size": raw["StoreSize"],
        "Temperature": raw["Temperature"], "Fuel_Price": raw["Fuel_Price"],
        "CPI": raw["CPI"], "Unemployment": raw["Unemployment"],
        # clean_data()'s markdown-fillna step expects these 5 columns to
        # exist; feeding the total pre-summed into MarkDown1 and leaving
        # the rest at 0 gives engineer_features() the same TotalMarkdown
        # it would compute from 5 separate real values.
        "MarkDown1": raw["TotalMarkdown"], "MarkDown2": 0, "MarkDown3": 0,
        "MarkDown4": 0, "MarkDown5": 0,
        "Weekly_Sales": 0,  # placeholder; unused for scoring but clean_data()/engineer_features() expect the column
    }
    raw_df = pd.DataFrame([raw_row])
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)

    # WHY WE OVERWRITE THE LAG/ROLLING COLUMNS AFTER engineer_features():
    # engineer_features() computes these via groupby().shift() across the
    # dataframe it's given — for a single-row request with no history in
    # the frame, that computation can only produce NaN. The caller-
    # supplied lag values (from the real sales-history lookup a
    # production system would do) are the actual source of truth here;
    # engineer_features() is still called first so every OTHER
    # transformation (date features, markdown summary, one-hot) stays
    # identical to training.
    engineered["Lag_1"] = lag_1
    engineered["Lag_52"] = lag_52
    engineered["Rolling_4wk_mean"] = rolling_4wk

    X = engineered.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred = float(MODEL.predict(X)[0])

    importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
    top_factors = importances.sort_values(ascending=False).head(3).index.tolist()

    return ForecastResponse(
        forecast_sales=round(pred, 2),
        top_factors=top_factors,
    )
