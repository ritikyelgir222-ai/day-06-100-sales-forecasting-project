"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation, elimination, and engineered feature below has a
one-line WHY comment next to it — the intent is that someone reviewing
this file (a stakeholder, a teammate, or future-you) can audit every
decision without having to guess at the reasoning.

FINAL FEATURE LIST is built at the bottom as FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd

MARKDOWN_COLUMNS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
ONE_HOT_COLUMNS = ["Type"]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: MarkDown1-5 are missing for the large majority of rows
    # (64-74% each). WHY IT HAPPENS: per the dataset's own documentation,
    # Walmart's markdown promotion program only started being tracked in
    # November 2011 — earlier rows are missing NOT because of a data
    # collection error, but because the program didn't exist yet.
    # BUSINESS LOGIC: filled with 0, meaning "no markdown was running,"
    # which is factually correct for the pre-program period and for
    # store-weeks with no active promotion after the program started —
    # not a statistical guess.
    # -----------------------------------------------------------------
    for col in MARKDOWN_COLUMNS:
        df[col] = df[col].fillna(0)

    # -----------------------------------------------------------------
    # Fix 2: 1,285 rows (0.305%) have negative Weekly_Sales.
    # WHY IT HAPPENS: a week where returns/refunds exceeded new purchases
    # for that store-department — a genuine, if unusual, retail event,
    # not a data entry error (values are small and plausible, not wildly
    # out of range).
    # BUSINESS LOGIC: clipped to 0 rather than dropped. A forecasting
    # model's job is to predict expected DEMAND, and negative demand
    # isn't a meaningful concept for inventory/staffing planning — but
    # dropping the rows entirely would also silently remove real
    # observations from genuinely slow-moving store-departments. Clipping
    # keeps the row (and its other feature values) while making the
    # target sensible. Flagged as a modeling simplification, not hidden
    # — see PROJECT_DOCUMENTATION.md Section 8.
    # -----------------------------------------------------------------
    df["Weekly_Sales"] = df["Weekly_Sales"].clip(lower=0)

    return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Date features
    # BUSINESS LOGIC: Year/Month/WeekOfYear let the model learn the
    # seasonality EDA found directly (e.g. the week-51 pre-Christmas
    # spike) without having to rediscover it purely from IsHoliday alone,
    # which only flags 4 specific weeks a year and says nothing about
    # the broader run-up to them.
    # -----------------------------------------------------------------
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    # -----------------------------------------------------------------
    # Markdown summary features
    # BUSINESS LOGIC: TotalMarkdown compresses 5 individually sparse
    # columns into one "how much total promotion is running" signal;
    # HasMarkdown gives the model an explicit binary split for "any
    # promotion at all" vs. "none," which is plausibly more informative
    # than 5 separate near-zero columns for the ~64-74% of rows with no
    # markdown running.
    # -----------------------------------------------------------------
    df["TotalMarkdown"] = df[MARKDOWN_COLUMNS].sum(axis=1)
    df["HasMarkdown"] = (df["TotalMarkdown"] > 0).astype(int)

    # -----------------------------------------------------------------
    # One-hot encode store Type (A/B/C) — no ordinal relationship
    # -----------------------------------------------------------------
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    dummy_cols = [c for c in df.columns if c.startswith("Type_")]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # -----------------------------------------------------------------
    # Lag & rolling features — the single most important feature group
    # for this problem.
    # WHY THESE ARE ESSENTIAL FOR A FORECASTING PROBLEM (unlike the
    # other projects in this series): a store-department's own recent
    # sales history is almost always the single strongest predictor of
    # its near-future sales — far more informative than any static
    # attribute like store Type or Size alone. Sorting by
    # (Store, Dept, Date) and using shift() ensures every lag/rolling
    # value for a given row uses ONLY that store-department's own past
    # weeks, never a future week — this is what makes these features
    # leakage-safe to compute BEFORE the train/test split (see split.py
    # for why the split itself still has to be time-based regardless).
    #
    # Lag_1: last week's actual sales for this exact store-department —
    #        the strongest, simplest signal (recent weeks look like
    #        recent weeks).
    # Lag_52: same week one year ago — captures yearly seasonality
    #        directly (e.g. this store-department's own historical
    #        Thanksgiving spike), which WeekOfYear alone can only
    #        capture as an AVERAGE across all store-departments, not
    #        this specific one's own pattern.
    # Rolling_4wk_mean: trailing 4-week average, shifted so it never
    #        includes the current week — smooths out single-week noise
    #        that Lag_1 alone is sensitive to.
    # -----------------------------------------------------------------
    df = df.sort_values(["Store", "Dept", "Date"])
    grp = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    df["Lag_1"] = grp.shift(1)
    df["Lag_52"] = grp.shift(52)
    # WHY transform() here rather than grp.shift(1).rolling(4): applying
    # .rolling() directly to a plain shifted Series ignores the
    # (Store, Dept) group boundaries and would compute a rolling window
    # that bleeds across different store-departments. transform() runs
    # the shift+rolling calculation separately within each group.
    df["Rolling_4wk_mean"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )

    # WHY FILL MISSING LAGS WITH Lag_1 (and Lag_1 itself with the
    # store-department's own mean): the first weeks of any
    # store-department's history have no prior week (Lag_1) or no prior
    # year (Lag_52) to look back on. Rather than dropping these rows
    # (which would throw away real early-history observations) or
    # filling with 0 (which would look like "predict zero sales," a
    # meaningless signal for a store that's clearly open and selling),
    # missing Lag_52/Rolling_4wk_mean fall back to Lag_1 (the next-best
    # available recent-history signal), and any still-missing Lag_1
    # (true first week of a series) falls back to that store-department's
    # own overall mean — a reasonable "no history yet, use the series'
    # typical level" default.
    # -----------------------------------------------------------------
    series_mean = df.groupby(["Store", "Dept"])["Weekly_Sales"].transform("mean")
    df["Lag_1"] = df["Lag_1"].fillna(series_mean)
    df["Lag_52"] = df["Lag_52"].fillna(df["Lag_1"])
    df["Rolling_4wk_mean"] = df["Rolling_4wk_mean"].fillna(df["Lag_1"])

    return df


NUMERIC_MODEL_FEATURES = [
    "Store", "Dept", "Size", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Year", "Month", "WeekOfYear", "TotalMarkdown", "HasMarkdown",
    "Lag_1", "Lag_52", "Rolling_4wk_mean",
]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if c.startswith("Type_")]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw merged columns: {len(raw.columns)}  ->  Final feature columns: {len(feature_cols)}")
    print("\nFinal feature columns:")
    for c in feature_cols:
        print(f"  - {c}")
    print(f"\nRemaining nulls in feature columns: {engineered[feature_cols].isnull().sum().sum()}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
