"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the Walmart "Recruiting - Store Sales Forecasting"
dataset from Kaggle:
    https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

It contains weekly sales for 45 Walmart stores across 81 departments
(421,570 store-department-week rows, 2010-02-05 to 2012-10-26), split
across three files:
    - train.csv    — Store, Dept, Date, Weekly_Sales, IsHoliday
    - features.csv — Store, Date, Temperature, Fuel_Price, 5 MarkDown
                      columns, CPI, Unemployment, IsHoliday
    - stores.csv    — Store, Type (A/B/C), Size

If you're following along on Kaggle: download all three files and place
them at `data/train.csv`, `data/features.csv`, `data/stores.csv` — the
schema is identical to the files already included in this project.

WHY THIS DATASET
-----------------
- It's real, multi-year retail sales data with genuine seasonality,
  holiday effects, and store-to-store heterogeneity — exactly the
  structure a demand-planning team would actually work with, unlike a
  single-snapshot dataset (see split.py for how this changes the
  train/test split strategy compared to the other projects in this
  series).
- It ships with well-documented real-world data quality issues (missing
  markdown data before Nov 2011, a handful of negative sales values,
  missing macroeconomic indicators for some weeks) that make for an
  honest Phase 5 case study — see `clean_and_engineer.py`.
"""

import pandas as pd


def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["Date"])
    features = pd.read_csv(f"{data_dir}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")

    # WHY DROP features' IsHoliday: train.csv already has an IsHoliday
    # column for the exact same (Store, Date) grain — features.csv
    # duplicates it. Keeping both would create a naming collision on
    # merge; train's version is used as the single source of truth.
    features = features.drop(columns=["IsHoliday"])

    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date"], how="left")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns after merging train + features + stores")
    print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Stores: {df['Store'].nunique()}  |  Departments: {df['Dept'].nunique()}  |  Store-Dept series: {df.groupby(['Store', 'Dept']).ngroups}")
    print(f"\nNegative Weekly_Sales rows: {(df['Weekly_Sales'] < 0).sum()} (returns exceeding sales in that week — see clean_and_engineer.py)")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
