"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A TIME-BASED SPLIT HERE (unlike the stratified random splits used in
the other projects in this series): this dataset is genuinely
longitudinal — the same 3,331 store-department series observed weekly
over ~2.7 years — so a random split would let the model train on, say,
March 2011 AND test on January 2011 for the same store, which is not
just unrealistic but actively leaks future information backward. A
forecasting model has to be evaluated the way it will actually be used:
trained on the past, tested on weeks it has never seen, all of which
come after every training week.

WHY THE LAST 12 WEEKS AS TEST (and the 12 before that as validation):
12 weeks (~1 quarter) is a realistic forecasting horizon for a retail
demand-planning cycle — long enough to be meaningful, short enough that
the model isn't evaluated on a period so far removed from training that
the comparison stops being useful.
"""

import pandas as pd


def split_data(df: pd.DataFrame, date_col: str = "Date", test_weeks: int = 12, val_weeks: int = 12):
    max_date = df[date_col].max()
    test_cutoff = max_date - pd.Timedelta(weeks=test_weeks)
    val_cutoff = test_cutoff - pd.Timedelta(weeks=val_weeks)

    train_df = df[df[date_col] <= val_cutoff].copy()
    val_df = df[(df[date_col] > val_cutoff) & (df[date_col] <= test_cutoff)].copy()
    test_df = df[df[date_col] > test_cutoff].copy()

    return train_df, val_df, test_df
