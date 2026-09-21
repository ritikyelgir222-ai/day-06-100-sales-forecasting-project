"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: a demand-planning stakeholder cares about three
things first — how big is the holiday effect (so staffing/inventory can
flex for it), which stores/departments actually drive volume (so
forecasting effort is prioritized correctly), and whether the
macroeconomic features are worth the complexity of including them at all.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()

    print(f"=== Holiday vs non-holiday average weekly sales ===")
    print(df.groupby("IsHoliday")["Weekly_Sales"].mean().round(2))
    # WHY: quantifies the holiday effect directly — the business's whole
    # reason for wanting better forecasts is to plan around swings like
    # this rather than react to them after the fact.

    df["Week"] = df["Date"].dt.isocalendar().week
    week_avg = df.groupby("Week")["Weekly_Sales"].mean()
    print(f"\n=== Average sales, weeks around Thanksgiving/Christmas ===")
    print(week_avg.loc[[47, 48, 49, 50, 51, 52, 1]].round(0))
    # WHY: tests exactly WHEN the holiday effect happens — a common
    # misconception is that Christmas week itself is the peak; checking
    # this directly avoids building a feature on a wrong assumption.

    print(f"\n=== Average weekly sales by store type ===")
    print(df.groupby("Type")["Weekly_Sales"].mean().round(2))
    # WHY: confirms whether store Type/Size is worth including as a
    # feature, or whether it's redundant with Store itself.

    print(f"\n=== Correlation: macroeconomic features vs Weekly_Sales ===")
    print(df[["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]].corr()["Weekly_Sales"].round(4))
    # WHY: this is a genuinely useful NEGATIVE finding worth checking
    # explicitly rather than assuming — if these macro features barely
    # correlate with sales at the raw store-week level, that changes
    # whether they're worth the added complexity (spoiler: they don't
    # correlate much on their own; store/department identity and
    # seasonality dominate — see PROJECT_DOCUMENTATION.md Section 4).

    print(f"\n=== Top 5 departments by average weekly sales ===")
    print(df.groupby("Dept")["Weekly_Sales"].mean().sort_values(ascending=False).head(5).round(0))
    print(f"\n=== Bottom 5 departments by average weekly sales ===")
    print(df.groupby("Dept")["Weekly_Sales"].mean().sort_values().head(5).round(0))
    # WHY: confirms departments are NOT interchangeable — a single
    # company-wide average forecast would be meaningless; this is why
    # Dept is kept as an explicit model feature.

    print(f"\n=== Data quality: negative sales rows ===")
    neg = (df["Weekly_Sales"] < 0).sum()
    print(f"{neg} rows ({neg / len(df):.3%}) — returns exceeding purchases in that store-week.")
    print("Handled in clean_and_engineer.py (see the WHY comment there), not silently ignored.")

    # ---- Chart: seasonality + store type + top departments ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    week_avg.plot(ax=axes[0], color="#4C72B0", title="Avg weekly sales by week of year")
    axes[0].axvline(51, color="red", linestyle="--", alpha=0.6, label="Week before Christmas")
    axes[0].set_ylabel("Avg weekly sales ($)")
    axes[0].legend(fontsize=8)

    df.groupby("Type")["Weekly_Sales"].mean().plot(
        kind="bar", ax=axes[1], color="#55A868", title="Avg weekly sales by store type"
    )
    axes[1].set_ylabel("Avg weekly sales ($)")
    axes[1].tick_params(axis="x", rotation=0)

    df.groupby("Dept")["Weekly_Sales"].mean().sort_values(ascending=False).head(10).plot(
        kind="bar", ax=axes[2], color="#C44E52", title="Top 10 departments by avg weekly sales"
    )
    axes[2].set_ylabel("Avg weekly sales ($)")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
