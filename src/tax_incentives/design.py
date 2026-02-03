### Creates event flags (GFC/COVID) + a simple “high vs low exposure” cohort based on one bucket.

from __future__ import annotations

import pandas as pd


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple national event window flags based on period string YYYYQ#.
    Adjust windows as desired.
    """
    out = df.copy()

    def _in_range(p: str, start: str, end: str) -> bool:
        return start <= p <= end

    out["event_gfc"] = out["period"].apply(lambda p: int(_in_range(p, "2008Q3", "2010Q4")))
    out["event_covid"] = out["period"].apply(lambda p: int(_in_range(p, "2020Q1", "2021Q4")))
    return out


def add_exposure_cohort(
    df: pd.DataFrame,
    exposure_col: str,
    baseline_period_start: str = "2015Q1",
    baseline_period_end: str = "2019Q4",
) -> pd.DataFrame:
    """
    Cohort label: high vs low based on baseline-period average exposure_col.
    """
    out = df.copy()
    base = out[(out["period"] >= baseline_period_start) & (out["period"] <= baseline_period_end)]
    baseline_avg = base.groupby("state")[exposure_col].mean().rename("baseline_exposure")
    out = out.merge(baseline_avg, on="state", how="left")

    # median split across states
    med = out["baseline_exposure"].median()
    out["cohort_high_exposure"] = (out["baseline_exposure"] >= med).astype(int)
    return out
