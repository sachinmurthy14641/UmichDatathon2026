### Starter outcome set: sector concentration (if sector columns exist)
### + dependency ratio (if demo columns exist) + simple drift measures.

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assert_required_columns


def build_outcomes(
    panel_state_quarter: pd.DataFrame,
    demographics: pd.DataFrame,
    economics: pd.DataFrame,
    drift_lag_q: int = 8,
) -> pd.DataFrame:
    """
    Outputs: one row per (state, period) with a few starter structural outcomes.
    Expand later (sector shares, HHI on sector GDP, etc.).
    """
    panel = panel_state_quarter.copy()

    # Bring in additional demo columns if available
    demo_cols = ["state", "period"]
    optional_demo = ["pop_youth", "pop_working", "pop_senior", "age_median"]
    demo = demographics.copy()
    for c in optional_demo:
        if c in demo.columns:
            demo_cols.append(c)
    demo = demo[demo_cols].copy()

    out = panel.merge(demo, on=["state", "period"], how="left")

    # Dependency ratio if we have cohorts
    if all(c in out.columns for c in ["pop_youth", "pop_working", "pop_senior"]):
        out["dependency_ratio"] = (out["pop_youth"] + out["pop_senior"]) / out["pop_working"].replace({0: np.nan})
        out["senior_share"] = out["pop_senior"] / out["population"].replace({0: np.nan})
        out["working_share"] = out["pop_working"] / out["population"].replace({0: np.nan})

        # drift over time (lagged change)
        out = out.sort_values(["state", "period"]).reset_index(drop=True)
        for c in ["dependency_ratio", "senior_share", "working_share"]:
            out[f"delta_{c}_lag{drift_lag_q}"] = out.groupby("state")[c].diff(drift_lag_q)

    # GDP per capita + drift
    out["gdp_per_capita"] = out["gdp_total"] / out["population"].replace({0: np.nan})
    out = out.sort_values(["state", "period"]).reset_index(drop=True)
    out[f"delta_gdp_per_capita_lag{drift_lag_q}"] = out.groupby("state")["gdp_per_capita"].diff(drift_lag_q)

    keep = ["state", "period", "gdp_per_capita", f"delta_gdp_per_capita_lag{drift_lag_q}"]
    for c in ["dependency_ratio", "senior_share", "working_share",
              f"delta_dependency_ratio_lag{drift_lag_q}",
              f"delta_senior_share_lag{drift_lag_q}",
              f"delta_working_share_lag{drift_lag_q}"]:
        if c in out.columns:
            keep.append(c)

    final = out[keep].copy().sort_values(["state", "period"]).reset_index(drop=True)
    return final
