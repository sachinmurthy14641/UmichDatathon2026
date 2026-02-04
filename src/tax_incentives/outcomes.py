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

# --- ECON SECTOR COLUMNS (for structural outcomes) ---
    econ = economics.copy()

    # We only need a subset of columns: keys + totals + selected sectors
    # (We'll define the sector list in the next step.)
    econ_cols = ["state", "period", "gdp_total", "gdp_private_total", "gdp_gov_total"]

    # We'll extend econ_cols with sector columns after we define them.

 # --> broad, non-overlapping private sectors + government total
    # Logic : exclude overlapping subcomponents like durable/nondurable if manufacturing total exists.
    sector_cols = [
        "gdp_ag_fish_hunt",
        "gdp_mining_oil_gas",
        "gdp_utilities",
        "gdp_construction",
        "gdp_manufacturing",          # keep total manufacturing 
        "gdp_wholesale",
        "gdp_retail",
        "gdp_transport_warehousing",
        "gdp_information",
        "gdp_finance_insurance",
        "gdp_real_estate",
        "gdp_prof_tech_services",
        "gdp_management",
        "gdp_admin_waste",
        "gdp_education",
        "gdp_healthcare_social",
        "gdp_arts_recreation",
        "gdp_accom_food",
        "gdp_other_services",
        "gdp_gov_total",              # government treated as one sector
    ]

    # Filter to only columns that actually exist in this dataset
    sector_cols = [c for c in sector_cols if c in econ.columns]

    # Add those to the econ merge list
    econ_cols = [c for c in econ_cols if c in econ.columns] + sector_cols

    econ = econ[econ_cols].copy()
    out = out.merge(econ, on=["state", "period"], how="left")

 # Private vs government GDP shares 
    if all(c in out.columns for c in ["gdp_total", "gdp_private_total"]):
        out["gdp_private_share"] = out["gdp_private_total"] / out["gdp_total"].replace({0: np.nan})

    if all(c in out.columns for c in ["gdp_total", "gdp_gov_total"]):
        out["gdp_gov_share"] = out["gdp_gov_total"] / out["gdp_total"].replace({0: np.nan})

# --- Economic structure: sector shares -> HHI, entropy ---
    if ("gdp_total" in out.columns) and (len(sector_cols) >= 2):
        total = out["gdp_total"].replace({0: np.nan})

        shares = out[sector_cols].div(total, axis=0)
        shares = shares.where(shares > 0)

        out["econ_hhi"] = (shares ** 2).sum(axis=1, skipna=True)

        K = len(sector_cols)
        entropy = -(shares * np.log(shares)).sum(axis=1, skipna=True)
        out["econ_entropy_norm"] = entropy / np.log(K) if K > 1 else np.nan
    
    # Dependency ratio if we have cohorts
    if all(c in out.columns for c in ["pop_youth", "pop_working", "pop_senior"]):
        out["dependency_ratio"] = (out["pop_youth"] + out["pop_senior"]) / out["pop_working"].replace({0: np.nan})
        out["senior_share"] = out["pop_senior"] / out["population"].replace({0: np.nan})
        out["working_share"] = out["pop_working"] / out["population"].replace({0: np.nan})

        # drift over time (lagged change)
        out = out.sort_values(["state", "period"]).reset_index(drop=True)
        for c in ["dependency_ratio", "senior_share", "working_share"]:
            out[f"delta_{c}_lag{drift_lag_q}"] = out.groupby("state")[c].diff(drift_lag_q)
        
        # Optional analysis for drift: Compute L1 distance between share vectors at t and t-L
        # shares_mat = shares.to_numpy()
        # # We'll build lagged shares by shifting within each state
        # def lagged_matrix(arr, grp, lag):
        #     out_arr = np.full_like(arr, np.nan, dtype=float)
        #     for s in grp.unique():
        #         idx = (grp == s).to_numpy()
        #         out_arr[idx] = np.roll(arr[idx], shift=lag, axis=0)
        #         out_arr[idx][:lag] = np.nan
        #     return out_arr

        # lagged = lagged_matrix(shares_mat, out["state"], drift_lag_q)
        # out[f"econ_structural_change_l1_lag{drift_lag_q}"] = np.nansum(np.abs(shares_mat - lagged), axis=1)

    # GDP per capita + drift
    out["gdp_per_capita"] = out["gdp_total"] / out["population"].replace({0: np.nan})
    out = out.sort_values(["state", "period"]).reset_index(drop=True)
    out[f"delta_gdp_per_capita_lag{drift_lag_q}"] = out.groupby("state")["gdp_per_capita"].diff(drift_lag_q)

    keep = ["state", "period", "gdp_per_capita", f"delta_gdp_per_capita_lag{drift_lag_q}"]

# Demographic Outcomes:    
    for c in ["dependency_ratio", "senior_share", "working_share",
              f"delta_dependency_ratio_lag{drift_lag_q}",
              f"delta_senior_share_lag{drift_lag_q}",
              f"delta_working_share_lag{drift_lag_q}"]:
        if c in out.columns:
            keep.append(c)

# Economic structure Outcomes:
    for c in [
        "econ_hhi",
        "econ_entropy_norm",
        "gdp_private_share",
        "gdp_gov_share",
        f"econ_structural_change_l1_lag{drift_lag_q}",
    ]:
        if c in out.columns:
            keep.append(c)
            
    final = out[keep].copy().sort_values(["state", "period"]).reset_index(drop=True)
    return final
