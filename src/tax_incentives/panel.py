### Builds the canonical (state, period) panel and persists it to CSV.

## Because economic dataset already includes quarterly population,
## we treat it as the “truth” for the canonical quarterly panel.
## Demographics becomes “enrichment.”

from __future__ import annotations

import pandas as pd

from .io import assert_required_columns


def build_state_quarter_panel(
    economics: pd.DataFrame,
    demographics_quarterly: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Canonical panel: one row per (state, period).
    Uses economics as the base (it already has quarterly population, GDP, unemployment, etc.)
    Optionally merges in quarterly-expanded demographics.
    """
    econ = economics.copy()

    keep_econ = ["state", "period", "year", "quarter", "population", "gdp_total", "unemployment_rate"]
    econ = econ[keep_econ].copy()

    assert_required_columns(econ, ["state", "period", "population", "gdp_total"], "economics_base_panel")

    panel = econ

    if demographics_quarterly is not None:
        demo = demographics_quarterly.copy()

        # Keep a small set now; teammates can expand later
        keep_demo = ["state", "period"]
        for c in ["pop_youth", "pop_working", "pop_senior", "age_median"]:
            if c in demo.columns:
                keep_demo.append(c)

        demo = demo[keep_demo].drop_duplicates(["state", "period"])
        panel = panel.merge(demo, on=["state", "period"], how="left")

    panel = panel.sort_values(["state", "period"]).reset_index(drop=True)
    return panel


def write_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
