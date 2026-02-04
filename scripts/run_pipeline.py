### runs end to end pipeline

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tax_incentives.config import get_paths, DEFAULT_BUCKETS_ORDER
from tax_incentives.io import (
    load_tax_revenue_csv,
    load_economics_csv,
    load_demographics_csv_annual_expand_to_quarterly,
)
from tax_incentives.panel import build_state_quarter_panel, write_csv
from tax_incentives.signals import build_tax_signals
from tax_incentives.outcomes import build_outcomes
from tax_incentives.design import add_event_flags, add_exposure_cohort
from tax_incentives.models_panel import fit_fe_ols
from tax_incentives.viz import save_line_plot
from tax_incentives.validation import run_step5

def main() -> None:
    paths = get_paths()

    # ---- Input raw files (renamed to actual names) ----
    tax_path = paths.data_raw / "1. DatathonMasterStateTaxData_2004_2025Q2.csv"
    econ_path = paths.data_raw / "2. DatathonMasterEconomicDataset_2004_2025Q2.csv"
    demo_path = paths.data_raw / "3. (optional) DatathonAnnualDemographicsDataset.csv"

    tax = load_tax_revenue_csv(tax_path)
    econ = load_economics_csv(econ_path)
    demo_q = load_demographics_csv_annual_expand_to_quarterly(demo_path)


    # ---- Step 0: Canonical panel ----
    panel = build_state_quarter_panel(economics=econ, demographics_quarterly=demo_q)
    write_csv(panel, paths.data_processed / "panel_state_quarter.csv")

    # ---- Step 1: Tax signals ----
    signals = build_tax_signals(tax_revenue=tax, panel_state_quarter=panel, rolling_q=12, lag_q=8)
    write_csv(signals, paths.data_processed / "tax_signals.csv")

    # ---- Step 2: Outcomes ----
    outcomes = build_outcomes(panel_state_quarter=panel, demographics=demo_q, economics=econ, drift_lag_q=8)
    write_csv(outcomes, paths.data_processed / "outcomes.csv")

    # ---- Step 3: Design table ----
    design = signals.merge(outcomes, on=["state", "period"], how="inner")
    design = add_event_flags(design)

    # pick a default exposure column for cohort split (edit if you want)
    # Use lagged rolling share if present:
    exposure_candidate = None
    for b in DEFAULT_BUCKETS_ORDER:
        cand = f"{b}_roll12_lag8"
        if cand in design.columns:
            exposure_candidate = cand
            break
    if exposure_candidate is None:
        # fallback to raw bucket share
        exposure_candidate = DEFAULT_BUCKETS_ORDER[0] if DEFAULT_BUCKETS_ORDER[0] in design.columns else "other"

    design = add_exposure_cohort(design, exposure_col=exposure_candidate)
    write_csv(design, paths.data_processed / "analysis_design.csv")

    # ---- Step 4: One quick FE model (thin slice) ----
    # Example: do lagged exposure predict gdp_per_capita drift?
    y = "delta_gdp_per_capita_lag8"
    x = exposure_candidate
    controls = ["unemployment_rate"]

    if y in design.columns and x in design.columns:
        coef_table = fit_fe_ols(design, y=y, x=x, controls=controls)
        write_csv(coef_table, paths.outputs_tables / "panel_model_summary.csv")
    else:
        print(f"Skipping model: missing y={y} or x={x}")

    # Step 5: full validation & synthesis
    run_step5(design=design, panel=panel, paths=paths)

    # ---- Step 5: One quick plot for storytelling ----
    # Plot gdp_per_capita over time for a few states
    plot_df = panel.copy()
    plot_df["gdp_per_capita"] = plot_df["gdp_total"] / plot_df["population"].replace({0: pd.NA})
    save_line_plot(
        df=plot_df,
        x="period",
        y="gdp_per_capita",
        group="state",
        title="GDP per Capita (sample states)",
        out_path=paths.outputs_figures / "gdp_per_capita_sample.png",
        max_groups=8,
    )

    print("Done. See data/processed and outputs/.")


if __name__ == "__main__":
    main()
 