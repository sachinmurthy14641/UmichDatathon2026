# step5_validation.py
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tax_incentives.models_panel import fit_fe_ols
from tax_incentives.io import write_csv


def run_step5(design: pd.DataFrame, panel: pd.DataFrame, paths) -> None:
    """
    Step 5: Validation & Synthesis
    1. Multi-outcome FE regressions
    2. Cohort/event line plots
    3. Coefficient heatmap
    4. Directionality/significance table
    """

    # ---- 1. FE regressions ----
    outcomes = [
        "delta_gdp_per_capita_lag8",
        "delta_dependency_ratio_lag8",
        "delta_senior_share_lag8",
        "delta_working_share_lag8",
        "econ_hhi",
        "econ_entropy_norm",
    ]
    exposures = [col for col in design.columns if col.endswith("_roll12_lag8")]
    controls = ["unemployment_rate"]

    all_results = []
    for y in outcomes:
        for x in exposures:
            if y in design.columns and x in design.columns:
                coef_table = fit_fe_ols(design, y=y, x=x, controls=controls)
                coef_table["outcome"] = y
                coef_table["exposure"] = x
                all_results.append(coef_table)

    summary_df = pd.concat(all_results, ignore_index=True)
    write_csv(summary_df, paths.outputs_tables / "panel_model_summary_all.csv")
    print("FE regression summary saved.")

    # ---- 2. Cohort/event line plots ----
    cohort_col = "cohort_high_exposure"
    event_cols = [c for c in design.columns if c.startswith("event_")]

    for outcome in outcomes:
        plt.figure(figsize=(8, 5))
        for cohort in design[cohort_col].unique():
            cohort_mask = design[cohort_col] == cohort
            mean_series = (
                design[cohort_mask].groupby("period")[outcome].mean()
            )
            plt.plot(mean_series.index, mean_series.values, label=f"{cohort}")
        plt.title(f"{outcome} by Cohort Over Time")
        plt.xlabel("Period")
        plt.ylabel(outcome)
        plt.legend()
        plt.tight_layout()
        out_path = paths.outputs_figures / f"{outcome}_cohort_plot.png"
        plt.savefig(out_path)
        plt.close()
    print("Cohort/event line plots saved.")

    # ---- 3. Regression coefficient heatmap ----
    # Pivot summary for heatmap: outcomes x exposures
    coef_pivot = summary_df.pivot_table(
        index="outcome",
        columns="exposure",
        values="coef",
        aggfunc="first"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(coef_pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("FE Regression Coefficients (Outcome ~ Exposure)")
    plt.tight_layout()
    heatmap_path = paths.outputs_figures / "coef_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    print("Coefficient heatmap saved.")

    # ---- 4. Directionality & significance table ----
    summary_df["directionally_consistent"] = summary_df["coef"] > 0
    summary_df["significant"] = summary_df["p_value"] < 0.05

    dir_table = summary_df.groupby(["outcome", "exposure"]).agg(
        coef=("coef", "first"),
        p_value=("p_value", "first"),
        directionally_consistent=("directionally_consistent", "first"),
        significant=("significant", "first")
    ).reset_index()

    write_csv(dir_table, paths.outputs_tables / "directionality_significance.csv")
    print("Directionality/significance table saved.")
