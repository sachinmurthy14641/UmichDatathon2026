from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assert_required_columns


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    return out


def _ensure_state_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "state" not in out.columns:
        for cand in ["state_name", "state_code"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "state"})
                break

    if "period" not in out.columns:
        if "year" in out.columns and "quarter" in out.columns:
            q = out["quarter"].astype(str).str.upper().str.replace("Q", "", regex=False)
            out["period"] = out["year"].astype(int).astype(str) + "Q" + q

    return out


def build_outcomes(
    panel_state_quarter: pd.DataFrame,
    demographics: pd.DataFrame,
    economics: pd.DataFrame,
    tax_signals: pd.DataFrame,
    drift_lag_q: int = 8,
) -> pd.DataFrame:
    """
    Outputs: one row per (state, period) with structural outcomes (levels + drift).

    Adds:
      - GDP per-capita + drift
      - Demographic structure + drift (dependency ratio, cohort shares)
      - Economic structure (sector HHI/entropy) + drift
      - Knowledge-economy GDP share + drift
      - Business dynamism (formations/applications) + drift
      - Fiscal adaptation (tax diversification + dependency ratios) + drift   <-- NEW
    """

    # -----------------------
    # PANEL (base)
    # -----------------------
    panel = _ensure_state_period(_normalize_columns(panel_state_quarter))
    assert_required_columns(panel, ["state", "period", "population", "gdp_total"], "panel_state_quarter")
    out = panel.copy()

    # -----------------------
    # DEMOGRAPHICS (optional merge)
    # -----------------------
    demo = _ensure_state_period(_normalize_columns(demographics))
    if all(c in demo.columns for c in ["state", "period"]):
        demo_cols = ["state", "period"]
        for c in ["pop_youth", "pop_working", "pop_senior", "age_median"]:
            if c in demo.columns:
                demo_cols.append(c)
        demo = demo[demo_cols].copy()
        out = out.merge(demo, on=["state", "period"], how="left", suffixes=("", "_demo"))

    # -----------------------
    # ECONOMICS (sector GDP + biz dynamism)
    # -----------------------
    econ = _ensure_state_period(_normalize_columns(economics))
    if not all(c in econ.columns for c in ["state", "period"]):
        econ = None

    sector_cols = [
        "gdp_ag_fish_hunt",
        "gdp_mining_oil_gas",
        "gdp_utilities",
        "gdp_construction",
        "gdp_manufacturing",
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
        "gdp_gov_total",
    ]

    econ_base_cols = [
        "state",
        "period",
        "gdp_total",
        "gdp_private_total",
        "gdp_gov_total",
        "applications",
        "formations",
    ]

    if econ is not None:
        present_sector_cols = [c for c in sector_cols if c in econ.columns]
        econ_cols = [c for c in econ_base_cols if c in econ.columns] + present_sector_cols
        econ = econ[econ_cols].copy()

        out = out.merge(econ, on=["state", "period"], how="left", suffixes=("", "_econ"))
        out = out.loc[:, ~out.columns.duplicated()].copy()

    # -----------------------
    # TAX SIGNALS (NEW fiscal outcomes)
    # -----------------------
    tax = _ensure_state_period(_normalize_columns(tax_signals))
    assert_required_columns(tax, ["state", "period", "tax_total"], "tax_signals")

    # We only need a small subset for outcomes
    tax_bucket_cols = [
        "corporate_income",
        "general_sales",
        "labor_income",
        "other",
        "property",
        "resource_severance",
        "selective_excise",
    ]

    tax_keep = ["state", "period", "tax_total", "tax_hhi", "tax_entropy"]
    tax_keep += [c for c in tax_bucket_cols if c in tax.columns]
    tax = tax[tax_keep].copy()

    out = out.merge(tax, on=["state", "period"], how="left", suffixes=("", "_tax"))
    out = out.loc[:, ~out.columns.duplicated()].copy()

    # -----------------------
    # GDP shares (private/gov)
    # -----------------------
    if all(c in out.columns for c in ["gdp_total", "gdp_private_total"]):
        out["gdp_private_share"] = out["gdp_private_total"] / out["gdp_total"].replace({0: np.nan})

    if all(c in out.columns for c in ["gdp_total", "gdp_gov_total"]):
        out["gdp_gov_share"] = out["gdp_gov_total"] / out["gdp_total"].replace({0: np.nan})

    # -----------------------
    # Economic structure: sector shares -> HHI, entropy
    # -----------------------
    if ("gdp_total" in out.columns) and (econ is not None):
        present_sector_cols = [c for c in sector_cols if c in out.columns]
        if len(present_sector_cols) >= 2:
            total = out["gdp_total"].replace({0: np.nan})
            shares = out[present_sector_cols].div(total, axis=0)
            shares = shares.where(shares > 0)

            out["econ_hhi"] = (shares ** 2).sum(axis=1, skipna=True)

            K = len(present_sector_cols)
            entropy = -(shares * np.log(shares)).sum(axis=1, skipna=True)
            out["econ_entropy_norm"] = entropy / np.log(K) if K > 1 else np.nan

    # -----------------------
    # Knowledge economy share + drift
    # -----------------------
    knowledge_cols = ["gdp_information", "gdp_prof_tech_services", "gdp_finance_insurance"]
    if ("gdp_total" in out.columns) and any(c in out.columns for c in knowledge_cols):
        numer = 0.0
        has_any = False
        for c in knowledge_cols:
            if c in out.columns:
                numer = numer + out[c].fillna(0)
                has_any = True
        if has_any:
            out["gdp_knowledge_share"] = numer / out["gdp_total"].replace({0: np.nan})

    # -----------------------
    # Demographics: dependency ratio & shares
    # -----------------------
    if all(c in out.columns for c in ["pop_youth", "pop_working", "pop_senior", "population"]):
        out["dependency_ratio"] = (out["pop_youth"] + out["pop_senior"]) / out["pop_working"].replace({0: np.nan})
        out["senior_share"] = out["pop_senior"] / out["population"].replace({0: np.nan})
        out["working_share"] = out["pop_working"] / out["population"].replace({0: np.nan})

    # -----------------------
    # Business dynamism outcomes + drift
    # -----------------------
    if all(c in out.columns for c in ["formations", "population"]):
        out["business_formation_rate_per_1k"] = out["formations"] / out["population"].replace({0: np.nan}) * 1000.0

    if all(c in out.columns for c in ["applications", "formations"]):
        out["applications_to_formations_ratio"] = out["applications"] / out["formations"].replace({0: np.nan})

    # -----------------------
    # Fiscal adaptation outcomes (NEW)
    # -----------------------
    # Dependency ratios (shares of tax_total)
    if all(c in out.columns for c in ["tax_total", "general_sales"]):
        out["sales_tax_dependency"] = out["general_sales"] / out["tax_total"].replace({0: np.nan})

    if all(c in out.columns for c in ["tax_total", "labor_income", "corporate_income"]):
        out["income_tax_dependency"] = (out["labor_income"] + out["corporate_income"]) / out["tax_total"].replace(
            {0: np.nan}
        )

    if all(c in out.columns for c in ["tax_total", "property"]):
        out["property_tax_dependency"] = out["property"] / out["tax_total"].replace({0: np.nan})

    if all(c in out.columns for c in ["tax_total", "resource_severance"]):
        out["resource_severance_dependency"] = out["resource_severance"] / out["tax_total"].replace({0: np.nan})

    # Normalized tax entropy across major buckets (robust even if some buckets missing)
    present_tax_buckets = [c for c in tax_bucket_cols if c in out.columns]
    if ("tax_total" in out.columns) and (len(present_tax_buckets) >= 2):
        ttot = out["tax_total"].replace({0: np.nan})
        tshares = out[present_tax_buckets].div(ttot, axis=0)
        tshares = tshares.where(tshares > 0)

        Kt = len(present_tax_buckets)
        tentropy = -(tshares * np.log(tshares)).sum(axis=1, skipna=True)
        out["tax_entropy_norm"] = tentropy / np.log(Kt) if Kt > 1 else np.nan

    # -----------------------
    # GDP per capita + drift
    # -----------------------
    out["gdp_per_capita"] = out["gdp_total"] / out["population"].replace({0: np.nan})

    # -----------------------
    # Drift measures (lagged changes)
    # -----------------------
    out = out.sort_values(["state", "period"]).reset_index(drop=True)

    drift_cols = [
        # core
        "gdp_per_capita",
        "dependency_ratio",
        "senior_share",
        "working_share",
        "econ_hhi",
        "econ_entropy_norm",
        "gdp_knowledge_share",
        "business_formation_rate_per_1k",
        "applications_to_formations_ratio",
        # fiscal (new)
        "tax_hhi",
        "tax_entropy_norm",
        "sales_tax_dependency",
        "income_tax_dependency",
        "property_tax_dependency",
        "resource_severance_dependency",
    ]

    for c in drift_cols:
        if c in out.columns:
            out[f"delta_{c}_lag{drift_lag_q}"] = out.groupby("state")[c].diff(drift_lag_q)

    # -----------------------
    # Keep output schema
    # -----------------------
    keep = ["state", "period"]

    desired = [
        "gdp_per_capita",
        f"delta_gdp_per_capita_lag{drift_lag_q}",
        "dependency_ratio",
        "senior_share",
        "working_share",
        f"delta_dependency_ratio_lag{drift_lag_q}",
        f"delta_senior_share_lag{drift_lag_q}",
        f"delta_working_share_lag{drift_lag_q}",
        "econ_hhi",
        "econ_entropy_norm",
        f"delta_econ_hhi_lag{drift_lag_q}",
        f"delta_econ_entropy_norm_lag{drift_lag_q}",
        "gdp_private_share",
        "gdp_gov_share",
        "gdp_knowledge_share",
        f"delta_gdp_knowledge_share_lag{drift_lag_q}",
        "business_formation_rate_per_1k",
        "applications_to_formations_ratio",
        f"delta_business_formation_rate_per_1k_lag{drift_lag_q}",
        f"delta_applications_to_formations_ratio_lag{drift_lag_q}",
        # fiscal adaptation
        "tax_hhi",
        "tax_entropy_norm",
        f"delta_tax_hhi_lag{drift_lag_q}",
        f"delta_tax_entropy_norm_lag{drift_lag_q}",
        "sales_tax_dependency",
        "income_tax_dependency",
        "property_tax_dependency",
        "resource_severance_dependency",
        f"delta_sales_tax_dependency_lag{drift_lag_q}",
        f"delta_income_tax_dependency_lag{drift_lag_q}",
        f"delta_property_tax_dependency_lag{drift_lag_q}",
        f"delta_resource_severance_dependency_lag{drift_lag_q}",
    ]

    for c in desired:
        if c in out.columns:
            keep.append(c)

    final = out[keep].copy().sort_values(["state", "period"]).reset_index(drop=True)
    return final
