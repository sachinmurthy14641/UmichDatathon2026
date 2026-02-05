# Creates your “implicit incentive signals” (shares, per-capita, per-GDP, rolling exposures, HHI/entropy).

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .config import TAX_BUCKETS, DEFAULT_BUCKETS_ORDER
from .io import assert_required_columns


# IMPORTANT:
# We confirmed the tax revenue "amount" in the dataset is in THOUSANDS of dollars.
# We also confirmed that the GDP amounts in the dataset are in MILLIONS of dollars,
# and they are annualized rates (SAAR) per quarter.

TAX_DOLLARS_PER_UNIT = 1_000       # tax amounts are in $1,000s (confirmed)
GDP_DOLLARS_PER_UNIT = 1_000_000   # GDP appears in $ millions
GDP_IS_SAAR = True                # treat quarterly GDP as annualized rate


def add_tax_buckets(tax: pd.DataFrame) -> pd.DataFrame:
    df = tax.copy()
    df["bucket"] = df["tax_code"].map(TAX_BUCKETS).fillna("other")
    return df


def build_tax_signals(
    tax_revenue: pd.DataFrame,
    panel_state_quarter: pd.DataFrame,
    rolling_q: int = 12,   # 3 years of quarters (kept as default to match scaffold)
    lag_q: int = 8,        # 2 years lag (kept as default to match scaffold)
) -> pd.DataFrame:
    """
    Output: one row per (state, period) with:
      - bucket shares (baseline)
      - per-capita bucket levels
      - bucket levels as share of GDP (pct_gdp)
      - HHI/entropy (tax_code-level mix stats)
      - rolling exposure (lagged) for baseline shares (both lag8 and lag4)
    """
    tax = tax_revenue.copy()
    panel = panel_state_quarter.copy()

    assert_required_columns(tax, ["state", "period", "amount", "tax_code"], "tax_revenue")
    assert_required_columns(panel, ["state", "period", "population", "gdp_total"], "panel_state_quarter")

    tax = add_tax_buckets(tax)

    # total tax by state-period (for shares + per-capita + pct_gdp totals)
    totals = (
        tax.groupby(["state", "period"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "tax_total"})
    )
    tax = tax.merge(totals, on=["state", "period"], how="left")
    tax["tax_share"] = np.where(tax["tax_total"] > 0, tax["amount"] / tax["tax_total"], np.nan)

    # Bucket share by state-period (baseline signals)
    bucket_share = (
        tax.groupby(["state", "period", "bucket"], as_index=False)["tax_share"].sum()
        .pivot(index=["state", "period"], columns="bucket", values="tax_share")
        .fillna(0.0)
        .reset_index()
    )

    # Bucket amounts by state-period (for per-capita and pct_gdp)
    bucket_amount = (
        tax.groupby(["state", "period", "bucket"], as_index=False)["amount"].sum()
        .pivot(index=["state", "period"], columns="bucket", values="amount")
        .fillna(0.0)
        .reset_index()
    )

    # Ensure consistent bucket columns (both shares + amounts)
    for b in DEFAULT_BUCKETS_ORDER + ["other"]:
        if b not in bucket_share.columns:
            bucket_share[b] = 0.0
        if b not in bucket_amount.columns:
            bucket_amount[b] = 0.0

    # Rename amount columns to avoid collisions with share columns
    amt_rename = {b: f"{b}_amt" for b in (DEFAULT_BUCKETS_ORDER + ["other"])}
    bucket_amount = bucket_amount.rename(columns=amt_rename)

    # HHI & entropy of tax mix at tax_code level
    def _hhi(group: pd.DataFrame) -> float:
        s = group["tax_share"].fillna(0.0).to_numpy()
        return float(np.sum(s * s))

    def _entropy(group: pd.DataFrame) -> float:
        s = group["tax_share"].fillna(0.0).to_numpy()
        s = s[s > 0]
        return float(entropy(s)) if len(s) else 0.0

    mix_stats = (
        tax.groupby(["state", "period"])
        .apply(lambda g: pd.Series({"tax_hhi": _hhi(g), "tax_entropy": _entropy(g)}))
        .reset_index()
    )

    # Merge everything onto the panel
    signals = (
        panel.merge(bucket_share, on=["state", "period"], how="left")
        .merge(bucket_amount, on=["state", "period"], how="left")
        .merge(mix_stats, on=["state", "period"], how="left")
        .merge(totals, on=["state", "period"], how="left")
    )

    # --- Per-capita & pct-GDP constructions ---
    pop = signals["population"].replace({0: np.nan})

    gdp_dollars = signals["gdp_total"].replace({0: np.nan}) * GDP_DOLLARS_PER_UNIT  # GDP in $ (millions -> dollars)
    if GDP_IS_SAAR:
        gdp_dollars = gdp_dollars / 4.0  # convert SAAR to quarterly GDP dollars

    # Total tax per-capita (dollars/person)
    signals["tax_total_per_capita"] = (signals["tax_total"] * TAX_DOLLARS_PER_UNIT) / pop

    # Total tax as share of GDP (ratio)
    signals["tax_total_pct_gdp"] = (signals["tax_total"] * TAX_DOLLARS_PER_UNIT) / gdp_dollars

    bucket_cols = [c for c in signals.columns if c in (DEFAULT_BUCKETS_ORDER + ["other"])]
    for c in bucket_cols:
        amt_col = f"{c}_amt"
        signals[f"{c}_per_capita"] = (signals[amt_col] * TAX_DOLLARS_PER_UNIT) / pop
        signals[f"{c}_pct_gdp"] = (signals[amt_col] * TAX_DOLLARS_PER_UNIT) / gdp_dollars

    # --- Rolling exposure (smoothed bucket shares) + lags (baseline shares only) ---
    signals = signals.sort_values(["state", "period"]).reset_index(drop=True)

    # Existing scaffold: roll12_lag8. Add: roll12_lag4.
    lag4 = 4

    for c in bucket_cols:
        roll = (
            signals.groupby("state")[c]
            .rolling(window=rolling_q, min_periods=max(4, rolling_q // 3))
            .mean()
            .reset_index(level=0, drop=True)
        )
        signals[f"{c}_roll{rolling_q}"] = roll

        # lagged rolling exposures
        signals[f"{c}_roll{rolling_q}_lag{lag_q}"] = signals.groupby("state")[f"{c}_roll{rolling_q}"].shift(lag_q)
        signals[f"{c}_roll{rolling_q}_lag{lag4}"] = signals.groupby("state")[f"{c}_roll{rolling_q}"].shift(lag4)

    # Keep a clean subset (easy for teammates)
    keep = [
        "state",
        "period",
        "population",
        "gdp_total",
        "unemployment_rate",
        "tax_total",
        "tax_total_per_capita",
        "tax_total_pct_gdp",
        "tax_hhi",
        "tax_entropy",
    ]
    keep += bucket_cols
    keep += [f"{c}_per_capita" for c in bucket_cols]
    keep += [f"{c}_pct_gdp" for c in bucket_cols]
    keep += [f"{c}_roll{rolling_q}_lag{lag_q}" for c in bucket_cols]
    keep += [f"{c}_roll{rolling_q}_lag{lag4}" for c in bucket_cols]

    out = signals[keep].copy()
    out = out.sort_values(["state", "period"]).reset_index(drop=True)
    return out
