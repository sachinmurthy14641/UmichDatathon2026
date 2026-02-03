# Creates your “implicit incentive signals” (shares, per-capita, rolling exposures, HHI/entropy).

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy

from .config import TAX_BUCKETS, DEFAULT_BUCKETS_ORDER
from .io import assert_required_columns


def add_tax_buckets(tax: pd.DataFrame) -> pd.DataFrame:
    df = tax.copy()
    df["bucket"] = df["tax_code"].map(TAX_BUCKETS).fillna("other")
    return df


def build_tax_signals(
    tax_revenue: pd.DataFrame,
    panel_state_quarter: pd.DataFrame,
    rolling_q: int = 12,   # 3 years of quarters
    lag_q: int = 8,        # 2 years lag
) -> pd.DataFrame:
    """
    Output: one row per (state, period) with bucket shares, HHI/entropy, rolling exposure (lagged).
    """
    tax = tax_revenue.copy()
    panel = panel_state_quarter.copy()

    assert_required_columns(tax, ["state", "period", "amount", "tax_code"], "tax_revenue")
    assert_required_columns(panel, ["state", "period", "population", "gdp_total"], "panel_state_quarter")

    tax = add_tax_buckets(tax)

    # total tax by state-period (for shares)
    totals = tax.groupby(["state", "period"], as_index=False)["amount"].sum().rename(columns={"amount": "tax_total"})
    tax = tax.merge(totals, on=["state", "period"], how="left")
    tax["tax_share"] = np.where(tax["tax_total"] > 0, tax["amount"] / tax["tax_total"], np.nan)

    # Bucket share by state-period
    bucket_share = (
        tax.groupby(["state", "period", "bucket"], as_index=False)["tax_share"].sum()
        .pivot(index=["state", "period"], columns="bucket", values="tax_share")
        .fillna(0.0)
        .reset_index()
    )

    # Ensure consistent bucket columns
    for b in DEFAULT_BUCKETS_ORDER + ["other"]:
        if b not in bucket_share.columns:
            bucket_share[b] = 0.0

    # HHI & entropy of tax mix at tax_code level
    def _hhi(group: pd.DataFrame) -> float:
        s = group["tax_share"].fillna(0.0).to_numpy()
        return float(np.sum(s * s))

    def _entropy(group: pd.DataFrame) -> float:
        s = group["tax_share"].fillna(0.0).to_numpy()
        s = s[s > 0]
        return float(entropy(s)) if len(s) else 0.0

    mix_stats = tax.groupby(["state", "period"]).apply(
        lambda g: pd.Series({"tax_hhi": _hhi(g), "tax_entropy": _entropy(g)})
    ).reset_index()

    # Merge with panel for per-capita and per-GDP options if desired later
    signals = panel.merge(bucket_share, on=["state", "period"], how="left").merge(mix_stats, on=["state", "period"], how="left")

    # Rolling exposure (smoothed bucket shares) + lag
    signals = signals.sort_values(["state", "period"]).reset_index(drop=True)

    # rolling by state for each bucket column
    bucket_cols = [c for c in signals.columns if c in (DEFAULT_BUCKETS_ORDER + ["other"])]
    for c in bucket_cols:
        roll = (
            signals.groupby("state")[c]
            .rolling(window=rolling_q, min_periods=max(4, rolling_q // 3))
            .mean()
            .reset_index(level=0, drop=True)
        )
        signals[f"{c}_roll{rolling_q}"] = roll
        # lagged rolling exposure
        signals[f"{c}_roll{rolling_q}_lag{lag_q}"] = signals.groupby("state")[f"{c}_roll{rolling_q}"].shift(lag_q)

    # Keep a clean subset (easy for teammates)
    keep = ["state", "period", "population", "gdp_total", "unemployment_rate", "tax_hhi", "tax_entropy"]
    keep += bucket_cols
    keep += [f"{c}_roll{rolling_q}_lag{lag_q}" for c in bucket_cols]

    out = signals[keep].copy()
    out = out.sort_values(["state", "period"]).reset_index(drop=True)
    return out
