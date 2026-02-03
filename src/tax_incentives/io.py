### handles column-name normalization

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, List

from .config import (
    TAX_REQUIRED_CANONICAL,
    DEMO_REQUIRED_CANONICAL,
    ECON_REQUIRED_CANONICAL,
)


def assert_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("%", "pct")
    )
    return df


def _apply_column_map(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=col_map)


def _coerce_quarter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["year"].astype(int)

    if df["quarter"].dtype == object:
        df["quarter"] = df["quarter"].astype(str).str.upper().str.replace("Q", "")
    df["quarter"] = df["quarter"].astype(int)

    df["period"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
    return df


def load_tax_revenue_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)

    # Your file 1 columns:
    # Year, Quarter, State, Tax_Category, Tax_Code, Amount
    col_map = {
        "year": "year",
        "quarter": "quarter",
        "state": "state",
        "tax_category": "tax_category",
        "tax_code": "tax_code",
        "amount": "amount",
    }
    df = _apply_column_map(df, col_map)

    df["state"] = df["state"].astype(str).str.strip()
    df["tax_code"] = df["tax_code"].astype(str).str.strip().str.upper()
    df["tax_category"] = df["tax_category"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    df = _coerce_quarter(df)

    assert_required_columns(df, TAX_REQUIRED_CANONICAL + ["period"], "tax_revenue")
    return df


def load_economics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _standardize_columns(df)

    # Your file 2 columns include:
    # State, Year, Quarter, Population, Unemployment_Rate, GDP_Total, ...
    col_map = {
        "state": "state",
        "year": "year",
        "quarter": "quarter",
        "population": "population",
        "unemployment_rate": "unemployment_rate",
        "gdp_total": "gdp_total",
        # keep others as-is (already standardized)
    }
    df = _apply_column_map(df, col_map)

    df["state"] = df["state"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["gdp_total"] = pd.to_numeric(df["gdp_total"], errors="coerce")
    df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")

    df = _coerce_quarter(df)

    assert_required_columns(df, ECON_REQUIRED_CANONICAL + ["period"], "economics")
    return df


def load_demographics_csv_annual_expand_to_quarterly(path: Path) -> pd.DataFrame:
    """
    File 3 is ANNUAL by state-year (no quarter).
    We expand each year into 4 quarters (Q1-Q4) with the same annual values.
    This is a speed-first approach and is acceptable for scaffolding.
    """
    df = pd.read_csv(path)
    df = _standardize_columns(df)

    # Your file 3 columns include:
    # State, Year, Total Population, ..., Pop_Youth, Pop_Working, Pop_Senior, Age_Median, Population, ...
    col_map = {
        "state": "state",
        "year": "year",
        # prefer quarterly-merge-friendly canonical names:
        "population": "population",
        "total_population": "total_population",
        "pop_youth": "pop_youth",
        "pop_working": "pop_working",
        "pop_senior": "pop_senior",
        "age_median": "age_median",
        # keep the rest (race/ethnicity, immigration) as standardized names
    }
    df = _apply_column_map(df, col_map)

    df["state"] = df["state"].astype(str).str.strip()
    df["year"] = df["year"].astype(int)

    # Make sure population exists (some files have both total_population and population)
    if "population" not in df.columns and "total_population" in df.columns:
        df["population"] = df["total_population"]

    # Coerce numeric where relevant (safe to attempt)
    for c in ["population", "pop_youth", "pop_working", "pop_senior", "age_median"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Expand to quarters
    expanded = []
    for q in [1, 2, 3, 4]:
        tmp = df.copy()
        tmp["quarter"] = q
        tmp["period"] = tmp["year"].astype(str) + "Q" + str(q)
        expanded.append(tmp)

    out = pd.concat(expanded, ignore_index=True)

    # Minimum required for merge/pipeline
    assert_required_columns(out, ["state", "year", "quarter", "period", "population"], "demographics_annual_expanded")
    return out
