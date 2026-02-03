from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    data_processed: Path
    outputs_figures: Path
    outputs_tables: Path


def get_paths(project_root: Path | None = None) -> Paths:
    root = project_root or Path(__file__).resolve().parents[2]
    return Paths(
        project_root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        outputs_figures=root / "outputs" / "figures",
        outputs_tables=root / "outputs" / "tables",
    )


# ---- Canonical column names (edit once, used everywhere) ----
# Tax revenue file expected columns:
# Year, Quarter, State, Amount, Tax Code, Tax Category  (names may vary; map in io.py)
TAX_REQUIRED_CANONICAL = ["year", "quarter", "state", "amount", "tax_code", "tax_category"]

# Demographics file expected columns (starter minimum):
DEMO_REQUIRED_CANONICAL = ["year", "quarter", "state", "population"]

# Economics file expected columns (starter minimum):
ECON_REQUIRED_CANONICAL = ["year", "quarter", "state", "gdp_total", "unemployment_rate"]


# ---- Tax bucket mapping (edit for your narrative) ----
# NOTE: these are “revealed signal buckets” (not statutory rates).
TAX_BUCKETS: Dict[str, str] = {
    "T40": "labor_income",
    "T41": "corporate_income",
    "T09": "general_sales",
    "T01": "property",
    "T53": "resource_severance",
    # add selective excises if you want:
    "T10": "selective_excise",
    "T13": "selective_excise",
    "T16": "selective_excise",
    "T19": "selective_excise",
    "T59": "selective_excise",
}

DEFAULT_BUCKETS_ORDER: List[str] = [
    "labor_income",
    "corporate_income",
    "general_sales",
    "property",
    "resource_severance",
    "selective_excise",
]
