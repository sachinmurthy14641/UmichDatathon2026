### A simple fixed-effects style regression using statsmodels
### with state/time dummies (fast + dependency-light).

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def fit_fe_ols(
    df: pd.DataFrame,
    y: str,
    x: str,
    controls: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fixed effects (state + period) via categorical dummies.
    Returns a tidy coefficient table.
    """
    controls = controls or []
    cols_needed = ["state", "period", y, x] + controls
    data = df[cols_needed].dropna().copy()

    # state and period fixed effects via C()
    rhs = " + ".join([x] + controls + ["C(state)", "C(period)"])
    formula = f"{y} ~ {rhs}"

    model = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["state"]})

    # tidy-ish table
    summary = (
        pd.DataFrame(
            {
                "term": model.params.index,
                "coef": model.params.values,
                "std_err": model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
            }
        )
        .sort_values("p_value")
        .reset_index(drop=True)
    )
    return summary
