### visualizations

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    out_path: Path,
    max_groups: int = 8,
) -> None:
    """
    Simple multi-line plot for quick story assets.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()

    # pick top groups by data coverage
    counts = df.groupby(group)[y].count().sort_values(ascending=False)
    groups = list(counts.head(max_groups).index)

    for g in groups:
        sub = df[df[group] == g].sort_values(x)
        plt.plot(sub[x], sub[y], label=str(g))

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
