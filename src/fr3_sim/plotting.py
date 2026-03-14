"""Plotting utilities.

Keeps matplotlib logic out of the experiment runner.

This module must NOT:
- Run simulations
- Modify configs
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_sweep(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Sequence[str],
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    dpi: int = 200,
    file_format: str = "png",
) -> Path:
    """Plot one or more y-columns against an x-column."""

    if df.empty:
        raise ValueError("Empty DataFrame; nothing to plot.")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = df[x_col].to_numpy()
    for y in y_cols:
        if y not in df.columns:
            continue
        ax.plot(x, df[y].to_numpy(), marker="o", label=y)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    if len(y_cols) > 1:
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = out_path.with_suffix(f".{file_format}")
    fig.savefig(final_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return final_path
