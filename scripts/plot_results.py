"""Generate plots from a finished experiment directory.

Example:
  python scripts/plot_results.py --results_dir results/baseline_wmmse_20260115_123000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fr3_sim.plotting import plot_sweep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FR3 WMMSE experiment results")
    p.add_argument("--results_dir", type=str, required=True, help="Path to a results/<exp_*/> directory")
    p.add_argument("--x_col", type=str, default="sweep_value", help="x column in metrics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rdir = Path(args.results_dir)
    metrics_csv = rdir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    figs_dir = rdir / "figures"
    figs_dir.mkdir(exist_ok=True, parents=True)

    plot_sweep(
        df,
        x_col=args.x_col,
        y_cols=[
            "sum_rate_bps_per_hz",
            "max_bs_power_violation_watt",
            "max_fs_violation_watt",
        ],
        title=f"FR3 WMMSE baseline: {rdir.name}",
        out_path=figs_dir / "summary.png",
    )

    print("Saved figures to", figs_dir)


if __name__ == "__main__":
    main()
