#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fr3_sim.config import load_config
from fr3_twc.config_utils import results_root_dir, twc_get


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twc_full.yaml")
    ap.add_argument("--require-legacy", action="store_true")
    ap.add_argument("--min-figure-count", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    root_dir = results_root_dir(cfg)
    pipe_dir = root_dir / "twc_pipeline"
    fig_dir = pipe_dir / "figures"

    required = [
        pipe_dir / "run_metadata.json",
        pipe_dir / "metrics_summary.csv",
        pipe_dir / "metrics_summary_with_legacy.csv",
        pipe_dir / "metrics_per_sample.csv",
        pipe_dir / "tone_grouping_error.csv",
        pipe_dir / "user_weight_sensitivity.csv",
        pipe_dir / "algorithm_history.csv",
    ]
    if bool(twc_get(cfg, ("unfolded", "enabled"), True)):
        required.append(pipe_dir / "training_history.csv")

    for path in required:
        _require(path)

    summary_df = pd.read_csv(pipe_dir / "metrics_summary.csv")
    summary_all_df = pd.read_csv(pipe_dir / "metrics_summary_with_legacy.csv")
    if summary_df.empty:
        raise RuntimeError("metrics_summary.csv is empty")

    algorithms = set(summary_df["algorithm"].astype(str))
    expected = {"static_notch_mf", "risk_neutral_pgd", "wideband_pgd"}
    if bool(twc_get(cfg, ("unfolded", "enabled"), True)):
        expected.add("scenario_adaptive_unfolded")
    missing_algorithms = sorted(expected - algorithms)
    if missing_algorithms:
        raise RuntimeError(f"Missing algorithms in metrics_summary.csv: {missing_algorithms}")

    if not (summary_df["sweep_tag"].astype(str) == "fs_in_target_db").any():
        raise RuntimeError("metrics_summary.csv does not contain fs_in_target_db sweep rows")

    if bool(twc_get(cfg, ("unfolded", "enabled"), True)):
        train_df = pd.read_csv(pipe_dir / "training_history.csv")
        if train_df.empty:
            raise RuntimeError("training_history.csv is empty")

    _require(fig_dir)
    figure_files = [p for p in fig_dir.iterdir() if p.is_file()]
    if len(figure_files) < int(args.min_figure_count):
        raise RuntimeError(f"Expected at least {args.min_figure_count} figure files, found {len(figure_files)}")

    if args.require_legacy:
        legacy_algorithms = {a for a in summary_all_df["algorithm"].astype(str) if a.startswith("legacy_")}
        required_legacy = {"legacy_budget_dual", "legacy_hard_null", "legacy_no_fs"}
        missing_legacy = sorted(required_legacy - legacy_algorithms)
        if missing_legacy:
            raise RuntimeError(f"Missing legacy algorithms in metrics_summary_with_legacy.csv: {missing_legacy}")

    print(f"Verified TWC outputs: {pipe_dir}")
    print("Algorithms:", ", ".join(sorted(algorithms)))
    legacy_found = sorted({a for a in summary_all_df["algorithm"].astype(str) if a.startswith("legacy_")})
    if legacy_found:
        print("Legacy algorithms:", ", ".join(legacy_found))
    print(f"Figure files: {len(figure_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
