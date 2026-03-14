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
from fr3_twc.config_utils import results_root_dir
from fr3_twc.figures import generate_all_figures
from fr3_twc.wideband_channel import build_wideband_batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twc_full.yaml")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    root_dir = results_root_dir(cfg)
    pipe_dir = root_dir / "twc_pipeline"

    summary_path = pipe_dir / "metrics_summary_with_legacy.csv"
    if not summary_path.exists():
        summary_path = pipe_dir / "metrics_summary.csv"

    summary_df = pd.read_csv(summary_path)
    per_sample_df = pd.read_csv(pipe_dir / "metrics_per_sample.csv")
    history_path = pipe_dir / "algorithm_history.csv"
    tone_path = pipe_dir / "tone_grouping_error.csv"
    weight_path = pipe_dir / "user_weight_sensitivity.csv"

    history_df = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    tone_df = pd.read_csv(tone_path) if tone_path.exists() else pd.DataFrame()
    weight_df = pd.read_csv(weight_path) if weight_path.exists() else pd.DataFrame()

    batch_reference = build_wideband_batch(cfg, batch_size=1, user_weight_profile="uniform")
    generate_all_figures(
        summary_df=summary_df,
        per_sample_df=per_sample_df,
        history_df=history_df,
        tone_df=tone_df,
        weight_df=weight_df,
        batch_reference=batch_reference,
        fig_dir=pipe_dir / "figures",
    )
    print(pipe_dir / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
