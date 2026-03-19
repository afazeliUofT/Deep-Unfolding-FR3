#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TARGET_ALGS = [
    "rate_recovered_primal_dual_unfolded",
    "budgeted_primal_dual_pgd_repair_recover",
    "budget_aware_primal_dual_unfolded",
    "legacy_budget_dual",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--pattern", default="twc_bcz_*_bundle/twc_pipeline/metrics_summary_with_legacy.csv")
    ap.add_argument("--out", default="results/twc_bcz_specialized_summary.csv")
    args = ap.parse_args()

    root = Path(args.results_root)
    rows = []
    for fp in sorted(root.glob(args.pattern)):
        bundle = fp.parts[-4]
        df = pd.read_csv(fp)
        sub = df[df["algorithm"].isin(TARGET_ALGS)].copy()
        if sub.empty:
            continue
        sub.insert(0, "bundle", bundle)
        rows.append(sub)
    if not rows:
        raise SystemExit("No matching specialized metrics files found.")

    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(args.out, index=False)

    focus = out_df[
        (out_df["bundle"].str.contains("_full_"))
        & (out_df["sweep_value"] == -10.0)
        & (out_df["algorithm"].isin(["rate_recovered_primal_dual_unfolded", "budgeted_primal_dual_pgd_repair_recover"]))
    ].copy()
    focus = focus.sort_values(
        ["algorithm", "fs_outage_prob_any", "sum_rate_bps_per_hz_mean", "runtime_ms_mean"],
        ascending=[True, True, False, True],
    )
    print("Full -10 dB specialized candidates:")
    print(focus[["bundle", "algorithm", "sum_rate_bps_per_hz_mean", "fs_outage_prob_any", "runtime_ms_mean", "worst_fs_excess_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
