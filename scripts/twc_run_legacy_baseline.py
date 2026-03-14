#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fr3_sim.config import load_config
from fr3_sim.runner import run_experiment


def run_legacy_suite(config_path: str | Path) -> None:
    config_path = Path(config_path)
    base_cfg = load_config(str(config_path))

    base_exp = str(base_cfg.raw["output"]["experiment_name"])
    common = [
        f"output.results_root={str(Path(base_cfg.raw['output']['results_root']) / base_exp / 'legacy_baseline')}",
        "sweep.enabled=true",
        "sweep.variable=fs_in_target_db",
        "sweep.values=[-6.0,-10.0,-14.0]",
        "experiment.freeze_topology=true",
    ]

    variants = {
        "legacy_budget_dual": [
            "receiver.wmmse.fs_enforcement=budget_dual",
            "receiver.wmmse.fs_lambda_search=true",
        ],
        "legacy_hard_null": [
            "receiver.wmmse.fs_enforcement=hard_null",
            "receiver.wmmse.fs_lambda_search=false",
        ],
        "legacy_no_fs": [
            "receiver.wmmse.fs_enforcement=none",
            "receiver.wmmse.fs_lambda_search=false",
        ],
    }

    for name, overrides in variants.items():
        ov = common + [f"output.experiment_name={name}"] + overrides
        cfg = load_config(str(config_path), overrides=ov)
        print(f"Running {name} ...")
        out_dir = run_experiment(cfg)
        print(f"Finished {name}: {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twc_legacy_baseline.yaml")
    args = ap.parse_args()

    run_legacy_suite(REPO_ROOT / args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
