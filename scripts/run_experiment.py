"""Run a simulation experiment from the command line.

Example:
  python scripts/run_experiment.py --config configs/default.yaml

Overrides (dot notation):
  python scripts/run_experiment.py --config configs/default.yaml \
      --set sweep.enabled=true --set sweep.values='[0,5,10]'
"""

from __future__ import annotations

import argparse

from fr3_sim.config import load_config
from fr3_sim.runner import run_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FR3 WMMSE baseline simulator (no deep unfolding)")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=None,
        help="Override config value, e.g., 'sweep.values=[0,5,10]'",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    results_dir = run_experiment(cfg)
    print("\nResults saved to:", results_dir)


if __name__ == "__main__":
    main()
