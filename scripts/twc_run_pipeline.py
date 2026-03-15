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
from fr3_twc.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twc_full.yaml")
    args = ap.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    out_dir = run_pipeline(cfg)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
