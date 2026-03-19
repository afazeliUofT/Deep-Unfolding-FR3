#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            if k in merged:
                merged[k] = deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged
    return override


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", required=True, help="Path to override YAML")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    override_path = (repo_root / args.override).resolve() if not Path(args.override).is_absolute() else Path(args.override).resolve()
    if not override_path.exists():
        raise SystemExit(f"Override config not found: {override_path}")

    override_cfg = yaml.safe_load(override_path.read_text()) or {}
    base_config_rel = override_cfg.pop("base_config", None)
    if not base_config_rel:
        raise SystemExit("Override config must include 'base_config'.")

    base_path = (repo_root / base_config_rel).resolve() if not Path(base_config_rel).is_absolute() else Path(base_config_rel).resolve()
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")

    base_cfg = yaml.safe_load(base_path.read_text()) or {}
    merged = deep_merge(base_cfg, override_cfg)

    output_cfg = merged.setdefault("output", {})
    exp_name = output_cfg.get("experiment_name")
    if not exp_name:
        raise SystemExit("Merged config must define output.experiment_name")
    results_root = Path(output_cfg.get("results_root", "results"))
    pipe_dir = repo_root / results_root / exp_name / "twc_pipeline"
    if pipe_dir.exists():
        import shutil
        shutil.rmtree(pipe_dir)

    tmp_dir = repo_root / results_root / "_resolved_overrides"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{exp_name}.yaml"
    tmp_path.write_text(yaml.safe_dump(merged, sort_keys=False))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    steps = []
    if not args.skip_probe:
        steps.append([sys.executable, "scripts/twc_probe_env.py", "--strict"])
    steps.extend([
        [sys.executable, "scripts/twc_run_legacy_baseline.py", "--config", str(tmp_path.relative_to(repo_root))],
        [sys.executable, "scripts/twc_run_pipeline.py", "--config", str(tmp_path.relative_to(repo_root))],
        [sys.executable, "scripts/twc_plot_figures.py", "--config", str(tmp_path.relative_to(repo_root))],
        [sys.executable, "scripts/twc_verify_outputs.py", "--config", str(tmp_path.relative_to(repo_root)), "--require-legacy"],
    ])

    for cmd in steps:
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
