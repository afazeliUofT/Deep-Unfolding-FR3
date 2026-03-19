#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Path-valued fields that must preserve the semantics of the base config.
# The specialized v20 overrides inherit configs whose dataset paths are written
# relative to the repo's `configs/` directory, e.g. `../data/...`.
# When the merged config is materialized elsewhere, those paths break unless
# they are rewritten against the base config directory.
KNOWN_RELATIVE_PATH_KEYS = [
    ("fixed_service", "ised_sms", "fixed_service_csv"),
    ("fixed_service", "ised_sms", "antenna_reference_csv"),
    ("fixed_service", "ised_sms", "radio_reference_csv"),
]


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = deep_merge(merged[key], value) if key in merged else value
        return merged
    return override


def _resolve_from_base(base_dir: Path, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip() == "":
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _absolutize_known_paths(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False)) or {}
    for keys in KNOWN_RELATIVE_PATH_KEYS:
        cur: Any = out
        ok = True
        for key in keys[:-1]:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and isinstance(cur, dict) and keys[-1] in cur:
            cur[keys[-1]] = _resolve_from_base(base_dir, cur[keys[-1]])
    return out


def _materialize_resolved_config(repo_root: Path, exp_name: str, merged_cfg: dict[str, Any]) -> Path:
    # Keep the resolved YAML directly under `configs/` so that any remaining
    # relative paths such as `../data/...` continue to resolve the same way they
    # do for the original repo configs.
    tmp_path = repo_root / "configs" / f"__resolved_override_{exp_name}.yaml"
    tmp_path.write_text(yaml.safe_dump(merged_cfg, sort_keys=False), encoding="utf-8")
    return tmp_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", required=True, help="Path to override YAML")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    override_arg = Path(args.override)
    override_path = override_arg.resolve() if override_arg.is_absolute() else (repo_root / override_arg).resolve()
    if not override_path.exists():
        raise SystemExit(f"Override config not found: {override_path}")

    override_cfg = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    base_config_rel = override_cfg.pop("base_config", None)
    if not base_config_rel:
        raise SystemExit("Override config must include 'base_config'.")

    base_arg = Path(base_config_rel)
    base_path = base_arg.resolve() if base_arg.is_absolute() else (repo_root / base_arg).resolve()
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")

    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    merged = deep_merge(base_cfg, override_cfg)
    merged = _absolutize_known_paths(merged, base_path.parent)

    output_cfg = merged.setdefault("output", {})
    exp_name = output_cfg.get("experiment_name")
    if not exp_name:
        raise SystemExit("Merged config must define output.experiment_name")

    results_root_cfg = Path(str(output_cfg.get("results_root", "results")))
    results_root = results_root_cfg if results_root_cfg.is_absolute() else (repo_root / results_root_cfg)
    pipe_dir = results_root / exp_name / "twc_pipeline"
    if pipe_dir.exists():
        shutil.rmtree(pipe_dir)

    tmp_path = _materialize_resolved_config(repo_root, exp_name, merged)
    rel_cfg = str(tmp_path.relative_to(repo_root))

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    steps: list[list[str]] = []
    if not args.skip_probe:
        steps.append([sys.executable, "scripts/twc_probe_env.py", "--strict"])
    steps.extend(
        [
            [sys.executable, "scripts/twc_run_legacy_baseline.py", "--config", rel_cfg],
            [sys.executable, "scripts/twc_run_pipeline.py", "--config", rel_cfg],
            [sys.executable, "scripts/twc_plot_figures.py", "--config", rel_cfg],
            [sys.executable, "scripts/twc_verify_outputs.py", "--config", rel_cfg, "--require-legacy"],
        ]
    )

    try:
        for cmd in steps:
            subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    finally:
        # Keep the repo clean after the workflow finishes.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
