"""I/O utilities for experiments.

All file-system side-effects are intentionally centralized here.
Other modules must NOT write to disk.

What this module does:
- Create a unique results directory per experiment
- Configure a logger that writes both to console and a log file
- Save resolved configs
- Save metrics tables (CSV)
- Save figures
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import yaml


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths for a single experiment run."""

    root: Path
    metrics_csv: Path
    config_resolved_yaml: Path
    log_file: Path
    figures_dir: Path


def create_experiment_dir(results_root: str, experiment_name: str, overwrite: bool = False) -> ExperimentPaths:
    """Create a timestamped experiment directory.

    Parameters
    ----------
    results_root : str
        Base output directory (e.g., "results").
    experiment_name : str
        Human-friendly experiment name.
    overwrite : bool
        If True and the directory exists, it will be re-used.

    Returns
    -------
    ExperimentPaths
        Common file paths inside the created directory.
    """

    results_root_p = Path(results_root)
    results_root_p.mkdir(parents=True, exist_ok=True)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = results_root_p / f"{experiment_name}_{stamp}"

    if exp_dir.exists() and not overwrite:
        # Add a short disambiguator
        for i in range(1, 1000):
            candidate = Path(f"{exp_dir}_{i:03d}")
            if not candidate.exists():
                exp_dir = candidate
                break

    exp_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = exp_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentPaths(
        root=exp_dir,
        metrics_csv=exp_dir / "metrics.csv",
        config_resolved_yaml=exp_dir / "config_resolved.yaml",
        log_file=exp_dir / "run.log",
        figures_dir=figures_dir,
    )


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger that logs to both console and a file."""

    logger = logging.getLogger("fr3_sim")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers in notebooks
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(level)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def save_resolved_config(path: Path, cfg_dict: Dict[str, Any]) -> None:
    """Save the resolved configuration to YAML."""

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)


def save_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Save metrics rows to CSV."""

    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)


def save_json(path: Path, obj: Any) -> None:
    """Save an object to JSON (best-effort)."""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
