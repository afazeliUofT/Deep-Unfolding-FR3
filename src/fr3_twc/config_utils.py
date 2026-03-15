"""Helpers for reading the extra `twc` configuration block."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence


def twc_get(cfg: Any, path: Sequence[str], default: Any = None) -> Any:
    """Read a value from cfg.raw['twc'] with a safe default."""
    cur: Any = cfg.raw.get("twc", {})
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def twc_require(cfg: Any, path: Sequence[str]) -> Any:
    """Read a required value from cfg.raw['twc']."""
    cur: Any = cfg.raw.get("twc", {})
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError("Missing twc config key: " + ".".join(path))
        cur = cur[key]
    return cur


def results_root_dir(cfg: Any) -> Path:
    """Return the experiment root directory used by both legacy and TWC pipelines."""
    return Path(str(cfg.raw["output"]["results_root"])) / str(cfg.raw["output"]["experiment_name"])
