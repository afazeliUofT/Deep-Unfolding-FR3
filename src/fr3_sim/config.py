"""Configuration loading, validation, and derived-parameter resolution.

All simulation parameters are controlled from a single YAML config.
This module:
- Loads YAML
- Applies optional command-line overrides (dot-notation)
- Validates required fields
- Computes derived constants (e.g., num BSs, noise power in Watt, RE scaling factors)

This module must NOT:
- Generate channels
- Run experiments
- Write result files (except optional config dump handled in io_utils)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import copy
import math
import os

import yaml


# ---------
# Helpers
# ---------

def _deep_get(d: Mapping[str, Any], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            raise KeyError("Missing config key: " + ".".join(keys))
        cur = cur[k]
    return cur


def _deep_set(d: MutableMapping[str, Any], keys: List[str], value: Any) -> None:
    cur: MutableMapping[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], MutableMapping):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_override(override: str) -> Tuple[List[str], Any]:
    """Parse a single override of the form 'a.b.c=value'.

    The right-hand side is parsed using YAML, so you can pass lists/dicts, e.g.
    sweep.values=[-5,0,5]
    """
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected format key=value")
    key, raw_value = override.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{override}': empty key")
    # YAML-safe parse for numbers, lists, booleans, etc.
    value = yaml.safe_load(raw_value)
    keys = key.split(".")
    return keys, value


try:
    # Prefer Sionna's conversion helpers when available
    from sionna.phy.utils import dbm_to_watt as _dbm_to_watt  # type: ignore
    from sionna.phy.utils import watt_to_dbm as _watt_to_dbm  # type: ignore
except Exception:  # pragma: no cover
    def _dbm_to_watt(dbm: float) -> float:
        return 1e-3 * (10.0 ** (dbm / 10.0))

    def _watt_to_dbm(watt: float) -> float:
        return 10.0 * math.log10(max(watt, 1e-30) / 1e-3)


def dbm_to_watt(dbm: float) -> float:
    """Convert dBm to Watt."""
    return float(_dbm_to_watt(float(dbm)))


def watt_to_dbm(watt: float) -> float:
    """Convert Watt to dBm."""
    return float(_watt_to_dbm(float(watt)))


def thermal_noise_dbm(bandwidth_hz: float, temperature_K: float = 290.0) -> float:
    """Thermal noise power in dBm over `bandwidth_hz`.

    Uses -174 dBm/Hz at 290K, adjusted for temperature.
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    t_ratio = max(temperature_K, 1e-9) / 290.0
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + 10.0 * math.log10(t_ratio)


# ------------------
# Public structures
# ------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """Config wrapper holding both raw and derived parameters."""

    raw: Dict[str, Any]
    derived: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a combined dict (raw + derived)."""
        out = copy.deepcopy(self.raw)
        out["derived"] = copy.deepcopy(self.derived)
        return out


# ------------------
# Loading + resolve
# ------------------


def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> ResolvedConfig:
    """Load a YAML config and resolve derived parameters.

    Parameters
    ----------
    path: str | Path
        Path to the YAML config file.
    overrides: list[str] | None
        Optional list of overrides (dot-notation), e.g. ['sweep.values=[0,5,10]'].

    Returns
    -------
    ResolvedConfig
        The resolved config.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} did not parse to a dict")

    cfg = copy.deepcopy(cfg)

    # Apply overrides
    if overrides:
        for ov in overrides:
            keys, val = _parse_override(ov)
            _deep_set(cfg, keys, val)

    _validate_minimum(cfg)
    derived = _derive(cfg)
    # Keep track of where the YAML came from so other modules can resolve
    # relative dataset paths (e.g., ISED SMS CSVs) without relying on the CWD.
    derived["config_dir"] = str(path.parent.resolve())
    derived["config_path"] = str(path.resolve())
    return ResolvedConfig(raw=cfg, derived=derived)


def _validate_minimum(cfg: Mapping[str, Any]) -> None:
    """Minimal validation (extend as needed)."""

    # Base required keys (layout-agnostic)
    required_paths = [
        "reproducibility.seed",
        "system_model.carrier_frequency_hz",
        "system_model.subcarrier_spacing_hz",
        "system_model.num_re_total",
        "system_model.num_re_sim",
        "system_model.bs_total_tx_power_dbm",
        "noise.ue_noise_figure_db",
        "noise.temperature_K",
        "topology.scenario",
        "topology.num_ut_per_sector",
        "channel_model.num_bs_ant",
        "channel_model.num_ut_ant",
        "receiver.name",
        "receiver.wmmse.num_iterations",
        "output.results_root",
        "output.experiment_name",
    ]
    for p in required_paths:
        _deep_get(cfg, p.split("."))

    # Layout-specific required keys
    topo = cfg.get("topology", {}) or {}
    layout = str(topo.get("layout", topo.get("mode", "hexgrid"))).lower().strip()

    if layout in ("pcp", "poisson_cluster", "poisson_cluster_process", "cluster_process"):
        _deep_get(cfg, ["topology", "pcp", "num_sites"])
    else:
        _deep_get(cfg, ["topology", "num_rings"])



def _derive(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute derived parameters."""
    derived: Dict[str, Any] = {}

    # Precision
    precision = str(_deep_get(cfg, ["reproducibility", "precision"]))
    if precision not in ("single", "double"):
        raise ValueError("reproducibility.precision must be 'single' or 'double'")
    derived["tf_precision"] = precision

        # Topology-derived sizes
    topo = cfg.get("topology", {}) or {}
    layout = str(topo.get("layout", topo.get("mode", "hexgrid"))).lower().strip()

    u_per_bs = int(_deep_get(cfg, ["topology", "num_ut_per_sector"]))
    if u_per_bs <= 0:
        raise ValueError("topology.num_ut_per_sector must be > 0")

    if layout in ("pcp", "poisson_cluster", "poisson_cluster_process", "cluster_process"):
        pcp = topo.get("pcp", {}) or {}
        sectors_per_site = int(pcp.get("sectors_per_site", 3))
        if sectors_per_site <= 0:
            raise ValueError("topology.pcp.sectors_per_site must be > 0")

        num_sites = int(pcp.get("num_sites", 0))
        if num_sites <= 0:
            raise ValueError("For topology.layout=pcp, please set topology.pcp.num_sites > 0")

        num_bs = num_sites * sectors_per_site
        num_ut = num_bs * u_per_bs

        derived["num_cells"] = num_sites  # interpret as "sites" for PCP
        derived["num_bs"] = num_bs
        derived["num_ut"] = num_ut
        derived["u_per_bs"] = u_per_bs

    else:
        num_rings = int(_deep_get(cfg, ["topology", "num_rings"]))
        if num_rings <= 0:
            raise ValueError("For hexgrid layout, topology.num_rings must be > 0")

        num_cells = 1 + 3 * num_rings * (num_rings + 1)
        num_bs = 3 * num_cells
        num_ut = num_bs * u_per_bs

        derived["num_cells"] = num_cells
        derived["num_bs"] = num_bs
        derived["num_ut"] = num_ut
        derived["u_per_bs"] = u_per_bs


    # Resource grid / RE scaling
    num_re_total = int(_deep_get(cfg, ["system_model", "num_re_total"]))
    simulate_full = bool(_deep_get(cfg, ["system_model", "simulate_full_band"]))
    num_re_sim = int(_deep_get(cfg, ["system_model", "num_re_sim"]))
    if simulate_full:
        num_re_sim = num_re_total
    if num_re_sim <= 0 or num_re_total <= 0:
        raise ValueError("num_re_total and num_re_sim must be positive")
    re_scaling = num_re_total / float(num_re_sim)
    if abs(re_scaling - round(re_scaling)) > 1e-9:
        # Allow non-integer scaling, but warn via derived field
        derived["warning_non_integer_re_scaling"] = True
    derived["num_re_sim"] = num_re_sim
    derived["re_scaling"] = re_scaling

    subcarrier_spacing_hz = float(_deep_get(cfg, ["system_model", "subcarrier_spacing_hz"]))
    derived["bandwidth_total_hz"] = num_re_total * subcarrier_spacing_hz
    derived["bandwidth_sim_hz"] = num_re_sim * subcarrier_spacing_hz

    # Power budgets
    p_tot_dbm = float(_deep_get(cfg, ["system_model", "bs_total_tx_power_dbm"]))
    p_tot_watt = dbm_to_watt(p_tot_dbm)
    derived["bs_total_tx_power_watt"] = p_tot_watt
    # Per-sim-RE-group budget (for reduced-band simulation)
    derived["bs_power_budget_sim_watt"] = p_tot_watt / re_scaling

    # UE noise per RE (per simulated RE)
    nf_ue_db = float(_deep_get(cfg, ["noise", "ue_noise_figure_db"]))
    temp_K = float(_deep_get(cfg, ["noise", "temperature_K"]))
    extra_noise_db = float(_deep_get(cfg, ["noise", "extra_noise_db"]))
    # Noise over one subcarrier (RE)
    n0_dbm = thermal_noise_dbm(subcarrier_spacing_hz, temperature_K=temp_K)
    noise_re_dbm = n0_dbm + nf_ue_db + extra_noise_db
    noise_re_watt = dbm_to_watt(noise_re_dbm)
    derived["ue_noise_re_dbm"] = noise_re_dbm
    derived["ue_noise_re_watt"] = noise_re_watt

    # Fixed service noise and I_max
    fs_enabled = bool(_deep_get(cfg, ["fixed_service", "enabled"]))
    derived["fixed_service_enabled"] = fs_enabled
    if fs_enabled:
        fs_bw = float(_deep_get(cfg, ["fixed_service", "rx_bandwidth_hz"]))
        fs_nf_db = float(_deep_get(cfg, ["fixed_service", "noise_figure_db"]))
        fs_noise_dbm_cfg = _deep_get(cfg, ["fixed_service", "noise_power_dbm"])
        if fs_noise_dbm_cfg is None:
            fs_noise_dbm = thermal_noise_dbm(fs_bw, temperature_K=temp_K) + fs_nf_db
        else:
            fs_noise_dbm = float(fs_noise_dbm_cfg)
        in_target_db = float(_deep_get(cfg, ["fixed_service", "in_target_db"]))
        i_max_dbm_override = _deep_get(cfg, ["fixed_service", "i_max_dbm_override"])
        if i_max_dbm_override is None:
            i_max_dbm = fs_noise_dbm + in_target_db
        else:
            i_max_dbm = float(i_max_dbm_override)
        derived["fs_noise_dbm"] = fs_noise_dbm
        derived["fs_noise_watt"] = dbm_to_watt(fs_noise_dbm)
        derived["fs_i_max_dbm"] = i_max_dbm
        derived["fs_i_max_watt"] = dbm_to_watt(i_max_dbm)

    return derived
