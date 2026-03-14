"""Experiment runner (orchestration + sweeps).

This layer wires together:
- config
- deterministic seeding
- topology generation
- channel + FS-stat generation
- receiver baseline (classical WMMSE)
- metrics aggregation
- optional plotting + result saving

It must NOT:
- Contain the details of WMMSE math (receiver.py)
- Re-implement pathloss/channel models (channel.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from .channel import FsStats, generate_fs_stats, generate_ue_channels
from .config import ResolvedConfig, dbm_to_watt
from .io_utils import create_experiment_dir, save_metrics_csv, save_resolved_config, setup_logger
from .metrics import compute_metrics, metrics_to_flat_dict
from .plotting import plot_sweep
from .receiver import WmmseReceiver
from .seeding import set_global_seed
from .topology import TopologyData, generate_fixed_service_locations, generate_hexgrid_topology



def _compute_effective_noise_var_watt(cfg: ResolvedConfig, snr_db: float) -> float:
    """Compute effective UE noise variance per RE in Watt.

    In this codebase, an SNR sweep is implemented as a *relative* noise scaling:

        sigma2_eff = sigma2_base * 10^(-snr_db/10)

    where sigma2_base is derived from the config (kTB + NF + noise.extra_noise_db).
    """
    base = float(cfg.derived["ue_noise_re_watt"])
    return base * (10.0 ** (-float(snr_db) / 10.0))


def _compute_effective_bs_power_watt(bs_total_tx_power_dbm: float) -> float:
    """Convert BS total TX power budget (dBm, over full band) to Watt."""
    return dbm_to_watt(float(bs_total_tx_power_dbm))


def _compute_effective_fs_i_max_watt(cfg: ResolvedConfig, fs_in_target_db: float) -> float:
    """Compute FS interference threshold I_max (Watt).

    The paper defines:
        I_max = N * 10^(I/N / 10)
    with N being the FS noise power over its RX bandwidth.

    This code uses the derived FS noise power in dBm (cfg.derived['fs_noise_dbm']).
    """
    fs_noise_dbm = float(cfg.derived["fs_noise_dbm"])
    i_max_dbm = fs_noise_dbm + float(fs_in_target_db)
    return dbm_to_watt(i_max_dbm)


def _mean_dicts(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Elementwise mean over a list of dicts with numeric values."""
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows if k in r], dtype=np.float64)
        out[k] = float(vals.mean()) if vals.size > 0 else float("nan")
    return out

def _tile_topology_to_batch(topo_single: TopologyData, batch_size: int) -> TopologyData:
    """Tile a single-snapshot topology (batch=1) to an arbitrary batch_size."""
    if int(topo_single.bs_loc.shape[0]) == batch_size:
        return topo_single
    if int(topo_single.bs_loc.shape[0]) != 1:
        raise ValueError(
            "freeze_topology expects a single-snapshot topology with batch dimension 1, "
            f"got batch={int(topo_single.bs_loc.shape[0])}."
        )

    bs_loc = tf.tile(topo_single.bs_loc, [batch_size, 1, 1])
    ut_loc = tf.tile(topo_single.ut_loc, [batch_size, 1, 1])
    bs_virtual_loc = tf.tile(topo_single.bs_virtual_loc, [batch_size, 1, 1, 1])

    ut_orientations = tf.tile(topo_single.ut_orientations, [batch_size, 1, 1])
    bs_orientations = tf.tile(topo_single.bs_orientations, [batch_size, 1, 1])
    ut_velocities = tf.tile(topo_single.ut_velocities, [batch_size, 1, 1])

    ut_is_indoor = None
    if topo_single.ut_is_indoor is not None:
        ut_is_indoor = tf.tile(topo_single.ut_is_indoor, [batch_size, 1])

    return TopologyData(
        bs_loc=bs_loc,
        ut_loc=ut_loc,
        bs_virtual_loc=bs_virtual_loc,
        ut_orientations=ut_orientations,
        bs_orientations=bs_orientations,
        ut_velocities=ut_velocities,
        serving_bs=topo_single.serving_bs,
        ut_is_indoor=ut_is_indoor,
    )



def run_experiment(cfg: ResolvedConfig) -> str:
    """Run a (possibly swept) experiment and write outputs under `output.results_root`.

    Parameters
    ----------
    cfg : ResolvedConfig
        Loaded + resolved configuration.

    Returns
    -------
    str
        Path to the created results directory.
    """

    # -----------------
    # Setup
    # -----------------
    seed = int(cfg.raw["reproducibility"]["seed"])
    deterministic_tf = bool(cfg.raw["reproducibility"].get("deterministic_tf", True))
    set_global_seed(seed, deterministic_tf=deterministic_tf)

    paths = create_experiment_dir(
        results_root=str(cfg.raw["output"]["results_root"]),
        experiment_name=str(cfg.raw["output"]["experiment_name"]),
        overwrite=bool(cfg.raw["output"].get("overwrite", False)),
    )
    logger = setup_logger(paths.log_file)

    logger.info("Experiment directory: %s", str(paths.root))
    logger.info("Selected derived parameters: %s", {k: cfg.derived[k] for k in sorted(cfg.derived.keys())})

    if bool(cfg.raw["output"].get("save_config_resolved", True)):
        save_resolved_config(paths.config_resolved_yaml, cfg.to_dict())

    # -----------------
    # Receiver
    # -----------------
    recv_name = str(cfg.raw.get("receiver", {}).get("name", "wmmse")).lower()
    if recv_name != "wmmse":
        raise ValueError(f"Unsupported receiver '{recv_name}'. Only 'wmmse' is implemented in this baseline.")
    receiver = WmmseReceiver()

    # -----------------
    # Sweep setup
    # -----------------
    sweep_cfg = cfg.raw.get("sweep", {})
    sweep_enabled = bool(sweep_cfg.get("enabled", False))
    variable = str(sweep_cfg.get("variable", "snr_db")).lower()
    values = sweep_cfg.get("values", None)
    if values is None:
        # Backward-compatible alias from the README draft
        values = sweep_cfg.get("values_db", [0.0])
    values = list(values) if sweep_enabled else [0.0]

    batch_size = int(cfg.raw["experiment"]["batch_size"])
    num_batches = int(cfg.raw["experiment"]["num_batches"])

        # Fixed Service (FS) config
    fs_cfg = cfg.raw.get("fixed_service", {}) or {}
    fs_enabled = bool(fs_cfg.get("enabled", False))


        # Optional: freeze topology (single fixed BS/UE geometry for all batches + sweep points)
    exp_cfg = cfg.raw.get("experiment", {}) or {}
    freeze_topology = bool(exp_cfg.get("freeze_topology", False))

    topo_fixed = None
    fs_stats_fixed = None

    if freeze_topology:
        logger.info("freeze_topology=True -> using a single fixed BS/UE geometry for all batches and sweep points.")
        topo_single = generate_hexgrid_topology(cfg, batch_size=1)
        topo_fixed = _tile_topology_to_batch(topo_single, batch_size=batch_size)

        # FS is already deterministic with selection.random_seed, but precomputing also saves time.
        if fs_enabled:
            fs_loc_fixed = generate_fixed_service_locations(cfg, topo_fixed, batch_size=batch_size)
            fs_stats_fixed = generate_fs_stats(cfg, topo_fixed, fs_loc_fixed, batch_size=batch_size)



    logger.info("Sweep enabled=%s variable=%s values=%s", sweep_enabled, variable, values)
    logger.info("Monte Carlo: num_batches=%d batch_size=%d", num_batches, batch_size)
    logger.info("Fixed service enabled=%s", fs_enabled)

    # Baseline values from config
    base_bs_power_dbm = float(cfg.raw["system_model"]["bs_total_tx_power_dbm"])
    base_fs_in_target_db = float(fs_cfg.get("in_target_db", -6.0))

    all_rows: List[Dict[str, Any]] = []

    for val in values:
        # Resolve sweep point
        snr_db = 0.0
        bs_power_dbm = base_bs_power_dbm
        fs_in_target_db = base_fs_in_target_db

        if variable == "snr_db":
            snr_db = float(val)
        elif variable == "bs_total_tx_power_dbm":
            bs_power_dbm = float(val)
        elif variable == "fs_in_target_db":
            fs_in_target_db = float(val)
        else:
            raise ValueError(
                "Unsupported sweep.variable. Supported: snr_db, bs_total_tx_power_dbm, fs_in_target_db"
            )

        noise_var_watt = _compute_effective_noise_var_watt(cfg, snr_db)
        p_tot_watt = _compute_effective_bs_power_watt(bs_power_dbm)
        i_max_watt = _compute_effective_fs_i_max_watt(cfg, fs_in_target_db) if fs_enabled else 0.0

        logger.info(
            "Sweep point %s=%s -> snr_db=%.3f, P_tot=%.3f W, noise=%.3e W, I_max=%.3e W",
            variable,
            str(val),
            snr_db,
            p_tot_watt,
            noise_var_watt,
            i_max_watt,
        )

        per_batch: List[Dict[str, float]] = []

        for _ in tqdm(range(num_batches), desc=f"{variable}={val}"):
            topo = topo_fixed if topo_fixed is not None else generate_hexgrid_topology(cfg, batch_size=batch_size)


            H = generate_ue_channels(cfg, topo, batch_size=batch_size)

            fs_stats: Optional[FsStats]
            if fs_enabled:
                if fs_stats_fixed is not None:
                    fs_stats = fs_stats_fixed
                else:
                    fs_loc = generate_fixed_service_locations(cfg, topo, batch_size=batch_size)
                    fs_stats = generate_fs_stats(cfg, topo, fs_loc, batch_size=batch_size)
                # Override I_max (sweep dependent)
                # Override I_max depending on sweep_override_i_max mode
                fs_override_mode = str(fs_cfg.get("sweep_override_i_max", "scalar")).lower().strip()

                if fs_override_mode == "none":
                    i_max_tensor = fs_stats.i_max_watt

                elif fs_override_mode == "scalar":
                    # Legacy behavior: ignore per-FS I_max and use scalar (noise + I/N)
                    i_max_watt = _compute_effective_fs_i_max_watt(cfg, fs_in_target_db)
                    i_max_tensor = tf.ones_like(fs_stats.i_max_watt) * tf.cast(i_max_watt, fs_stats.i_max_watt.dtype)

                elif fs_override_mode == "scale":
                    # Scale per-FS I_max by delta in I/N (dB) relative to base config value
                    scale_db = float(fs_in_target_db) - float(base_fs_in_target_db)
                    scale_lin = 10.0 ** (scale_db / 10.0)
                    i_max_tensor = fs_stats.i_max_watt * tf.cast(scale_lin, fs_stats.i_max_watt.dtype)

                else:
                    raise ValueError(f"Unknown fixed_service.sweep_override_i_max='{fs_override_mode}'. Use none/scalar/scale.")

                fs_stats = FsStats(
                    bar_beta=fs_stats.bar_beta,
                    epsilon=fs_stats.epsilon,
                    delta=fs_stats.delta,
                    i_max_watt=i_max_tensor,
                    correlation=fs_stats.correlation,
                    a_bs_fs=fs_stats.a_bs_fs,
                )


            else:
                fs_stats = None

            # Solve for beamformers
            result = receiver.solve(
                cfg,
                H,
                noise_var_watt=float(noise_var_watt),
                fs=fs_stats,
                bs_total_tx_power_watt=float(p_tot_watt),
            )

            # Metrics
            m = compute_metrics(
                w=result.w,
                mmse=result.mmse,
                num_re_sim=int(cfg.derived["num_re_sim"]),
                re_scaling=float(cfg.derived["re_scaling"]),
                p_tot_watt=float(p_tot_watt),
                fs=fs_stats,
            )
            d = metrics_to_flat_dict(m)
            d["converged"] = float(1.0 if result.converged else 0.0)
            d["num_iter"] = float(result.num_iter)
            per_batch.append({k: float(v) for k, v in d.items()})

        mean_metrics = _mean_dicts(per_batch)
        mean_metrics.update(
            {
                "sweep_variable": variable,
                "sweep_value": float(val),
                "snr_db": float(snr_db),
                "bs_total_tx_power_dbm": float(bs_power_dbm),
                "fs_in_target_db": float(fs_in_target_db),
            }
        )
        all_rows.append(mean_metrics)

    # -----------------
    # Save outputs
    # -----------------
    if bool(cfg.raw["output"].get("save_metrics_csv", True)):
        save_metrics_csv(paths.metrics_csv, all_rows)
        logger.info("Saved metrics to %s", str(paths.metrics_csv))

    # Plot
    plot_enabled = bool(cfg.raw.get("plotting", {}).get("enabled", True)) and bool(
        cfg.raw["output"].get("save_plots", True)
    )
    if plot_enabled:
        try:
            df = pd.DataFrame(all_rows)
            plot_path = plot_sweep(
                df=df,
                x_col="sweep_value",
                y_cols=[
                    "sum_rate_bps_per_hz",
                    "max_bs_power_violation_watt",
                    "max_fs_violation_watt",
                ],
                title=f"FR3 WMMSE baseline: {paths.root.name}",
                x_label=variable,
                y_label="metric",
                out_path=paths.figures_dir / "summary.png",
                dpi=int(cfg.raw.get("plotting", {}).get("dpi", 200)),
                file_format=str(cfg.raw.get("plotting", {}).get("format", "png")),
            )
            logger.info("Saved plot to %s", str(plot_path))
        except Exception as e:  # pragma: no cover
            logger.warning("Plotting failed: %s", str(e))

    return str(paths.root)
