"""Metrics for algorithm comparison and paper-style outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from .algorithms import fs_interference, user_rate_tensors
from .types import AlgorithmResult, WidebandBatch


def _to_float(x: tf.Tensor | np.ndarray | float) -> float:
    if isinstance(x, tf.Tensor):
        return float(x.numpy())
    if isinstance(x, np.ndarray):
        return float(x.item())
    return float(x)


def summarize_algorithm(
    batch: WidebandBatch,
    result: AlgorithmResult,
    sweep_tag: str,
    sweep_value: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rr = user_rate_tensors(batch, result.w)
    rate = rr["rate"]                                                   # [S,K,U]
    sum_rate = tf.reduce_sum(rate, axis=2)                              # [S,K]
    mean_sum_rate = tf.reduce_mean(sum_rate, axis=1)                    # [S]

    user_weights = tf.cast(batch.user_weights[None, None, :], rate.dtype)
    weighted_rate = tf.reduce_mean(tf.reduce_sum(rate * user_weights, axis=2), axis=1)

    i_fs = fs_interference(batch, result.w)                             # [S,L]
    i_max = tf.cast(batch.fs_stats.i_max_watt, i_fs.dtype)[None, :] + 1e-15
    excess = i_fs / i_max - 1.0
    worst_excess = tf.reduce_max(excess, axis=1)
    mean_positive_excess = tf.reduce_mean(tf.nn.relu(excess), axis=1)
    fs_outage = tf.cast(worst_excess > 0.0, i_fs.dtype)

    pow_b = tf.reduce_sum(tf.abs(result.w) ** 2, axis=[1, 3, 4])
    power_violation = tf.reduce_mean(tf.nn.relu(pow_b / tf.cast(batch.bs_power_budget_watt, pow_b.dtype) - 1.0), axis=1)

    active_subband_fraction = tf.reduce_mean(
        tf.cast(
            tf.reduce_sum(tf.abs(result.w) ** 2, axis=[2, 3, 4]) > 1e-12,
            rate.dtype,
        ),
        axis=1,
    )

    flat_excess = tf.reshape(tf.nn.relu(excess), [-1]).numpy()
    if flat_excess.size == 0:
        cvar95 = 0.0
    else:
        q = np.quantile(flat_excess, 0.95)
        tail = flat_excess[flat_excess >= q]
        cvar95 = float(tail.mean()) if tail.size else 0.0

    summary = {
        "algorithm": result.name,
        "sweep_tag": str(sweep_tag),
        "sweep_value": float(sweep_value),
        "sum_rate_bps_per_hz_mean": float(tf.reduce_mean(mean_sum_rate).numpy()),
        "weighted_sum_rate_mean": float(tf.reduce_mean(weighted_rate).numpy()),
        "sum_rate_bps_per_hz_p50": float(np.percentile(mean_sum_rate.numpy(), 50.0)),
        "sum_rate_bps_per_hz_p05": float(np.percentile(mean_sum_rate.numpy(), 5.0)),
        "sum_rate_bps_per_hz_p95": float(np.percentile(mean_sum_rate.numpy(), 95.0)),
        "fs_outage_prob_any": float(tf.reduce_mean(fs_outage).numpy()),
        "worst_fs_excess_mean": float(tf.reduce_mean(worst_excess).numpy()),
        "mean_positive_fs_excess": float(tf.reduce_mean(mean_positive_excess).numpy()),
        "cvar95_fs_excess": cvar95,
        "power_violation_mean": float(tf.reduce_mean(power_violation).numpy()),
        "active_subband_fraction_mean": float(tf.reduce_mean(active_subband_fraction).numpy()),
        "runtime_ms_mean": float(result.runtime_s * 1e3 / max(1, int(rate.shape[0]))),
        "mean_delay_spread_ns": float(tf.reduce_mean(batch.delay_spread_ns).numpy()),
        "blocked_subband_fraction_mean": float(1.0 - tf.reduce_mean(batch.static_gate).numpy()),
    }

    per_sample_rows: List[Dict[str, float]] = []
    for idx in range(int(rate.shape[0])):
        per_sample_rows.append(
            {
                "algorithm": result.name,
                "sweep_tag": str(sweep_tag),
                "sweep_value": float(sweep_value),
                "sample_id": idx,
                "sum_rate_bps_per_hz": float(mean_sum_rate[idx].numpy()),
                "weighted_sum_rate": float(weighted_rate[idx].numpy()),
                "worst_fs_excess": float(worst_excess[idx].numpy()),
                "mean_positive_fs_excess": float(mean_positive_excess[idx].numpy()),
                "fs_outage_any": float(fs_outage[idx].numpy()),
                "power_violation": float(power_violation[idx].numpy()),
                "active_subband_fraction": float(active_subband_fraction[idx].numpy()),
                "runtime_ms": float(result.runtime_s * 1e3 / max(1, int(rate.shape[0]))),
            }
        )

    return summary, per_sample_rows


def tone_grouping_error(
    batch: WidebandBatch,
    group_sizes: Iterable[int],
) -> pd.DataFrame:
    """Compute a simple coherence-bandwidth proxy and grouping error."""
    h = batch.h_eff  # [S,K,B,U,M]
    k = int(h.shape[1])
    rows: List[Dict[str, float]] = []

    # Frequency-correlation proxy over adjacent subbands
    h0 = h[:, :-1, :, :, :]
    h1 = h[:, 1:, :, :, :]
    num = tf.reduce_sum(tf.math.conj(h0) * h1, axis=[2, 3, 4])
    den = tf.sqrt(
        tf.reduce_sum(tf.abs(h0) ** 2, axis=[2, 3, 4]) * tf.reduce_sum(tf.abs(h1) ** 2, axis=[2, 3, 4]) + 1e-15
    )
    rho_adj = tf.reduce_mean(tf.abs(num / den)).numpy()
    subband_bw = float(batch.grid.subband_bw_hz)
    coherence_bw_proxy_hz = subband_bw / max(1e-6, 1.0 - float(rho_adj))

    for g in group_sizes:
        g = int(max(1, g))
        if g > k:
            continue
        num_groups = int(np.ceil(k / g))
        h_hat_pieces = []
        for idx in range(num_groups):
            start = idx * g
            end = min(k, (idx + 1) * g)
            h_avg = tf.reduce_mean(h[:, start:end, ...], axis=1, keepdims=True)
            h_hat_pieces.append(tf.tile(h_avg, [1, end - start, 1, 1, 1]))
        h_hat = tf.concat(h_hat_pieces, axis=1)
        nmse = tf.reduce_mean(tf.abs(h - h_hat) ** 2) / tf.maximum(tf.reduce_mean(tf.abs(h) ** 2), 1e-15)

        rows.append(
            {
                "group_size": float(g),
                "num_groups": float(num_groups),
                "nmse": float(nmse.numpy()),
                "adjacent_freq_correlation": float(rho_adj),
                "coherence_bw_proxy_hz": float(coherence_bw_proxy_hz),
            }
        )

    return pd.DataFrame(rows)


def history_to_frame(history: Dict[str, List[float]], algorithm: str) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["algorithm", "layer", "utility", "mean_violation", "cvar", "power_violation"])
    n = max(len(v) for v in history.values())
    rows = []
    for i in range(n):
        row = {"algorithm": algorithm, "layer": i + 1}
        for key, vals in history.items():
            row[key] = float(vals[i]) if i < len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
