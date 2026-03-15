"""Metrics computation.

This module converts raw tensors (beamformers, SINR, constraint values)
into scalar metrics suitable for logging, CSV export, and plotting.

It must NOT:
- Run the receiver iterations
- Write files (handled by io_utils.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf

from .channel import FsStats
from .processing import MmseOutput


@dataclass(frozen=True)
class ExperimentMetrics:
    """Scalar metrics aggregated over the batch.

    All fields are python floats for easy JSON/CSV export.
    """

    sum_rate_bps_per_hz: float
    avg_user_rate_bps_per_hz: float
    avg_sinr_db: float
    max_bs_power_watt: float
    max_bs_power_violation_watt: float
    max_fs_interference_watt: float
    max_fs_violation_watt: float


def _log2(x: tf.Tensor) -> tf.Tensor:
    try:
        from sionna.phy.utils import log2  # type: ignore

        return log2(x)
    except Exception:
        return tf.math.log(x) / tf.math.log(tf.constant(2.0, dtype=x.dtype))


def compute_metrics(
    *,
    w: tf.Tensor,
    mmse: MmseOutput,
    num_re_sim: int,
    re_scaling: float,
    p_tot_watt: float,
    fs: Optional[FsStats],
) -> ExperimentMetrics:
    """Compute paper-style scalar metrics.

    Parameters
    ----------
    w : tf.Tensor
        Beamformers with shape [batch, T, B, M, U_per_bs].
    mmse : MmseOutput
        MMSE output computed for the final iterate. `mmse.sinr` has shape [batch*T, U].
    num_re_sim : int
        Number of simulated RE groups (T).
    re_scaling : float
        Number of physical REs represented by one simulated RE group (|N|/T).
    p_tot_watt : float
        Per-BS total TX power constraint over the full band.
    fs : FsStats | None
        FS statistics (if enabled).

    Returns
    -------
    ExperimentMetrics
        Scalar metrics (python floats).
    """

    real_dtype = tf.float64 if mmse.mse.dtype == tf.float64 else tf.float32

    # ----- Rates -----
    batch = tf.shape(w)[0]
    T = tf.shape(w)[1]
    B = tf.shape(w)[2]
    U_total = tf.shape(mmse.sinr)[1]

    sinr = tf.reshape(mmse.sinr, [batch, T, U_total])
    rate = _log2(1.0 + tf.cast(sinr, real_dtype))  # [batch, T, U]

    sum_rate_per_tone = tf.reduce_sum(rate, axis=-1)  # [batch, T]
    sum_rate = tf.reduce_mean(sum_rate_per_tone, axis=-1)  # [batch]
    sum_rate_bps_per_hz = tf.reduce_mean(sum_rate)  # scalar

    avg_user_rate = tf.reduce_mean(rate)  # scalar average over batch, T, U

    # SINR in dB
    sinr_db = 10.0 * tf.math.log(tf.cast(tf.maximum(sinr, 1e-12), real_dtype)) / tf.math.log(tf.constant(10.0, dtype=real_dtype))
    avg_sinr_db = tf.reduce_mean(sinr_db)

    # ----- Power -----
    # Power per BS over full band: re_scaling * sum_t sum_u ||w||^2
    pow_t_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])  # [batch, T, B]
    pow_b = tf.cast(re_scaling, real_dtype) * tf.reduce_sum(pow_t_b, axis=1)  # [batch, B]
    max_bs_power = tf.reduce_max(pow_b)
    # Report *positive* constraint violation (0 if satisfied)
    max_bs_power_violation = tf.reduce_max(tf.maximum(pow_b - tf.cast(p_tot_watt, real_dtype), 0.0))

    # ----- FS Interference -----
        # ----- FS Interference -----
    if fs is None:
        max_fs_int = tf.cast(0.0, real_dtype)
        max_fs_violation = tf.cast(0.0, real_dtype)
    else:
        corr = getattr(fs, "correlation", "identity")

        if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
            a = tf.cast(fs.a_bs_fs, w.dtype)  # [batch,B,L,M] complex

            proj = tf.einsum(
                "sblm,stbmu->stblu",
                tf.math.conj(a),
                w,
                optimize=True,
            )  # [batch,T,B,L,U_per_bs]

            p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)  # [batch,T,B,L]

            delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])  # [T,B]
            p_dir = tf.cast(p_dir, real_dtype) * delta_tb[None, :, :, None]

            I_fs = tf.cast(re_scaling, real_dtype) * tf.einsum(
                "stbl,tl,sbl->sl",
                p_dir,
                tf.cast(fs.epsilon, real_dtype),
                tf.cast(fs.bar_beta, real_dtype),
                optimize=True,
            )  # [batch,L]
        else:
            # trace model (legacy)
            pow_t_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])  # [batch,T,B]
            delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])  # [T,B]
            pow_t_b = pow_t_b * delta_tb[None, :, :]

            I_fs = tf.cast(re_scaling, real_dtype) * tf.einsum(
                "stb,tl,sbl->sl",
                pow_t_b,
                tf.cast(fs.epsilon, real_dtype),
                tf.cast(fs.bar_beta, real_dtype),
                optimize=True,
            )  # [batch,L]

        max_fs_int = tf.reduce_max(I_fs)
        max_fs_violation = tf.reduce_max(tf.maximum(I_fs - tf.cast(fs.i_max_watt[None, :], real_dtype), 0.0))


    return ExperimentMetrics(
        sum_rate_bps_per_hz=float(sum_rate_bps_per_hz.numpy()),
        avg_user_rate_bps_per_hz=float(avg_user_rate.numpy()),
        avg_sinr_db=float(avg_sinr_db.numpy()),
        max_bs_power_watt=float(max_bs_power.numpy()),
        max_bs_power_violation_watt=float(max_bs_power_violation.numpy()),
        max_fs_interference_watt=float(max_fs_int.numpy()),
        max_fs_violation_watt=float(max_fs_violation.numpy()),
    )


def metrics_to_flat_dict(metrics: ExperimentMetrics) -> Dict[str, Any]:
    """Convert metrics dataclass to a flat dict for Pandas/CSV."""

    return {
        "sum_rate_bps_per_hz": metrics.sum_rate_bps_per_hz,
        "avg_user_rate_bps_per_hz": metrics.avg_user_rate_bps_per_hz,
        "avg_sinr_db": metrics.avg_sinr_db,
        "max_bs_power_watt": metrics.max_bs_power_watt,
        "max_bs_power_violation_watt": metrics.max_bs_power_violation_watt,
        "max_fs_interference_watt": metrics.max_fs_interference_watt,
        "max_fs_violation_watt": metrics.max_fs_violation_watt,
    }
