### Updated ###
"""Wideband FR3 channel generation for the TWC upgrade."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf

from fr3_sim.channel import generate_fs_stats, generate_ue_channels, umi_los_probability
from fr3_sim.topology import generate_fixed_service_locations, generate_hexgrid_topology

from .config_utils import twc_get
from .sionna_nr import build_nr_grid
from .types import WidebandBatch


def _real_dtype(cfg: Any) -> tf.DType:
    return tf.float64 if str(cfg.derived.get("tf_precision", "single")).lower().startswith("double") else tf.float32


def _complex_dtype(cfg: Any) -> tf.DType:
    return tf.complex128 if _real_dtype(cfg) == tf.float64 else tf.complex64


def _complex_normal(shape: Tuple[int, ...], dtype: tf.DType) -> tf.Tensor:
    real_dtype = tf.float64 if dtype == tf.complex128 else tf.float32
    re = tf.random.normal(shape, dtype=real_dtype)
    im = tf.random.normal(shape, dtype=real_dtype)
    return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), dtype)


def _toeplitz_cholesky(size: int, rho: float, real_dtype: tf.DType) -> tf.Tensor:
    rho = float(max(0.0, min(0.999, rho)))
    idx = tf.range(size, dtype=real_dtype)
    dist = tf.abs(idx[:, None] - idx[None, :])
    mat = tf.pow(tf.cast(rho, real_dtype), dist)
    return tf.linalg.cholesky(mat + 1e-8 * tf.eye(size, dtype=real_dtype))


def _apply_spatial_correlation(g: tf.Tensor, rho_ue: float, rho_bs: float, cfg: Any) -> tf.Tensor:
    """Apply separable exponential spatial correlation.

    Parameters
    ----------
    g : [S,B,U,P,Nr,M] complex Gaussian taps
    """
    real_dtype = _real_dtype(cfg)
    nr = int(g.shape[-2])
    m = int(g.shape[-1])

    if nr <= 1 and m <= 1:
        return g

    if nr > 1 and rho_ue > 0.0:
        l_ue = tf.cast(_toeplitz_cholesky(nr, rho_ue, real_dtype), g.dtype)
        g = tf.einsum("ij,sbupjm->sbupim", l_ue, g, optimize=True)

    if m > 1 and rho_bs > 0.0:
        l_bs = tf.cast(_toeplitz_cholesky(m, rho_bs, real_dtype), g.dtype)
        g = tf.einsum("sbupim,mn->sbupin", g, tf.math.conj(tf.transpose(l_bs)), optimize=True)

    return g


def _sample_delay_spread_ns(cfg: Any, d2d_m: tf.Tensor, los_mask: tf.Tensor) -> tf.Tensor:
    """Sample frequency-dependent RMS delay spread in nanoseconds."""
    real_dtype = _real_dtype(cfg)
    fc_ghz = float(cfg.raw["system_model"]["carrier_frequency_hz"]) / 1e9

    ds_ref_los_ns = float(twc_get(cfg, ("wideband", "delay_spread_ref_ns_los"), 30.0))
    ds_ref_nlos_ns = float(twc_get(cfg, ("wideband", "delay_spread_ref_ns_nlos"), 90.0))
    sigma_ln = float(twc_get(cfg, ("wideband", "delay_spread_sigma_ln"), 0.35))
    alpha_f = float(twc_get(cfg, ("wideband", "delay_spread_freq_scaling_exponent"), 0.30))
    d_exponent = float(twc_get(cfg, ("wideband", "delay_spread_distance_exponent"), 0.10))
    d_ref_m = float(twc_get(cfg, ("wideband", "delay_spread_distance_ref_m"), 100.0))
    min_ns = float(twc_get(cfg, ("wideband", "delay_spread_min_ns"), 5.0))
    max_ns = float(twc_get(cfg, ("wideband", "delay_spread_max_ns"), 500.0))

    ds_ref = tf.where(
        los_mask,
        tf.cast(ds_ref_los_ns, real_dtype),
        tf.cast(ds_ref_nlos_ns, real_dtype),
    )
    freq_factor = tf.cast(fc_ghz ** (-alpha_f), real_dtype)
    dist_factor = tf.pow(tf.maximum(d2d_m, tf.cast(1.0, real_dtype)) / tf.cast(d_ref_m, real_dtype), tf.cast(d_exponent, real_dtype))
    log_scatter = tf.random.normal(tf.shape(d2d_m), dtype=real_dtype) * tf.cast(sigma_ln, real_dtype)
    ds_ns = ds_ref * freq_factor * dist_factor * tf.exp(log_scatter)
    return tf.clip_by_value(ds_ns, tf.cast(min_ns, real_dtype), tf.cast(max_ns, real_dtype))


def _sample_tap_structure(cfg: Any, ds_ns: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return tap delays [S,B,U,P] in seconds and normalized tap powers."""
    real_dtype = _real_dtype(cfg)
    num_taps = int(twc_get(cfg, ("wideband", "num_taps"), 8))
    max_delay_ns = float(twc_get(cfg, ("wideband", "max_delay_ns"), 500.0))

    ds_s = tf.maximum(ds_ns, tf.cast(1.0, real_dtype)) * tf.cast(1e-9, real_dtype)
    u = tf.random.uniform(tf.concat([tf.shape(ds_s), [num_taps]], axis=0), minval=1e-6, maxval=1.0, dtype=real_dtype)
    tau = -ds_s[..., None] * tf.math.log(u)
    tau = tf.minimum(tau, tf.cast(max_delay_ns * 1e-9, real_dtype))
    tau = tf.sort(tau, axis=-1)

    # Exponential power-delay profile with per-link normalization
    p = tf.exp(-tau / tf.maximum(ds_s[..., None], tf.cast(1e-12, real_dtype)))
    p = p / tf.maximum(tf.reduce_sum(p, axis=-1, keepdims=True), tf.cast(1e-12, real_dtype))
    return tau, p


def _rank1_los_component(shape_sbu: tf.Tensor, nr: int, m: int, cfg: Any) -> tf.Tensor:
    """Generate a simple rank-1 LOS matrix for each (sample, bs, user)."""
    complex_dtype = _complex_dtype(cfg)
    real_dtype = _real_dtype(cfg)

    theta_bs = tf.random.uniform(shape_sbu, minval=-0.9, maxval=0.9, dtype=real_dtype)
    theta_ue = tf.random.uniform(shape_sbu, minval=-0.9, maxval=0.9, dtype=real_dtype)

    n_bs = tf.cast(tf.range(m), real_dtype)
    n_ue = tf.cast(tf.range(nr), real_dtype)

    phase_bs = np.pi * theta_bs[..., None] * n_bs[None, None, None, :]
    phase_ue = np.pi * theta_ue[..., None] * n_ue[None, None, None, :]

    a_bs = tf.complex(tf.cos(phase_bs), tf.sin(phase_bs)) / tf.cast(tf.sqrt(tf.cast(m, real_dtype)), complex_dtype)
    a_ue = tf.complex(tf.cos(phase_ue), tf.sin(phase_ue)) / tf.cast(tf.sqrt(tf.cast(nr, real_dtype)), complex_dtype)
    return tf.einsum("sbun,sbum->sbunm", a_ue, tf.math.conj(a_bs), optimize=True)


def _build_wideband_channel(cfg: Any, topo: Any, grid: Any, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate a wideband channel tensor [S,K,B,U,Nr,M]."""
    real_dtype = _real_dtype(cfg)
    complex_dtype = _complex_dtype(cfg)

    h_flat = tf.cast(generate_ue_channels(cfg, topo, batch_size=batch_size), complex_dtype)  # [S,B,U,Nr,M]
    s = int(h_flat.shape[0]) if h_flat.shape[0] is not None else int(tf.shape(h_flat)[0].numpy())
    b = int(h_flat.shape[1]) if h_flat.shape[1] is not None else int(tf.shape(h_flat)[1].numpy())
    u = int(h_flat.shape[2]) if h_flat.shape[2] is not None else int(tf.shape(h_flat)[2].numpy())
    nr = int(h_flat.shape[3])
    m = int(h_flat.shape[4])
    k = int(grid.num_subbands)

    bs_v = tf.cast(topo.bs_virtual_loc, real_dtype)
    ut = tf.cast(topo.ut_loc, real_dtype)
    diff_xy = bs_v[:, :, :, :2] - ut[:, None, :, :2]
    d2d = tf.sqrt(tf.reduce_sum(tf.square(diff_xy), axis=-1) + 1e-12)

    p_los = umi_los_probability(d2d)
    los_mask = tf.random.uniform(tf.shape(p_los), dtype=real_dtype) < p_los

    ds_ns = _sample_delay_spread_ns(cfg, d2d_m=d2d, los_mask=los_mask)
    tau, p_tap = _sample_tap_structure(cfg, ds_ns)

    num_taps = int(p_tap.shape[-1])

    # Large-scale power of the flat baseline, averaged over spatial dimensions.
    beta = tf.reduce_mean(tf.abs(h_flat) ** 2, axis=[3, 4])  # [S,B,U]
    beta = tf.maximum(beta, tf.cast(1e-14, beta.dtype))

    g = _complex_normal((int(s), int(b), int(u), num_taps, nr, m), complex_dtype)
    rho_bs = float(twc_get(cfg, ("wideband", "spatial_correlation", "rho_bs"), 0.6))
    rho_ue = float(twc_get(cfg, ("wideband", "spatial_correlation", "rho_ue"), 0.3))
    g = _apply_spatial_correlation(g, rho_ue=rho_ue, rho_bs=rho_bs, cfg=cfg)

    h_tap = g * tf.cast(tf.sqrt(beta)[..., None, None, None], complex_dtype)
    h_tap = h_tap * tf.cast(tf.sqrt(p_tap)[..., None, None], complex_dtype)

    if bool(twc_get(cfg, ("wideband", "los", "enabled"), True)):
        k_db = float(twc_get(cfg, ("wideband", "los", "rician_k_db"), 7.0))
        k_lin = 10.0 ** (k_db / 10.0)
        h_los = _rank1_los_component(tf.shape(beta), nr=nr, m=m, cfg=cfg)  # [S,B,U,Nr,M]
        los_scale = tf.cast(tf.sqrt(k_lin / (1.0 + k_lin)), complex_dtype)
        nlos_scale = tf.cast(tf.sqrt(1.0 / (1.0 + k_lin)), complex_dtype)
        los_mask_c = tf.cast(los_mask[..., None, None], complex_dtype)
        h_tap = h_tap * nlos_scale
        h_tap_first = h_tap[:, :, :, 0, :, :]
        h_tap_first = h_tap_first + los_mask_c * los_scale * h_los * tf.cast(tf.sqrt(beta)[..., None, None], complex_dtype)
        h_tap = tf.concat([h_tap_first[:, :, :, None, :, :], h_tap[:, :, :, 1:, :, :]], axis=3)

    freq_offsets = tf.cast(grid.subband_center_hz - grid.carrier_frequency_hz, real_dtype)  # [K]
    phase = tf.exp(
        tf.complex(
            tf.zeros(tf.concat([tf.shape(tau), [k]], axis=0), dtype=real_dtype),
            -2.0 * np.pi * tau[..., None] * freq_offsets[None, None, None, None, :],
        )
    )  # [S,B,U,P,K]

    h_wb = tf.einsum("sbupnm,sbupk->skbunm", h_tap, phase, optimize=True)
    return h_wb, ds_ns, tf.cast(los_mask, tf.bool)


def _extract_effective_miso(h_wb: tf.Tensor, cfg: Any) -> tf.Tensor:
    """Collapse the Nr x M channel to an effective 1 x M channel via dominant left singular vector."""
    b = int(h_wb.shape[2])
    u_per_bs = int(cfg.derived["u_per_bs"])

    # Self channels H[s,k,u,nr,m] for the users served by each BS
    pieces = []
    for bs_idx in range(b):
        start = bs_idx * u_per_bs
        end = (bs_idx + 1) * u_per_bs
        pieces.append(h_wb[:, :, bs_idx, start:end, :, :])
    h_self = tf.concat(pieces, axis=2)  # [S,K,U,Nr,M]

    # Dominant left singular vector of the serving channel
    _, u_rx, _ = tf.linalg.svd(h_self, full_matrices=False, compute_uv=True)
    combiner = tf.math.conj(u_rx[..., 0])  # [S,K,U,Nr]

    # Apply the same combiner to desired and interfering channels for each user
    h_eff = tf.einsum("skun,skbunm->skbum", combiner, h_wb, optimize=True)  # [S,K,B,U,M]
    return h_eff


def _build_user_weights(cfg: Any, h_eff: tf.Tensor, topo: Any, profile: str) -> tf.Tensor:
    """Construct a user-weight vector for sensitivity analysis."""
    real_dtype = tf.float64 if h_eff.dtype == tf.complex128 else tf.float32
    b = int(h_eff.shape[2])
    u = int(h_eff.shape[3])
    u_per_bs = int(cfg.derived["u_per_bs"])
    w = tf.ones([u], dtype=real_dtype)

    profile = str(profile).lower().strip()
    if profile in ("uniform", "ones", "default"):
        return w

    # Serving-link average gain proxy
    gains = []
    for bs_idx in range(b):
        start = bs_idx * u_per_bs
        end = (bs_idx + 1) * u_per_bs
        h_serv = h_eff[:, :, bs_idx, start:end, :]
        gain = tf.reduce_mean(tf.reduce_sum(tf.abs(h_serv) ** 2, axis=-1), axis=[0, 1])
        gains.append(gain)
    g = tf.concat(gains, axis=0) + 1e-9

    if profile in ("inverse_serving_gain", "cell_edge", "fairness"):
        w = 1.0 / tf.cast(g, real_dtype)
    elif profile in ("hotspot_priority", "critical", "fs_sensitive"):
        # Prioritize users with stronger average overlap risk proxy:
        bs_v = tf.cast(topo.bs_virtual_loc, real_dtype)
        ut = tf.cast(topo.ut_loc, real_dtype)
        diff_xy = bs_v[:, :, :, :2] - ut[:, None, :, :2]
        d2d = tf.sqrt(tf.reduce_sum(tf.square(diff_xy), axis=-1) + 1e-12)
        nearest = tf.reduce_min(d2d, axis=1)
        prox = 1.0 / (nearest + 1.0)
        w = tf.reduce_mean(prox, axis=0)
    elif profile in ("lognormal", "random"):
        w = tf.exp(tf.random.normal([u], stddev=0.5, dtype=real_dtype))
    else:
        return tf.ones([u], dtype=real_dtype)

    w = w / tf.maximum(tf.reduce_mean(w), tf.cast(1e-9, real_dtype))
    return w


def build_wideband_batch(cfg: Any, batch_size: int, user_weight_profile: str = "uniform") -> WidebandBatch:
    """Generate one wideband coexistence mini-batch."""
    grid = build_nr_grid(cfg)
    topo = generate_hexgrid_topology(cfg, batch_size=batch_size)
    fs_loc = generate_fixed_service_locations(cfg, topo, batch_size=batch_size)
    fs_stats = generate_fs_stats(cfg, topo, fs_loc, batch_size=batch_size)

    h_wb, ds_ns, los_mask = _build_wideband_channel(cfg, topo, grid, batch_size=batch_size)
    h_eff = _extract_effective_miso(h_wb, cfg)

    # Risk score from actual ISED overlap and per-FS spatial coupling
    epsilon = tf.cast(fs_stats.epsilon, ds_ns.dtype)               # [K,L]
    i_max = tf.cast(fs_stats.i_max_watt, ds_ns.dtype)[None, None, :] + 1e-15
    bar_beta = tf.cast(fs_stats.bar_beta, ds_ns.dtype)             # [S,B,L]
    norm_coupling = tf.reduce_max(bar_beta / i_max, axis=1)        # [S,L]
    risk_score = tf.einsum("kl,sl->sk", epsilon, norm_coupling, optimize=True)
    risk_score = risk_score / tf.maximum(tf.reduce_max(risk_score, axis=1, keepdims=True), 1e-12)

    gate_thr = float(twc_get(cfg, ("coexistence", "static_notch_threshold"), 0.55))
    static_gate = tf.cast(risk_score <= tf.cast(gate_thr, risk_score.dtype), risk_score.dtype)

    mean_ds = tf.reduce_mean(ds_ns, axis=[1, 2])
    max_risk = tf.reduce_max(risk_score, axis=1)
    mean_risk = tf.reduce_mean(risk_score, axis=1)
    frac_blocked = 1.0 - tf.reduce_mean(static_gate, axis=1)
    mean_los = tf.reduce_mean(tf.cast(los_mask, mean_ds.dtype), axis=[1, 2])

    scenario_features = tf.stack([mean_ds, mean_risk, max_risk, frac_blocked, mean_los], axis=1)

    user_weights = _build_user_weights(cfg, h_eff, topo, user_weight_profile)

    return WidebandBatch(
        topo=topo,
        fs_loc=fs_loc,
        fs_stats=fs_stats,
        grid=grid,
        h_wb=h_wb,
        h_eff=h_eff,
        serving_bs=np.asarray(topo.serving_bs, dtype=np.int32),
        user_weights=user_weights,
        noise_var_watt=float(cfg.derived["ue_noise_re_watt"]),
        bs_power_budget_watt=float(cfg.derived.get("bs_power_budget_sim_watt", cfg.derived["bs_total_tx_power_watt"])),
        risk_score=risk_score,
        overlap=tf.cast(fs_stats.epsilon, risk_score.dtype),
        static_gate=static_gate,
        delay_spread_ns=ds_ns,
        los_mask=los_mask,
        scenario_features=scenario_features,
        metadata={
            "num_bs": int(cfg.derived["num_bs"]),
            "num_ut": int(cfg.derived["num_ut"]),
            "u_per_bs": int(cfg.derived["u_per_bs"]),
            "num_subbands": int(grid.num_subbands),
            "re_scaling": float(cfg.derived.get('re_scaling', 1.0)),
            "bs_power_budget_watt": float(cfg.derived.get("bs_power_budget_sim_watt", cfg.derived["bs_total_tx_power_watt"])),
            "static_notch_threshold": float(twc_get(cfg, ("coexistence", "static_notch_threshold"), 0.55)),
            "base_fs_in_target_db": float(cfg.raw["fixed_service"]["in_target_db"]),
        },
    )
