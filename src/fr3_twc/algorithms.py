"""Baseline wideband algorithms used by the TWC pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .config_utils import twc_get
from .types import AlgorithmResult, WidebandBatch


def _real_dtype_from_complex(dtype: tf.DType) -> tf.DType:
    return tf.float64 if dtype == tf.complex128 else tf.float32


def _scale_complex_by_real(x: tf.Tensor, coeff: tf.Tensor) -> tf.Tensor:
    """Scale a complex tensor by a real coefficient without real<->complex gradient casts."""
    real_dtype = _real_dtype_from_complex(x.dtype)
    coeff = tf.cast(coeff, real_dtype)
    re = coeff * tf.math.real(x)
    im = coeff * tf.math.imag(x)
    return tf.complex(re, im)


def project_per_bs_power(w: tf.Tensor, p_tot_watt: float) -> tf.Tensor:
    """Project to the per-BS total power budget."""
    real_dtype = _real_dtype_from_complex(w.dtype)
    p = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 3, 4])  # [S,B]
    scale = tf.minimum(
        tf.cast(1.0, real_dtype),
        tf.sqrt(tf.cast(p_tot_watt, real_dtype) / tf.maximum(p, tf.cast(1e-12, real_dtype))),
    )  # [S,B]
    return _scale_complex_by_real(w, scale[:, None, :, None, None])


def initial_mf_precoder(batch: WidebandBatch, use_gate: bool = True) -> tf.Tensor:
    """Matched-filter initialization, one stream per served user."""
    h_eff = batch.h_eff
    s = int(h_eff.shape[0])
    k = int(h_eff.shape[1])
    b = int(h_eff.shape[2])
    m = int(h_eff.shape[4])
    u_per_bs = int(batch.metadata["u_per_bs"])

    w_parts = []
    for bs_idx in range(b):
        start = bs_idx * u_per_bs
        end = (bs_idx + 1) * u_per_bs
        h_self = h_eff[:, :, bs_idx, start:end, :]                # [S,K,U_b,M]
        w_b = tf.math.conj(tf.transpose(h_self, [0, 1, 3, 2]))    # [S,K,M,U_b]
        norm = tf.sqrt(tf.reduce_sum(tf.abs(w_b) ** 2, axis=2, keepdims=True) + 1e-12)
        w_b = _scale_complex_by_real(w_b, tf.math.reciprocal(norm))
        w_parts.append(w_b)

    w = tf.stack(w_parts, axis=2)  # [S,K,B,M,U_b]
    if use_gate:
        w = _scale_complex_by_real(w, batch.static_gate[:, :, None, None, None])
    return project_per_bs_power(w, batch.bs_power_budget_watt)


def user_rate_tensors(batch: WidebandBatch, w: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Compute user-wise and sample-wise rate/SINR tensors."""
    h_eff = batch.h_eff
    real_dtype = _real_dtype_from_complex(w.dtype)
    b = int(h_eff.shape[2])
    u_per_bs = int(batch.metadata["u_per_bs"])

    z = tf.einsum("skbum,skbmq->skbuq", tf.math.conj(h_eff), w, optimize=True)  # [S,K,B,U,Q]
    total = tf.reduce_sum(tf.abs(z) ** 2, axis=[2, 4])                           # [S,K,U]

    desired_parts = []
    for bs_idx in range(b):
        start = bs_idx * u_per_bs
        end = (bs_idx + 1) * u_per_bs
        local_block = z[:, :, bs_idx, start:end, :]                              # [S,K,Q,Q]
        desired_parts.append(tf.abs(tf.linalg.diag_part(local_block)) ** 2)
    desired = tf.concat(desired_parts, axis=2)                                   # [S,K,U]

    interf = tf.maximum(total - desired, tf.cast(0.0, real_dtype))
    noise = tf.cast(batch.noise_var_watt, real_dtype)
    sinr = desired / tf.maximum(interf + noise, tf.cast(1e-15, real_dtype))
    rate = tf.math.log1p(sinr) / tf.math.log(tf.cast(2.0, real_dtype))
    return {
        "sinr": sinr,
        "rate": rate,
        "desired": desired,
        "interference": interf,
    }


def fs_interference(batch: WidebandBatch, w: tf.Tensor) -> tf.Tensor:
    """Compute per-sample per-FS interference [S,L]."""
    fs = batch.fs_stats
    real_dtype = _real_dtype_from_complex(w.dtype)

    epsilon = tf.cast(fs.epsilon, real_dtype)                                   # [K,L]
    bar_beta = tf.cast(fs.bar_beta, real_dtype)                                 # [S,B,L]
    re_scaling = tf.cast(float(batch.metadata.get("re_scaling", 1.0)), real_dtype)

    if fs.a_bs_fs is not None:
        a = tf.cast(fs.a_bs_fs, w.dtype)                                         # [S,B,L,M]
        proj = tf.einsum("sblm,skbmq->skblq", tf.math.conj(a), w, optimize=True)
        pdir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)                         # [S,K,B,L]
    else:
        pdir = tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])[:, :, :, None]         # [S,K,B,1]

    delta = tf.cast(tf.transpose(fs.delta, [1, 0]), real_dtype)                  # [K,B]
    pdir = pdir * delta[None, :, :, None]

    i_fs = re_scaling * tf.einsum("skbl,kl,sbl->sl", pdir, epsilon, bar_beta, optimize=True)
    return i_fs


def empirical_cvar(excess: tf.Tensor, alpha: float = 0.95) -> tf.Tensor:
    """Empirical CVaR of the positive excess values."""
    real_dtype = excess.dtype
    flat = tf.reshape(tf.nn.relu(excess), [-1])
    n = tf.size(flat)
    k = tf.maximum(1, tf.cast(tf.math.ceil((1.0 - tf.cast(alpha, real_dtype)) * tf.cast(n, real_dtype)), tf.int32))
    top = tf.math.top_k(flat, k=k, sorted=False).values
    return tf.reduce_mean(top)


def objective_terms(
    batch: WidebandBatch,
    w: tf.Tensor,
    fs_weight: float,
    cvar_weight: float,
    power_weight: float,
    alpha_cvar: float,
) -> Dict[str, tf.Tensor]:
    """Compute scalar objective terms used by the fixed and unfolded solvers."""
    rr = user_rate_tensors(batch, w)
    rate = rr["rate"]

    user_weights = tf.cast(batch.user_weights[None, None, :], rate.dtype)
    weighted_sum_rate = tf.reduce_mean(tf.reduce_sum(rate * user_weights, axis=2))  # average over samples and subbands

    i_fs = fs_interference(batch, w)
    i_max = tf.cast(batch.fs_stats.i_max_watt, i_fs.dtype)[None, :] + 1e-15
    excess = i_fs / i_max - 1.0
    mean_violation = tf.reduce_mean(tf.nn.relu(excess))
    cvar = empirical_cvar(excess, alpha=alpha_cvar)

    pow_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 3, 4])
    power_violation = tf.reduce_mean(tf.nn.relu(pow_b / tf.cast(batch.bs_power_budget_watt, pow_b.dtype) - 1.0))

    utility = weighted_sum_rate - tf.cast(fs_weight, rate.dtype) * mean_violation - tf.cast(cvar_weight, rate.dtype) * cvar - tf.cast(power_weight, rate.dtype) * power_violation
    loss = -utility
    return {
        "utility": utility,
        "loss": loss,
        "weighted_sum_rate": weighted_sum_rate,
        "mean_violation": mean_violation,
        "cvar": cvar,
        "power_violation": power_violation,
        "i_fs": i_fs,
        "excess": excess,
        "rate": rate,
        "sinr": rr["sinr"],
    }


def fixed_pgd(
    batch: WidebandBatch,
    cfg: Any,
    name: str,
    steps: int,
    step_size: float,
    damping: float,
    fs_weight: float,
    cvar_weight: float,
    power_weight: float,
    alpha_cvar: float,
    use_gate: bool,
) -> AlgorithmResult:
    """Fixed-parameter projected-gradient baseline."""
    start = time.perf_counter()
    w = initial_mf_precoder(batch, use_gate=use_gate)

    hist: Dict[str, List[float]] = {"utility": [], "mean_violation": [], "cvar": [], "power_violation": []}

    for _ in range(int(steps)):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(w)
            terms = objective_terms(
                batch=batch,
                w=w,
                fs_weight=fs_weight,
                cvar_weight=cvar_weight,
                power_weight=power_weight,
                alpha_cvar=alpha_cvar,
            )
        grad = tf.stop_gradient(tape.gradient(terms["loss"], w))
        real_dtype = _real_dtype_from_complex(w.dtype)
        w_new = w - _scale_complex_by_real(grad, tf.cast(step_size, real_dtype))
        if use_gate:
            w_new = _scale_complex_by_real(w_new, batch.static_gate[:, :, None, None, None])
        w_new = project_per_bs_power(w_new, batch.bs_power_budget_watt)
        w = _scale_complex_by_real(w_new, tf.cast(damping, real_dtype)) + _scale_complex_by_real(w, tf.cast(1.0 - damping, real_dtype))

        hist["utility"].append(float(terms["utility"].numpy()))
        hist["mean_violation"].append(float(terms["mean_violation"].numpy()))
        hist["cvar"].append(float(terms["cvar"].numpy()))
        hist["power_violation"].append(float(terms["power_violation"].numpy()))

    runtime_s = time.perf_counter() - start
    return AlgorithmResult(name=name, w=w, runtime_s=runtime_s, history=hist)


def static_notch_mf(batch: WidebandBatch) -> AlgorithmResult:
    """Pure notch + matched-filter baseline."""
    start = time.perf_counter()
    w = initial_mf_precoder(batch, use_gate=True)
    runtime_s = time.perf_counter() - start
    return AlgorithmResult(name="static_notch_mf", w=w, runtime_s=runtime_s, history={})


def wideband_pgd_baseline(batch: WidebandBatch, cfg: Any) -> AlgorithmResult:
    return fixed_pgd(
        batch=batch,
        cfg=cfg,
        name="wideband_pgd",
        steps=int(twc_get(cfg, ("algorithm", "pgd_steps"), 10)),
        step_size=float(twc_get(cfg, ("algorithm", "fixed_step_size"), 0.15)),
        damping=float(twc_get(cfg, ("algorithm", "fixed_damping"), 0.75)),
        fs_weight=float(twc_get(cfg, ("algorithm", "fixed_fs_weight"), 15.0)),
        cvar_weight=float(twc_get(cfg, ("algorithm", "fixed_cvar_weight"), 8.0)),
        power_weight=float(twc_get(cfg, ("algorithm", "power_weight"), 20.0)),
        alpha_cvar=float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95)),
        use_gate=True,
    )


def risk_neutral_pgd(batch: WidebandBatch, cfg: Any) -> AlgorithmResult:
    return fixed_pgd(
        batch=batch,
        cfg=cfg,
        name="risk_neutral_pgd",
        steps=int(twc_get(cfg, ("algorithm", "pgd_steps"), 10)),
        step_size=float(twc_get(cfg, ("algorithm", "fixed_step_size"), 0.15)),
        damping=float(twc_get(cfg, ("algorithm", "fixed_damping"), 0.75)),
        fs_weight=0.0,
        cvar_weight=0.0,
        power_weight=float(twc_get(cfg, ("algorithm", "power_weight"), 20.0)),
        alpha_cvar=float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95)),
        use_gate=False,
    )
