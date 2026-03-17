"""Baseline, repair, and feasibility-restoration algorithms for the TWC pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

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
    """Project to the per-BS total power budget over the simulated band."""
    real_dtype = _real_dtype_from_complex(w.dtype)
    p = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 3, 4])  # [S,B]
    scale = tf.minimum(
        tf.cast(1.0, real_dtype),
        tf.sqrt(tf.cast(p_tot_watt, real_dtype) / tf.maximum(p, tf.cast(1e-12, real_dtype))),
    )
    return _scale_complex_by_real(w, scale[:, None, :, None, None])


def subband_power_map(w: tf.Tensor) -> tf.Tensor:
    """Per-sample, per-subband, per-BS transmit power."""
    return tf.reduce_sum(tf.abs(w) ** 2, axis=[3, 4])  # [S,K,B]


def initial_mf_precoder(batch: WidebandBatch, use_gate: bool = True) -> tf.Tensor:
    """Matched-filter initialization, one stream per served user."""
    h_eff = batch.h_eff
    b = int(h_eff.shape[2])
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


def fs_interference_components(batch: WidebandBatch, w: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Detailed FS interference components.

    Returns
    -------
    dict with keys:
      pdir: directional power [S,K,B,L]
      contrib_s_k_l: subband-to-FS interference contribution [S,K,L]
      i_fs: total FS interference [S,L]
    """
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

    contrib_s_k_b_l = re_scaling * pdir * epsilon[None, :, None, :] * bar_beta[:, None, :, :]
    contrib_s_k_l = tf.reduce_sum(contrib_s_k_b_l, axis=2)
    i_fs = tf.reduce_sum(contrib_s_k_l, axis=1)
    return {"pdir": pdir, "contrib_s_k_l": contrib_s_k_l, "i_fs": i_fs}


def fs_interference(batch: WidebandBatch, w: tf.Tensor) -> tf.Tensor:
    """Compute per-sample per-FS interference [S,L]."""
    return fs_interference_components(batch, w)["i_fs"]


def empirical_cvar(excess: tf.Tensor, alpha: float = 0.95) -> tf.Tensor:
    """Empirical CVaR of the positive excess values."""
    real_dtype = excess.dtype
    flat = tf.reshape(tf.nn.relu(excess), [-1])
    n = tf.size(flat)
    k = tf.maximum(1, tf.cast(tf.math.ceil((1.0 - tf.cast(alpha, real_dtype)) * tf.cast(n, real_dtype)), tf.int32))
    top = tf.math.top_k(flat, k=k, sorted=False).values
    return tf.reduce_mean(top)


def soft_outage_surrogate(excess: tf.Tensor, beta: float = 10.0, slack: float = 0.0) -> tf.Tensor:
    """Smooth outage surrogate based on the worst FS excess per sample."""
    real_dtype = excess.dtype
    max_excess = tf.reduce_max(excess, axis=1)
    return tf.reduce_mean(
        tf.sigmoid(tf.cast(beta, real_dtype) * (max_excess - tf.cast(slack, real_dtype)))
    )


def tail_excess_penalty(excess: tf.Tensor, slack: float = 0.0) -> tf.Tensor:
    """Quadratic penalty on positive FS excess above a slack level."""
    real_dtype = excess.dtype
    tail = tf.nn.relu(excess - tf.cast(slack, real_dtype))
    return tf.reduce_mean(tail ** 2)


def objective_terms(
    batch: WidebandBatch,
    w: tf.Tensor,
    fs_weight: float,
    cvar_weight: float,
    power_weight: float,
    alpha_cvar: float,
) -> Dict[str, tf.Tensor]:
    """Compute scalar objective terms used by fixed and legacy unfolded solvers."""
    rr = user_rate_tensors(batch, w)
    rate = rr["rate"]

    user_weights = tf.cast(batch.user_weights[None, None, :], rate.dtype)
    weighted_sum_rate = tf.reduce_mean(tf.reduce_sum(rate * user_weights, axis=2))

    i_fs = fs_interference(batch, w)
    i_max = tf.cast(batch.fs_stats.i_max_watt, i_fs.dtype)[None, :] + tf.cast(1e-15, i_fs.dtype)
    excess = i_fs / i_max - 1.0
    mean_violation = tf.reduce_mean(tf.nn.relu(excess))
    cvar = empirical_cvar(excess, alpha=alpha_cvar)

    power_map = subband_power_map(w)
    pow_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 3, 4])
    power_violation = tf.reduce_mean(tf.nn.relu(pow_b / tf.cast(batch.bs_power_budget_watt, pow_b.dtype) - 1.0))

    utility = (
        weighted_sum_rate
        - tf.cast(fs_weight, rate.dtype) * mean_violation
        - tf.cast(cvar_weight, rate.dtype) * cvar
        - tf.cast(power_weight, rate.dtype) * power_violation
    )
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
        "desired": rr["desired"],
        "power_map": power_map,
    }


def initial_dual_from_batch(batch: WidebandBatch, dual_init_scale: float = 1.0) -> tf.Tensor:
    """Initialize per-FS dual variables from normalized geometry/coupling risk."""
    real_dtype = batch.delay_spread_ns.dtype
    i_max = tf.cast(batch.fs_stats.i_max_watt, real_dtype)[None, :] + tf.cast(1e-15, real_dtype)
    prior = tf.reduce_max(tf.cast(batch.fs_stats.bar_beta, real_dtype) / i_max[:, None, :], axis=1)  # [S,L]
    prior = prior / tf.maximum(tf.reduce_mean(prior, axis=1, keepdims=True), tf.cast(1e-12, real_dtype))
    return tf.cast(dual_init_scale, real_dtype) * prior


def primal_dual_terms(
    batch: WidebandBatch,
    w: tf.Tensor,
    lam: tf.Tensor,
    cvar_weight: float,
    power_weight: float,
    alpha_cvar: float,
    violation_clip: float | None = None,
) -> Dict[str, tf.Tensor]:
    """Lagrangian terms for explicit FS-budget primal-dual optimization."""
    rr = user_rate_tensors(batch, w)
    rate = rr["rate"]
    real_dtype = rate.dtype

    user_weights = tf.cast(batch.user_weights[None, None, :], real_dtype)
    weighted_sum_rate = tf.reduce_mean(tf.reduce_sum(rate * user_weights, axis=2))

    i_fs = fs_interference(batch, w)
    i_max = tf.cast(batch.fs_stats.i_max_watt, i_fs.dtype)[None, :] + tf.cast(1e-15, i_fs.dtype)
    excess = i_fs / i_max - 1.0
    if violation_clip is not None:
        clip = tf.cast(float(violation_clip), i_fs.dtype)
        excess_for_dual = tf.clip_by_value(excess, -clip, clip)
    else:
        excess_for_dual = excess

    lam = tf.cast(lam, i_fs.dtype)
    dual_penalty = tf.reduce_mean(tf.reduce_sum(lam * excess_for_dual, axis=1))
    mean_violation = tf.reduce_mean(tf.nn.relu(excess))
    cvar = empirical_cvar(excess, alpha=alpha_cvar)

    power_map = subband_power_map(w)
    pow_b = tf.reduce_sum(tf.abs(w) ** 2, axis=[1, 3, 4])
    power_violation = tf.reduce_mean(tf.nn.relu(pow_b / tf.cast(batch.bs_power_budget_watt, pow_b.dtype) - 1.0))

    utility = (
        weighted_sum_rate
        - dual_penalty
        - tf.cast(cvar_weight, real_dtype) * cvar
        - tf.cast(power_weight, real_dtype) * power_violation
    )
    loss = -utility
    return {
        "utility": utility,
        "loss": loss,
        "weighted_sum_rate": weighted_sum_rate,
        "dual_penalty": dual_penalty,
        "mean_violation": mean_violation,
        "cvar": cvar,
        "power_violation": power_violation,
        "i_fs": i_fs,
        "excess": excess,
        "excess_for_dual": excess_for_dual,
        "rate": rate,
        "sinr": rr["sinr"],
        "desired": rr["desired"],
        "power_map": power_map,
    }


def _append_hist(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out = {k: list(v) for k, v in dst.items()}
    for key, vals in src.items():
        out.setdefault(key, [])
        out[key].extend(list(vals))
    return out


def _soft_gate_from_batch(batch: WidebandBatch, cfg: Any) -> tf.Tensor:
    gate_temp = float(twc_get(cfg, ("coexistence", "soft_gate_temperature"), 8.0))
    gate_bias = float(twc_get(cfg, ("coexistence", "repair_gate_bias"), 0.15))
    real_dtype = batch.risk_score.dtype
    return tf.sigmoid(
        tf.cast(gate_bias, real_dtype) - tf.cast(gate_temp, real_dtype) * tf.cast(batch.risk_score, real_dtype)
    )


def dual_calibration_refine(
    batch: WidebandBatch,
    w: tf.Tensor,
    lam: tf.Tensor,
    cfg: Any,
    use_gate: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, List[float]]]:
    """Short inference-time dual calibration loop starting from a student or PD solution."""
    steps = int(twc_get(cfg, ("algorithm", "pd_calibration_steps"), 0))
    if steps <= 0:
        return w, lam, {
            "utility": [],
            "mean_violation": [],
            "cvar": [],
            "power_violation": [],
            "dual_mean": [],
            "dual_max": [],
        }

    real_dtype = _real_dtype_from_complex(w.dtype)
    primal_step = float(twc_get(cfg, ("algorithm", "pd_calibration_primal_step"), 0.035))
    dual_step = float(twc_get(cfg, ("algorithm", "pd_calibration_dual_step"), 0.75))
    damping = float(twc_get(cfg, ("algorithm", "pd_calibration_damping"), 0.88))
    cvar_weight = float(twc_get(cfg, ("algorithm", "pd_calibration_cvar_weight"), twc_get(cfg, ("algorithm", "pd_cvar_weight"), 1.5)))
    power_weight = float(twc_get(cfg, ("algorithm", "power_weight"), 20.0))
    alpha_cvar = float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95))
    violation_clip = float(twc_get(cfg, ("algorithm", "pd_violation_clip"), 5.0))
    dual_boost = float(twc_get(cfg, ("algorithm", "pd_calibration_dual_boost"), 1.25))
    gate = _soft_gate_from_batch(batch, cfg) if bool(twc_get(cfg, ("coexistence", "use_soft_risk_gate"), True)) else batch.static_gate

    hist: Dict[str, List[float]] = {
        "utility": [],
        "mean_violation": [],
        "cvar": [],
        "power_violation": [],
        "dual_mean": [],
        "dual_max": [],
    }
    one = tf.cast(1.0, real_dtype)

    for _ in range(steps):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(w)
            terms = primal_dual_terms(
                batch=batch,
                w=w,
                lam=lam,
                cvar_weight=cvar_weight,
                power_weight=power_weight,
                alpha_cvar=alpha_cvar,
                violation_clip=violation_clip,
            )
        grad = tf.stop_gradient(tape.gradient(terms["loss"], w))
        w_new = w - _scale_complex_by_real(grad, tf.cast(primal_step, real_dtype))
        if use_gate:
            w_new = _scale_complex_by_real(w_new, gate[:, :, None, None, None])
        w_new = project_per_bs_power(w_new, batch.bs_power_budget_watt)
        w = _scale_complex_by_real(w_new, tf.cast(damping, real_dtype)) + _scale_complex_by_real(w, one - tf.cast(damping, real_dtype))

        dual_err = tf.clip_by_value(
            tf.stop_gradient(terms["excess"]),
            tf.cast(-1.0, lam.dtype),
            tf.cast(violation_clip, lam.dtype),
        )
        lam = tf.nn.relu(lam + tf.cast(dual_step * dual_boost, lam.dtype) * dual_err)

        hist["utility"].append(float(terms["utility"].numpy()))
        hist["mean_violation"].append(float(terms["mean_violation"].numpy()))
        hist["cvar"].append(float(terms["cvar"].numpy()))
        hist["power_violation"].append(float(terms["power_violation"].numpy()))
        hist["dual_mean"].append(float(tf.reduce_mean(lam).numpy()))
        hist["dual_max"].append(float(tf.reduce_max(lam).numpy()))

    return w, lam, hist


def repair_fs_feasibility(
    batch: WidebandBatch,
    w: tf.Tensor,
    cfg: Any,
    lam: tf.Tensor | None = None,
) -> Tuple[tf.Tensor, Dict[str, float]]:
    """Selective subband repair followed by a guaranteed global fallback.

    The selective stage shrinks only risky subbands. If any violation remains,
    the fallback applies one global power shrink that guarantees FS feasibility
    because interference is linear in the precoder power scaling.
    """
    real_dtype = _real_dtype_from_complex(w.dtype)
    repair_iters = int(twc_get(cfg, ("algorithm", "repair_iters"), 8))
    repair_eta = float(twc_get(cfg, ("algorithm", "repair_eta"), 1.5))
    alpha_min = float(twc_get(cfg, ("algorithm", "repair_alpha_min"), 0.05))
    tol = float(twc_get(cfg, ("algorithm", "repair_violation_tol"), 0.01))
    use_lambda_weight = bool(twc_get(cfg, ("algorithm", "repair_use_lambda_weight"), True))
    global_margin = float(twc_get(cfg, ("algorithm", "repair_global_margin"), 0.995))
    guarantee = bool(twc_get(cfg, ("algorithm", "repair_guarantee_feasibility"), True))

    comp = fs_interference_components(batch, w)
    contrib = tf.cast(comp["contrib_s_k_l"], real_dtype)                        # [S,K,L]
    i_max = tf.cast(batch.fs_stats.i_max_watt, real_dtype)[None, :] + tf.cast(1e-15, real_dtype)

    s = int(contrib.shape[0])
    k = int(contrib.shape[1])
    alpha = tf.ones([s, k], dtype=real_dtype)

    lam_weight = None
    if lam is not None and use_lambda_weight:
        lam_weight = tf.cast(lam, real_dtype)
        lam_weight = lam_weight / tf.maximum(tf.reduce_mean(lam_weight, axis=1, keepdims=True), tf.cast(1e-9, real_dtype))

    actual_iters = 0
    for it in range(repair_iters):
        i_est = tf.einsum("sk,skl->sl", alpha, contrib, optimize=True)
        excess = i_est / i_max - tf.cast(1.0, real_dtype)
        worst = tf.reduce_max(excess, axis=1)
        if float(tf.reduce_max(worst).numpy()) <= tol:
            break

        score_l = tf.nn.relu(excess)
        if lam_weight is not None:
            score_l = score_l * (tf.cast(1.0, real_dtype) + lam_weight)
        score_k = tf.einsum("sl,skl->sk", score_l / i_max, contrib, optimize=True)
        score_k = score_k / tf.maximum(tf.reduce_mean(score_k, axis=1, keepdims=True), tf.cast(1e-9, real_dtype))
        alpha = tf.clip_by_value(alpha * tf.exp(-tf.cast(repair_eta, real_dtype) * score_k), tf.cast(alpha_min, real_dtype), tf.cast(1.0, real_dtype))
        actual_iters = it + 1

    w_rep = _scale_complex_by_real(w, tf.sqrt(alpha)[:, :, None, None, None])
    w_rep = project_per_bs_power(w_rep, batch.bs_power_budget_watt)

    i_after = fs_interference(batch, w_rep)
    worst_after = tf.reduce_max(i_after / i_max - tf.cast(1.0, real_dtype), axis=1)
    used_global = tf.zeros_like(worst_after, dtype=tf.bool)
    gamma = tf.ones_like(worst_after, dtype=real_dtype)

    if guarantee:
        need_global = worst_after > tf.cast(tol, real_dtype)
        if bool(tf.reduce_any(need_global).numpy()):
            ratio = tf.cast(global_margin, real_dtype) * i_max / tf.maximum(i_after, tf.cast(1e-15, real_dtype))
            gamma_needed = tf.minimum(tf.cast(1.0, real_dtype), tf.reduce_min(ratio, axis=1))
            gamma = tf.where(need_global, gamma_needed, tf.cast(1.0, real_dtype))
            w_rep = _scale_complex_by_real(w_rep, tf.sqrt(tf.maximum(gamma, tf.cast(1e-8, real_dtype)))[:, None, None, None, None])
            used_global = need_global
            i_after = fs_interference(batch, w_rep)
            worst_after = tf.reduce_max(i_after / i_max - tf.cast(1.0, real_dtype), axis=1)

    stats = {
        "repair_selective_iters_mean": float(actual_iters),
        "repair_alpha_mean": float(tf.reduce_mean(alpha).numpy()),
        "repair_alpha_min": float(tf.reduce_min(alpha).numpy()),
        "repair_global_used_fraction": float(tf.reduce_mean(tf.cast(used_global, real_dtype)).numpy()),
        "repair_global_scale_mean": float(tf.reduce_mean(gamma).numpy()),
        "repair_worst_excess_mean": float(tf.reduce_mean(worst_after).numpy()),
    }
    return w_rep, stats


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


def budgeted_primal_dual_pgd(batch: WidebandBatch, cfg: Any) -> AlgorithmResult:
    """Explicit primal-dual baseline with per-FS dual variables."""
    start = time.perf_counter()
    real_dtype = batch.delay_spread_ns.dtype
    steps = int(twc_get(cfg, ("algorithm", "pd_steps"), twc_get(cfg, ("algorithm", "pgd_steps"), 10)))
    primal_step = float(twc_get(cfg, ("algorithm", "pd_primal_step_size"), 0.08))
    dual_step = float(twc_get(cfg, ("algorithm", "pd_dual_step_size"), 0.40))
    damping = float(twc_get(cfg, ("algorithm", "pd_damping"), twc_get(cfg, ("algorithm", "fixed_damping"), 0.75)))
    cvar_weight = float(twc_get(cfg, ("algorithm", "pd_cvar_weight"), 2.0))
    power_weight = float(twc_get(cfg, ("algorithm", "power_weight"), 20.0))
    alpha_cvar = float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95))
    dual_init_scale = float(twc_get(cfg, ("algorithm", "pd_dual_init_scale"), 0.6))
    violation_clip = float(twc_get(cfg, ("algorithm", "pd_violation_clip"), 5.0))
    use_gate = bool(twc_get(cfg, ("coexistence", "use_soft_risk_gate"), True))

    w = initial_mf_precoder(batch, use_gate=True)
    lam = initial_dual_from_batch(batch, dual_init_scale=dual_init_scale)

    hist: Dict[str, List[float]] = {
        "utility": [],
        "mean_violation": [],
        "cvar": [],
        "power_violation": [],
        "dual_mean": [],
        "dual_max": [],
    }

    for _ in range(steps):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(w)
            terms = primal_dual_terms(
                batch=batch,
                w=w,
                lam=lam,
                cvar_weight=cvar_weight,
                power_weight=power_weight,
                alpha_cvar=alpha_cvar,
                violation_clip=violation_clip,
            )
        grad = tf.stop_gradient(tape.gradient(terms["loss"], w))
        w_new = w - _scale_complex_by_real(grad, tf.cast(primal_step, real_dtype))
        if use_gate:
            w_new = _scale_complex_by_real(w_new, batch.static_gate[:, :, None, None, None])
        w_new = project_per_bs_power(w_new, batch.bs_power_budget_watt)
        w = _scale_complex_by_real(w_new, tf.cast(damping, real_dtype)) + _scale_complex_by_real(w, tf.cast(1.0 - damping, real_dtype))

        dual_err = tf.clip_by_value(
            tf.stop_gradient(terms["excess"]),
            tf.cast(-1.0, lam.dtype),
            tf.cast(violation_clip, lam.dtype),
        )
        lam = tf.nn.relu(lam + tf.cast(dual_step, lam.dtype) * dual_err)

        hist["utility"].append(float(terms["utility"].numpy()))
        hist["mean_violation"].append(float(terms["mean_violation"].numpy()))
        hist["cvar"].append(float(terms["cvar"].numpy()))
        hist["power_violation"].append(float(terms["power_violation"].numpy()))
        hist["dual_mean"].append(float(tf.reduce_mean(lam).numpy()))
        hist["dual_max"].append(float(tf.reduce_max(lam).numpy()))

    runtime_s = time.perf_counter() - start
    return AlgorithmResult(
        name="budgeted_primal_dual_pgd",
        w=w,
        runtime_s=runtime_s,
        history=hist,
        extra={"lambda_final": lam},
    )


def budgeted_primal_dual_pgd_repair(batch: WidebandBatch, cfg: Any) -> AlgorithmResult:
    """Primal-dual baseline plus inference-time calibration and guaranteed feasibility repair."""
    start = time.perf_counter()
    base = budgeted_primal_dual_pgd(batch, cfg)
    lam = tf.cast(base.extra.get("lambda_final"), batch.delay_spread_ns.dtype) if base.extra.get("lambda_final") is not None else initial_dual_from_batch(batch)
    w_cal, lam_cal, cal_hist = dual_calibration_refine(batch, base.w, lam, cfg, use_gate=True)
    w_rep, repair_stats = repair_fs_feasibility(batch, w_cal, cfg, lam=lam_cal)
    hist = _append_hist(base.history, cal_hist)
    for key, value in repair_stats.items():
        hist.setdefault(key, []).append(float(value))
    runtime_s = time.perf_counter() - start
    return AlgorithmResult(
        name="budgeted_primal_dual_pgd_repair",
        w=w_rep,
        runtime_s=runtime_s,
        history=hist,
        extra={"lambda_final": lam_cal, **repair_stats},
    )




def _sample_quantile(x: tf.Tensor, q: float) -> tf.Tensor:
    """Approximate per-sample quantile for a 2-D tensor [S,K]."""
    x_sorted = tf.sort(x, axis=1)
    k = tf.shape(x_sorted)[1]
    idx = tf.cast(tf.round(tf.cast(k - 1, tf.float32) * float(q)), tf.int32)
    idx = tf.clip_by_value(idx, 0, k - 1)
    return tf.gather(x_sorted, idx, axis=1, batch_dims=0)


def recover_rate_with_feasible_mask(
    batch: WidebandBatch,
    w_reference: tf.Tensor,
    w_feasible: tf.Tensor,
    cfg: Any,
    lam: tf.Tensor | None = None,
) -> Tuple[tf.Tensor, Dict[str, float]]:
    """Recover rate from a feasible point by blending back safe subbands.

    The recovery uses a masked convex combination between the feasible solution and
    a higher-rate reference point. Only subbands with low incumbent risk and low
    dual-weighted FS cost are allowed to move back toward the reference. A per-sample
    bisection then finds the largest feasible blend, followed by a tiny global rescue
    shrink for numerical safety.
    """
    real_dtype = _real_dtype_from_complex(w_feasible.dtype)
    risk_thr = float(twc_get(cfg, ('algorithm', 'recovery_safe_risk_threshold'), 0.45))
    cost_q = float(twc_get(cfg, ('algorithm', 'recovery_cost_quantile'), 0.60))
    bisect_iters = int(twc_get(cfg, ('algorithm', 'recovery_bisect_iters'), 12))
    min_safe_frac = float(twc_get(cfg, ('algorithm', 'recovery_min_safe_fraction'), 0.25))
    margin = float(twc_get(cfg, ('algorithm', 'recovery_feas_margin'), 0.999))
    use_lambda_weight = bool(twc_get(cfg, ('algorithm', 'recovery_use_lambda_weight'), True))

    comp_ref = fs_interference_components(batch, w_reference)
    contrib = tf.cast(comp_ref['contrib_s_k_l'], real_dtype)  # [S,K,L]
    i_max = tf.cast(batch.fs_stats.i_max_watt, real_dtype)[None, :] + tf.cast(1e-15, real_dtype)

    if lam is not None and use_lambda_weight:
        lam_norm = tf.cast(lam, real_dtype)
        lam_norm = lam_norm / tf.maximum(tf.reduce_mean(lam_norm, axis=1, keepdims=True), tf.cast(1e-9, real_dtype))
    else:
        lam_norm = tf.ones_like(i_max)

    cost_k = tf.einsum('sl,skl->sk', lam_norm / i_max, contrib, optimize=True)
    cost_k = cost_k / tf.maximum(tf.reduce_mean(cost_k, axis=1, keepdims=True), tf.cast(1e-9, real_dtype))
    risk_k = tf.cast(batch.risk_score, real_dtype)

    cost_thr = _sample_quantile(cost_k, cost_q)[:, None]
    safe_mask = tf.logical_and(risk_k <= tf.cast(risk_thr, real_dtype), cost_k <= cost_thr)

    min_k = tf.maximum(1, tf.cast(tf.round(tf.cast(tf.shape(cost_k)[1], tf.float32) * min_safe_frac), tf.int32))
    rank_mask = tf.sequence_mask(min_k, maxlen=tf.shape(cost_k)[1])
    sorted_idx = tf.argsort(cost_k, axis=1, direction='ASCENDING')
    scatter_shape = tf.shape(cost_k)
    batch_ids = tf.tile(tf.range(tf.shape(cost_k)[0])[:, None], [1, tf.shape(cost_k)[1]])
    gather_nd = tf.stack([batch_ids, sorted_idx], axis=-1)
    rank_mask_sorted = tf.cast(rank_mask, tf.bool)
    min_safe_mask = tf.scatter_nd(gather_nd[rank_mask_sorted], tf.ones([tf.reduce_sum(tf.cast(rank_mask_sorted, tf.int32))], dtype=tf.bool), scatter_shape)
    safe_mask = tf.logical_or(safe_mask, min_safe_mask)

    mask = tf.cast(safe_mask, real_dtype)[:, :, None, None, None]
    delta = tf.complex(mask * tf.math.real(w_reference - w_feasible), mask * tf.math.imag(w_reference - w_feasible))

    s = tf.shape(w_feasible)[0]
    lo = tf.zeros([s], dtype=real_dtype)
    hi = tf.ones([s], dtype=real_dtype)
    best = w_feasible

    target = tf.cast(margin, real_dtype) * i_max
    for _ in range(max(1, bisect_iters)):
        mid = (lo + hi) / tf.cast(2.0, real_dtype)
        w_try = w_feasible + _scale_complex_by_real(delta, mid[:, None, None, None, None])
        w_try = project_per_bs_power(w_try, batch.bs_power_budget_watt)
        i_try = fs_interference(batch, w_try)
        feasible = tf.reduce_all(i_try <= target, axis=1)
        best = tf.where(feasible[:, None, None, None, None], w_try, best)
        lo = tf.where(feasible, mid, lo)
        hi = tf.where(feasible, hi, mid)

    i_best = fs_interference(batch, best)
    worst_excess = tf.reduce_max(i_best / i_max - tf.cast(1.0, real_dtype), axis=1)
    need_rescue = worst_excess > tf.cast(0.0, real_dtype)
    rescue_ratio = tf.cast(margin, real_dtype) * i_max / tf.maximum(i_best, tf.cast(1e-15, real_dtype))
    gamma = tf.minimum(tf.cast(1.0, real_dtype), tf.reduce_min(rescue_ratio, axis=1))
    gamma = tf.where(need_rescue, gamma, tf.cast(1.0, real_dtype))
    best = _scale_complex_by_real(best, tf.sqrt(tf.maximum(gamma, tf.cast(1e-8, real_dtype)))[:, None, None, None, None])
    i_final = fs_interference(batch, best)
    worst_final = tf.reduce_max(i_final / i_max - tf.cast(1.0, real_dtype), axis=1)

    stats = {
        'recovery_safe_fraction_mean': float(tf.reduce_mean(tf.cast(safe_mask, real_dtype)).numpy()),
        'recovery_eta_mean': float(tf.reduce_mean(lo).numpy()),
        'recovery_eta_min': float(tf.reduce_min(lo).numpy()),
        'recovery_global_rescue_fraction': float(tf.reduce_mean(tf.cast(need_rescue, real_dtype)).numpy()),
        'recovery_global_scale_mean': float(tf.reduce_mean(gamma).numpy()),
        'recovery_worst_excess_mean': float(tf.reduce_mean(worst_final).numpy()),
    }
    return best, stats


def budgeted_primal_dual_pgd_repair_recover(batch: WidebandBatch, cfg: Any) -> AlgorithmResult:
    """PD baseline + calibration + guaranteed repair + masked feasible rate recovery."""
    start = time.perf_counter()
    base = budgeted_primal_dual_pgd(batch, cfg)
    lam0 = tf.cast(base.extra.get('lambda_final'), batch.delay_spread_ns.dtype) if base.extra.get('lambda_final') is not None else initial_dual_from_batch(batch)
    w_cal, lam_cal, cal_hist = dual_calibration_refine(batch, base.w, lam0, cfg, use_gate=True)
    w_rep, repair_stats = repair_fs_feasibility(batch, w_cal, cfg, lam=lam_cal)
    w_rec, rec_stats = recover_rate_with_feasible_mask(batch, w_cal, w_rep, cfg, lam=lam_cal)
    hist = _append_hist(base.history, cal_hist)
    for stats in (repair_stats, rec_stats):
        for key, value in stats.items():
            hist.setdefault(key, []).append(float(value))
    runtime_s = time.perf_counter() - start
    return AlgorithmResult(
        name='budgeted_primal_dual_pgd_repair_recover',
        w=w_rec,
        runtime_s=runtime_s,
        history=hist,
        extra={'lambda_final': lam_cal, **repair_stats, **rec_stats},
    )


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
