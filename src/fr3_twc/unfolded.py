########Update 5%
"""Unfolded learned precoders for the TWC pipeline."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import tensorflow as tf

from .algorithms import (
    dual_calibration_refine,
    initial_dual_from_batch,
    initial_mf_precoder,
    objective_terms,
    primal_dual_terms,
    project_per_bs_power,
    recover_rate_with_feasible_mask,
    repair_fs_feasibility,
)
from .config_utils import twc_get
from .types import AlgorithmResult, WidebandBatch


PD_MODES = {"budget_aware_primal_dual", "primal_dual_budgeted", "primal_dual"}
FR_MODES = {"feasibility_restored_primal_dual", "feasibility_restored", "fr_primal_dual"}


def _real_dtype_from_complex(dtype: tf.DType) -> tf.DType:
    return tf.float64 if dtype == tf.complex128 else tf.float32


def _scale_complex_by_real(x: tf.Tensor, coeff: tf.Tensor) -> tf.Tensor:
    """Scale a complex tensor by a real coefficient without complex->real backprop casts."""
    real_dtype = _real_dtype_from_complex(x.dtype)
    coeff = tf.cast(coeff, real_dtype)
    re = coeff * tf.math.real(x)
    im = coeff * tf.math.imag(x)
    return tf.complex(re, im)


def _append_hist(dst: Dict[str, List[float]], src: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out = {k: list(v) for k, v in dst.items()}
    for key, vals in src.items():
        out.setdefault(key, [])
        out[key].extend(list(vals))
    return out


class ScenarioAdaptiveUnfolded(tf.keras.Model):
    """Legacy risk-aware unfolded projected-gradient solver."""

    algorithm_name = "scenario_adaptive_unfolded"

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.layers_unfold = int(twc_get(cfg, ("unfolded", "layers"), 8))
        hidden_dim = int(twc_get(cfg, ("unfolded", "hidden_dim"), 32))
        feat_dim = 5

        self.conditioner = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(feat_dim,)),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.layers_unfold * 4, activation=None),
            ],
            name="scenario_conditioner",
        )

        init_step = float(twc_get(cfg, ("algorithm", "fixed_step_size"), 0.15))
        init_damp = float(twc_get(cfg, ("algorithm", "fixed_damping"), 0.75))
        init_fs = float(twc_get(cfg, ("algorithm", "fixed_fs_weight"), 15.0))
        init_cvar = float(twc_get(cfg, ("algorithm", "fixed_cvar_weight"), 8.0))

        self.base_log_step = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_step),
            trainable=True,
            name="base_log_step",
        )
        self.base_log_fs = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_fs),
            trainable=True,
            name="base_log_fs",
        )
        self.base_log_cvar = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_cvar),
            trainable=True,
            name="base_log_cvar",
        )
        self.base_damping = tf.Variable(
            tf.ones([self.layers_unfold], dtype=tf.float32) * init_damp,
            trainable=True,
            name="base_damping",
        )
        self.base_gate_bias = tf.Variable(
            tf.ones([self.layers_unfold], dtype=tf.float32) * 0.2,
            trainable=True,
            name="base_gate_bias",
        )

    def _layer_params(self, features: tf.Tensor) -> Dict[str, tf.Tensor]:
        raw = self.conditioner(features)
        raw = tf.reshape(raw, [-1, self.layers_unfold, 4])
        step = tf.nn.softplus(self.base_log_step[None, :] + raw[:, :, 0])
        fs_w = tf.nn.softplus(self.base_log_fs[None, :] + raw[:, :, 1])
        cvar_w = tf.nn.softplus(self.base_log_cvar[None, :] + raw[:, :, 2])
        damping = tf.sigmoid(self.base_damping[None, :] + raw[:, :, 3])
        gate_bias = self.base_gate_bias[None, :]
        return {"step": step, "fs_w": fs_w, "cvar_w": cvar_w, "damping": damping, "gate_bias": gate_bias}

    def call(self, batch: WidebandBatch, training: bool = False) -> Tuple[tf.Tensor, Dict[str, List[float]]]:
        params = self._layer_params(tf.cast(batch.scenario_features, tf.float32))
        w = initial_mf_precoder(batch, use_gate=True)
        hist: Dict[str, List[float]] = {"utility": [], "mean_violation": [], "cvar": [], "power_violation": []}

        power_weight = float(twc_get(self.cfg, ("algorithm", "power_weight"), 20.0))
        alpha_cvar = float(twc_get(self.cfg, ("algorithm", "alpha_cvar"), 0.95))
        use_soft_gate = bool(twc_get(self.cfg, ("coexistence", "use_soft_risk_gate"), True))
        gate_temp = float(twc_get(self.cfg, ("coexistence", "soft_gate_temperature"), 8.0))

        real_dtype = _real_dtype_from_complex(w.dtype)
        one = tf.cast(1.0, real_dtype)

        for ell in range(self.layers_unfold):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(w)
                terms = objective_terms(
                    batch=batch,
                    w=w,
                    fs_weight=tf.reduce_mean(params["fs_w"][:, ell]),
                    cvar_weight=tf.reduce_mean(params["cvar_w"][:, ell]),
                    power_weight=power_weight,
                    alpha_cvar=alpha_cvar,
                )
            grad = tf.stop_gradient(tape.gradient(terms["loss"], w))
            step = tf.cast(params["step"][:, ell], real_dtype)[:, None, None, None, None]
            damping = tf.cast(params["damping"][:, ell], real_dtype)[:, None, None, None, None]

            w_new = w - _scale_complex_by_real(grad, step)
            if use_soft_gate:
                gate = tf.sigmoid(
                    tf.cast(params["gate_bias"][:, ell], batch.risk_score.dtype)[:, None]
                    - tf.cast(gate_temp, batch.risk_score.dtype) * batch.risk_score
                )
            else:
                gate = batch.static_gate
            w_new = _scale_complex_by_real(w_new, gate[:, :, None, None, None])
            w_new = project_per_bs_power(w_new, batch.bs_power_budget_watt)
            w = _scale_complex_by_real(w_new, damping) + _scale_complex_by_real(w, one - damping)

            hist["utility"].append(float(terms["utility"].numpy()))
            hist["mean_violation"].append(float(terms["mean_violation"].numpy()))
            hist["cvar"].append(float(terms["cvar"].numpy()))
            hist["power_violation"].append(float(terms["power_violation"].numpy()))

        return w, hist


class BudgetAwarePrimalDualUnfolded(tf.keras.Model):
    """Budget-aware primal-dual unfolded solver with learned dual dynamics."""

    algorithm_name = "budget_aware_primal_dual_unfolded"

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.layers_unfold = int(twc_get(cfg, ("unfolded", "layers"), 8))
        hidden_dim = int(twc_get(cfg, ("unfolded", "hidden_dim"), 32))
        feat_dim = 5

        self.conditioner = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(feat_dim,)),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(hidden_dim, activation="relu"),
                tf.keras.layers.Dense(self.layers_unfold * 5, activation=None),
            ],
            name="pd_conditioner",
        )

        init_primal = float(twc_get(cfg, ("algorithm", "pd_primal_step_size"), 0.08))
        init_dual = float(twc_get(cfg, ("algorithm", "pd_dual_step_size"), 0.40))
        init_damp = float(twc_get(cfg, ("algorithm", "pd_damping"), twc_get(cfg, ("algorithm", "fixed_damping"), 0.75)))
        init_cvar = float(twc_get(cfg, ("algorithm", "pd_cvar_weight"), 2.0))
        init_dual_scale = float(twc_get(cfg, ("algorithm", "pd_dual_init_scale"), 0.6))

        self.base_log_primal_step = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_primal),
            trainable=True,
            name="base_log_primal_step",
        )
        self.base_log_dual_step = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_dual),
            trainable=True,
            name="base_log_dual_step",
        )
        self.base_damping = tf.Variable(
            tf.ones([self.layers_unfold], dtype=tf.float32) * init_damp,
            trainable=True,
            name="base_pd_damping",
        )
        self.base_log_cvar = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_cvar),
            trainable=True,
            name="base_log_pd_cvar",
        )
        self.base_log_dual_scale = tf.Variable(
            tf.math.log(tf.ones([self.layers_unfold], dtype=tf.float32) * init_dual_scale),
            trainable=True,
            name="base_log_dual_scale",
        )
        self.base_gate_bias = tf.Variable(
            tf.ones([self.layers_unfold], dtype=tf.float32) * 0.1,
            trainable=True,
            name="base_pd_gate_bias",
        )

    def _layer_params(self, features: tf.Tensor) -> Dict[str, tf.Tensor]:
        raw = tf.reshape(self.conditioner(features), [-1, self.layers_unfold, 5])
        primal_step = tf.nn.softplus(self.base_log_primal_step[None, :] + raw[:, :, 0])
        dual_step = tf.nn.softplus(self.base_log_dual_step[None, :] + raw[:, :, 1])
        damping = tf.sigmoid(self.base_damping[None, :] + raw[:, :, 2])
        cvar_w = tf.nn.softplus(self.base_log_cvar[None, :] + raw[:, :, 3])
        dual_scale = tf.nn.softplus(self.base_log_dual_scale[None, :] + raw[:, :, 4])
        gate_bias = self.base_gate_bias[None, :]
        return {
            "primal_step": primal_step,
            "dual_step": dual_step,
            "damping": damping,
            "cvar_w": cvar_w,
            "dual_scale": dual_scale,
            "gate_bias": gate_bias,
        }

    def call(self, batch: WidebandBatch, training: bool = False) -> Tuple[tf.Tensor, Dict[str, List[float]]]:
        params = self._layer_params(tf.cast(batch.scenario_features, tf.float32))
        w = initial_mf_precoder(batch, use_gate=True)

        dual_scale0 = tf.reduce_mean(params["dual_scale"][:, :1], axis=1)  # [S]
        lam = initial_dual_from_batch(batch, dual_init_scale=1.0) * tf.cast(dual_scale0[:, None], batch.delay_spread_ns.dtype)

        hist: Dict[str, List[float]] = {
            "utility": [],
            "mean_violation": [],
            "cvar": [],
            "power_violation": [],
            "dual_mean": [],
            "dual_max": [],
        }

        power_weight = float(twc_get(self.cfg, ("algorithm", "power_weight"), 20.0))
        alpha_cvar = float(twc_get(self.cfg, ("algorithm", "alpha_cvar"), 0.95))
        gate_temp = float(twc_get(self.cfg, ("coexistence", "soft_gate_temperature"), 8.0))
        use_soft_gate = bool(twc_get(self.cfg, ("coexistence", "use_soft_risk_gate"), True))
        violation_clip = float(twc_get(self.cfg, ("algorithm", "pd_violation_clip"), 5.0))

        real_dtype = _real_dtype_from_complex(w.dtype)
        one = tf.cast(1.0, real_dtype)

        for ell in range(self.layers_unfold):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(w)
                terms = primal_dual_terms(
                    batch=batch,
                    w=w,
                    lam=lam,
                    cvar_weight=tf.reduce_mean(params["cvar_w"][:, ell]),
                    power_weight=power_weight,
                    alpha_cvar=alpha_cvar,
                    violation_clip=violation_clip,
                )
            grad = tf.stop_gradient(tape.gradient(terms["loss"], w))
            step = tf.cast(params["primal_step"][:, ell], real_dtype)[:, None, None, None, None]
            damping = tf.cast(params["damping"][:, ell], real_dtype)[:, None, None, None, None]
            dual_step = tf.cast(params["dual_step"][:, ell], lam.dtype)[:, None]

            w_new = w - _scale_complex_by_real(grad, step)
            if use_soft_gate:
                gate = tf.sigmoid(
                    tf.cast(params["gate_bias"][:, ell], batch.risk_score.dtype)[:, None]
                    - tf.cast(gate_temp, batch.risk_score.dtype) * batch.risk_score
                )
                w_new = _scale_complex_by_real(w_new, gate[:, :, None, None, None])
            else:
                w_new = _scale_complex_by_real(w_new, batch.static_gate[:, :, None, None, None])
            w_new = project_per_bs_power(w_new, batch.bs_power_budget_watt)
            w = _scale_complex_by_real(w_new, damping) + _scale_complex_by_real(w, one - damping)

            dual_err = tf.clip_by_value(
                tf.stop_gradient(terms["excess"]),
                tf.cast(-1.0, lam.dtype),
                tf.cast(violation_clip, lam.dtype),
            )
            lam = tf.nn.relu(lam + dual_step * dual_err)

            hist["utility"].append(float(terms["utility"].numpy()))
            hist["mean_violation"].append(float(terms["mean_violation"].numpy()))
            hist["cvar"].append(float(terms["cvar"].numpy()))
            hist["power_violation"].append(float(terms["power_violation"].numpy()))
            hist["dual_mean"].append(float(tf.reduce_mean(lam).numpy()))
            hist["dual_max"].append(float(tf.reduce_max(lam).numpy()))

        self.last_lambda = lam
        return w, hist


class FeasibilityRestoredPrimalDualUnfolded(BudgetAwarePrimalDualUnfolded):
    """Same student backbone, but evaluated/trained with explicit feasibility restoration."""

    algorithm_name = "feasibility_restored_primal_dual_unfolded"


def unfolded_mode(cfg: Any) -> str:
    return str(twc_get(cfg, ("unfolded", "mode"), "scenario_adaptive")).strip().lower()


def train_unfolded_model(cfg: Any) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
    mode = unfolded_mode(cfg)
    if mode in FR_MODES:
        model = FeasibilityRestoredPrimalDualUnfolded(cfg)
    elif mode in PD_MODES:
        model = BudgetAwarePrimalDualUnfolded(cfg)
    else:
        model = ScenarioAdaptiveUnfolded(cfg)
    lr = float(twc_get(cfg, ("unfolded", "learning_rate"), 1e-3))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    return model, opt


def unfolded_inference(
    model: tf.keras.Model,
    batch: WidebandBatch,
    repair: bool = True,
    recover: bool = False,
    force_name: str | None = None,
) -> AlgorithmResult:
    start = time.perf_counter()
    w, hist = model(batch=batch, training=False)
    algo_name = force_name or getattr(model, "algorithm_name", "scenario_adaptive_unfolded")
    extra: Dict[str, Any] = {}

    last_lambda = getattr(model, "last_lambda", None)
    if last_lambda is not None:
        extra["lambda_final"] = last_lambda

    do_repair = repair
    if do_repair:
        lam0 = tf.cast(last_lambda, batch.delay_spread_ns.dtype) if last_lambda is not None else initial_dual_from_batch(batch)
        w_ref, lam1, cal_hist = dual_calibration_refine(batch, w, lam0, model.cfg, use_gate=True)
        hist = _append_hist(hist, cal_hist)
        w_feas, repair_stats = repair_fs_feasibility(batch, w_ref, model.cfg, lam=lam1)
        extra["lambda_final"] = lam1
        extra.update(repair_stats)
        for key, value in repair_stats.items():
            hist.setdefault(key, []).append(float(value))
        if recover:
            w, rec_stats = recover_rate_with_feasible_mask(batch, w_ref, w_feas, model.cfg, lam=lam1)
            extra.update(rec_stats)
            for key, value in rec_stats.items():
                hist.setdefault(key, []).append(float(value))
        else:
            w = w_feas

    runtime_s = time.perf_counter() - start
    return AlgorithmResult(name=algo_name, w=w, runtime_s=runtime_s, history=hist, extra=extra)
