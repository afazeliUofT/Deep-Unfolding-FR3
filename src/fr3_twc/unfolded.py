"""Scenario-adaptive unfolded risk-aware precoder."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import tensorflow as tf

from .algorithms import initial_mf_precoder, objective_terms, project_per_bs_power
from .config_utils import twc_get
from .types import AlgorithmResult, WidebandBatch


def _real_dtype_from_complex(dtype: tf.DType) -> tf.DType:
    return tf.float64 if dtype == tf.complex128 else tf.float32


def _scale_complex_by_real(x: tf.Tensor, coeff: tf.Tensor) -> tf.Tensor:
    """Scale a complex tensor by a real coefficient without complex->real backprop casts."""
    real_dtype = _real_dtype_from_complex(x.dtype)
    coeff = tf.cast(coeff, real_dtype)
    re = coeff * tf.math.real(x)
    im = coeff * tf.math.imag(x)
    return tf.complex(re, im)


class ScenarioAdaptiveUnfolded(tf.keras.Model):
    """Learnable unfolded projected-gradient solver.

    Notes
    -----
    - The gradient direction is taken from a fixed analytical surrogate.
    - `tf.stop_gradient` is applied to the inner gradient so training only learns
      layer parameters (step sizes, damping, and risk-gating scales) and does not
      require second-order differentiation through the optimizer.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.layers_unfold = int(twc_get(cfg, ("unfolded", "layers"), 8))
        hidden_dim = int(twc_get(cfg, ("unfolded", "hidden_dim"), 32))
        feat_dim = 5  # fixed in wideband_channel.build_wideband_batch

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
        raw = self.conditioner(features)  # [S, L*4]
        raw = tf.reshape(raw, [-1, self.layers_unfold, 4])  # [S,L,4]

        step = tf.nn.softplus(self.base_log_step[None, :] + raw[:, :, 0])
        fs_w = tf.nn.softplus(self.base_log_fs[None, :] + raw[:, :, 1])
        cvar_w = tf.nn.softplus(self.base_log_cvar[None, :] + raw[:, :, 2])
        damping = tf.sigmoid(self.base_damping[None, :] + raw[:, :, 3])
        gate_bias = self.base_gate_bias[None, :]
        return {
            "step": step,
            "fs_w": fs_w,
            "cvar_w": cvar_w,
            "damping": damping,
            "gate_bias": gate_bias,
        }

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



def train_unfolded_model(cfg: Any) -> Tuple[ScenarioAdaptiveUnfolded, tf.keras.optimizers.Optimizer]:
    model = ScenarioAdaptiveUnfolded(cfg)
    lr = float(twc_get(cfg, ("unfolded", "learning_rate"), 1e-3))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    return model, opt



def unfolded_inference(model: ScenarioAdaptiveUnfolded, batch: WidebandBatch) -> AlgorithmResult:
    start = time.perf_counter()
    w, hist = model(batch=batch, training=False)
    runtime_s = time.perf_counter() - start
    return AlgorithmResult(name="scenario_adaptive_unfolded", w=w, runtime_s=runtime_s, history=hist)
