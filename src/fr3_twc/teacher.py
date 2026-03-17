"""Teacher utilities for budget-dual distillation and warm-start calibration."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict

import tensorflow as tf

from fr3_sim.config import ResolvedConfig
from fr3_sim.receiver import WmmseReceiver

from .algorithms import budgeted_primal_dual_pgd_repair_recover, fs_interference, subband_power_map, user_rate_tensors
from .config_utils import twc_get
from .types import WidebandBatch


@dataclass
class TeacherResult:
    w: tf.Tensor
    lam: tf.Tensor
    runtime_s: float


def _teacher_cfg(cfg: Any, batch: WidebandBatch) -> ResolvedConfig:
    raw = copy.deepcopy(cfg.raw)
    derived = copy.deepcopy(cfg.derived)

    w_cfg = raw.setdefault("receiver", {}).setdefault("wmmse", {})
    w_cfg["fs_enforcement"] = "budget_dual"
    w_cfg["fs_lambda_search"] = bool(twc_get(cfg, ("teacher", "fs_lambda_search"), True))
    w_cfg["fs_lambda_search_max_iter"] = int(twc_get(cfg, ("teacher", "fs_lambda_search_max_iter"), 12))
    w_cfg["num_iterations"] = int(twc_get(cfg, ("teacher", "num_iterations"), 10))
    w_cfg["dual_step_mu"] = float(twc_get(cfg, ("teacher", "dual_step_mu"), w_cfg.get("dual_step_mu", 0.01)))
    w_cfg["dual_step_lambda"] = float(twc_get(cfg, ("teacher", "dual_step_lambda"), w_cfg.get("dual_step_lambda", 0.01)))
    w_cfg["verbose"] = False

    derived["num_re_sim"] = int(batch.grid.num_subbands)
    derived["re_scaling"] = 1.0
    derived["bs_total_tx_power_watt"] = float(batch.bs_power_budget_watt)
    return ResolvedConfig(raw=raw, derived=derived)


def solve_budget_dual_teacher(cfg: Any, batch: WidebandBatch) -> TeacherResult:
    """Run the classical budget-dual WMMSE on the exact wideband batch."""
    teacher_cfg = _teacher_cfg(cfg, batch)
    recv = WmmseReceiver()
    start = time.perf_counter()
    result = recv.solve(
        teacher_cfg,
        batch.h_wb,
        noise_var_watt=float(batch.noise_var_watt),
        fs=batch.fs_stats,
        bs_total_tx_power_watt=float(batch.bs_power_budget_watt),
    )
    runtime_s = time.perf_counter() - start
    return TeacherResult(w=result.w, lam=result.lam, runtime_s=runtime_s)


def solve_repair_recover_teacher(cfg: Any, batch: WidebandBatch) -> TeacherResult:
    """Run the repaired+recovered PD baseline on the exact wideband batch."""
    result = budgeted_primal_dual_pgd_repair_recover(batch, cfg)
    lam = result.extra.get("lambda_final")
    if lam is None:
        lam = tf.ones_like(batch.fs_stats.i_max_watt[None, :], dtype=batch.delay_spread_ns.dtype)
    else:
        lam = tf.cast(lam, batch.delay_spread_ns.dtype)
    return TeacherResult(w=result.w, lam=lam, runtime_s=float(result.runtime_s))


def distillation_terms(
    batch: WidebandBatch,
    student_w: tf.Tensor,
    teacher: TeacherResult,
    lambda_student: tf.Tensor | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, tf.Tensor]:
    """Phase-robust distillation on signal, FS, power, dual, and feasibility tails."""
    weights = weights or {}
    rr_s = user_rate_tensors(batch, student_w)
    rr_t = user_rate_tensors(batch, tf.stop_gradient(teacher.w))

    i_s = fs_interference(batch, student_w)
    i_t = tf.stop_gradient(fs_interference(batch, teacher.w))
    i_max = tf.cast(batch.fs_stats.i_max_watt, i_s.dtype)[None, :] + tf.cast(1e-15, i_s.dtype)

    p_s = tf.cast(subband_power_map(student_w), rr_s["rate"].dtype)
    p_t = tf.cast(tf.stop_gradient(subband_power_map(teacher.w)), rr_s["rate"].dtype)
    p_budget = tf.cast(batch.bs_power_budget_watt, p_s.dtype)

    desired_loss = tf.reduce_mean(
        (tf.math.log1p(rr_s["desired"]) - tf.math.log1p(tf.stop_gradient(rr_t["desired"]))) ** 2
    )
    rate_loss = tf.reduce_mean(
        (tf.math.log1p(rr_s["rate"]) - tf.math.log1p(tf.stop_gradient(rr_t["rate"]))) ** 2
    )
    fs_loss = tf.reduce_mean(((i_s / i_max) - (i_t / i_max)) ** 2)
    power_loss = tf.reduce_mean(((p_s / p_budget) - (p_t / p_budget)) ** 2)

    teacher_lam = tf.cast(tf.stop_gradient(teacher.lam), i_s.dtype)
    if lambda_student is None:
        lambda_loss = tf.cast(0.0, i_s.dtype)
    else:
        lam_s = tf.cast(lambda_student, i_s.dtype)
        lam_s = lam_s / tf.maximum(tf.reduce_mean(lam_s, axis=1, keepdims=True), tf.cast(1e-9, i_s.dtype))
        lam_t = teacher_lam / tf.maximum(tf.reduce_mean(teacher_lam, axis=1, keepdims=True), tf.cast(1e-9, i_s.dtype))
        lambda_loss = tf.reduce_mean((lam_s - lam_t) ** 2)

    tail_s = tf.nn.relu(i_s / i_max - 1.0)
    tail_t = tf.nn.relu(i_t / i_max - 1.0)
    tail_loss = tf.reduce_mean((tail_s - tail_t) ** 2)

    total = (
        tf.cast(float(weights.get("desired", 1.0)), desired_loss.dtype) * desired_loss
        + tf.cast(float(weights.get("rate", 1.0)), desired_loss.dtype) * rate_loss
        + tf.cast(float(weights.get("fs", 1.0)), desired_loss.dtype) * fs_loss
        + tf.cast(float(weights.get("power", 0.5)), desired_loss.dtype) * power_loss
        + tf.cast(float(weights.get("lambda", 0.2)), desired_loss.dtype) * lambda_loss
        + tf.cast(float(weights.get("tail", 1.0)), desired_loss.dtype) * tail_loss
    )
    return {
        "total": total,
        "desired_loss": desired_loss,
        "rate_loss": rate_loss,
        "fs_loss": fs_loss,
        "power_loss": power_loss,
        "lambda_loss": lambda_loss,
        "tail_loss": tail_loss,
    }
