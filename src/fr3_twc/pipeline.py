##Update 7
"""Top-level TWC pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.seeding import set_global_seed

from .algorithms import (
    budgeted_primal_dual_pgd,
    budgeted_primal_dual_pgd_repair,
    budgeted_primal_dual_pgd_repair_recover,
    objective_terms,
    risk_neutral_pgd,
    soft_outage_surrogate,
    static_notch_mf,
    tail_excess_penalty,
    wideband_pgd_baseline,
)
from .config_utils import results_root_dir, twc_get
from .figures import generate_all_figures
from .io import ensure_dir, save_json
from .metrics import history_to_frame, summarize_algorithm, tone_grouping_error
from .teacher import distillation_terms, solve_budget_dual_teacher, solve_repair_recover_teacher
from .types import WidebandBatch
from .unfolded import FR_MODES, PD_MODES, RR_MODES, train_unfolded_model, unfolded_inference, unfolded_mode
from .wideband_channel import build_wideband_batch


SUMMARY_KEY_COLS = ["algorithm", "sweep_tag", "sweep_value"]


def _mode_uses_primal_dual(mode: str) -> bool:
    return mode in PD_MODES or mode in FR_MODES or mode in RR_MODES


def _mode_uses_outage_curriculum(mode: str) -> bool:
    return mode in FR_MODES or mode in RR_MODES


def _teacher_for_mode(cfg: Any, batch: WidebandBatch, mode: str):
    if mode in RR_MODES:
        return solve_repair_recover_teacher(cfg, batch)
    return solve_budget_dual_teacher(cfg, batch)


def _validation_terms_for_mode(cfg: Any, model: tf.keras.Model, batch: WidebandBatch, mode: str) -> Dict[str, tf.Tensor]:
    if mode in RR_MODES:
        result = unfolded_inference(
            model,
            batch,
            repair=True,
            recover=True,
            force_name="rate_recovered_primal_dual_unfolded",
        )
        return _base_train_objective(cfg, batch, result.w)
    if mode in FR_MODES:
        result = unfolded_inference(
            model,
            batch,
            repair=True,
            recover=False,
            force_name="feasibility_restored_primal_dual_unfolded",
        )
        return _base_train_objective(cfg, batch, result.w)
    w_val, _ = model(batch=batch, training=False)
    return _base_train_objective(cfg, batch, w_val)


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _enabled_algorithms(cfg: Any) -> List[str]:
    algos = twc_get(cfg, ("algorithms", "enabled"), None)
    if algos is not None:
        return [str(a).strip() for a in algos]

    mode = unfolded_mode(cfg)
    out = ["static_notch_mf", "risk_neutral_pgd", "wideband_pgd"]
    if _mode_uses_primal_dual(mode):
        out.append("budgeted_primal_dual_pgd")
    if mode in FR_MODES or mode in RR_MODES:
        out.append("budgeted_primal_dual_pgd_repair")
    if mode in RR_MODES:
        out.append("budgeted_primal_dual_pgd_repair_recover")
    if bool(twc_get(cfg, ("unfolded", "enabled"), True)):
        if mode in FR_MODES:
            out.extend(["budget_aware_primal_dual_unfolded", "feasibility_restored_primal_dual_unfolded"])
        elif mode in RR_MODES:
            out.extend(["budget_aware_primal_dual_unfolded", "rate_recovered_primal_dual_unfolded"])
        elif mode in PD_MODES:
            out.append("budget_aware_primal_dual_unfolded")
        else:
            out.append("scenario_adaptive_unfolded")
    return out


def _clone_batch_with_fs_budget(
    cfg: Any,
    batch: WidebandBatch,
    fs_in_target_db: float,
    base_fs_in_target_db: float,
) -> WidebandBatch:
    """Scale incumbent thresholds for a budget sweep and refresh risk/gating."""
    delta_db = float(fs_in_target_db) - float(base_fs_in_target_db)
    scale = 10.0 ** (delta_db / 10.0)
    fs = batch.fs_stats
    i_max_watt = tf.cast(scale, fs.i_max_watt.dtype) * fs.i_max_watt
    fs_new = replace(fs, i_max_watt=i_max_watt)

    real_dtype = batch.delay_spread_ns.dtype
    epsilon = tf.cast(fs_new.epsilon, real_dtype)
    i_max = tf.cast(fs_new.i_max_watt, real_dtype)[None, None, :] + tf.cast(1e-15, real_dtype)
    bar_beta = tf.cast(fs_new.bar_beta, real_dtype)
    norm_coupling = tf.reduce_max(bar_beta / i_max, axis=1)
    risk_score = tf.einsum("kl,sl->sk", epsilon, norm_coupling, optimize=True)
    risk_score = risk_score / tf.maximum(
        tf.reduce_max(risk_score, axis=1, keepdims=True),
        tf.cast(1e-12, real_dtype),
    )

    gate_thr = tf.cast(float(twc_get(cfg, ("coexistence", "static_notch_threshold"), 0.55)), real_dtype)
    static_gate = tf.cast(risk_score <= gate_thr, real_dtype)

    mean_ds = tf.reduce_mean(batch.delay_spread_ns, axis=[1, 2])
    mean_risk = tf.reduce_mean(risk_score, axis=1)
    max_risk = tf.reduce_max(risk_score, axis=1)
    frac_blocked = tf.cast(1.0, real_dtype) - tf.reduce_mean(static_gate, axis=1)
    mean_los = tf.reduce_mean(tf.cast(batch.los_mask, real_dtype), axis=[1, 2])
    scenario_features = tf.stack([mean_ds, mean_risk, max_risk, frac_blocked, mean_los], axis=1)

    meta = dict(batch.metadata)
    meta["fs_in_target_db"] = float(fs_in_target_db)

    return replace(
        batch,
        fs_stats=fs_new,
        risk_score=risk_score,
        static_gate=static_gate,
        scenario_features=scenario_features,
        metadata=meta,
    )


def _clone_batch_with_user_weights(batch: WidebandBatch, weights: tf.Tensor, profile_name: str) -> WidebandBatch:
    meta = dict(batch.metadata)
    meta["weight_profile"] = profile_name
    return replace(batch, user_weights=weights, metadata=meta)


def _weight_vector_for_profile(batch: WidebandBatch, profile: str) -> tf.Tensor:
    """Build profile weights on the exact same geometry for fair comparison."""
    u = int(batch.user_weights.shape[0])
    if profile in ("uniform", "default"):
        return tf.ones_like(batch.user_weights)
    if profile in ("inverse_serving_gain", "cell_edge", "fairness"):
        gains = tf.reduce_mean(tf.reduce_sum(tf.abs(batch.h_eff) ** 2, axis=-1), axis=[0, 1, 2])[:u] + 1e-9
        w = 1.0 / gains
        return w / tf.reduce_mean(w)
    if profile in ("hotspot_priority", "critical", "fs_sensitive"):
        risk_user = tf.reduce_mean(tf.reduce_sum(tf.abs(batch.h_eff) ** 2, axis=-1), axis=[0, 1, 2])[:u]
        w = tf.cast(risk_user, batch.user_weights.dtype)
        return w / tf.reduce_mean(w)
    if profile in ("lognormal", "random"):
        w = tf.exp(tf.random.normal([u], stddev=0.5, dtype=batch.user_weights.dtype))
        return w / tf.reduce_mean(w)
    return tf.ones_like(batch.user_weights)


def _run_algorithms(cfg: Any, batch: WidebandBatch, unfolded_model: tf.keras.Model | None) -> List:
    enabled = set(_enabled_algorithms(cfg))
    mode = unfolded_mode(cfg)
    out = []
    if "static_notch_mf" in enabled:
        out.append(static_notch_mf(batch))
    if "risk_neutral_pgd" in enabled:
        out.append(risk_neutral_pgd(batch, cfg))
    if "wideband_pgd" in enabled:
        out.append(wideband_pgd_baseline(batch, cfg))
    if "budgeted_primal_dual_pgd" in enabled:
        out.append(budgeted_primal_dual_pgd(batch, cfg))
    if "budgeted_primal_dual_pgd_repair" in enabled:
        out.append(budgeted_primal_dual_pgd_repair(batch, cfg))
    if "budgeted_primal_dual_pgd_repair_recover" in enabled:
        out.append(budgeted_primal_dual_pgd_repair_recover(batch, cfg))
    if unfolded_model is not None:
        algo_name = getattr(unfolded_model, "algorithm_name", "scenario_adaptive_unfolded")
        if mode in FR_MODES:
            if "budget_aware_primal_dual_unfolded" in enabled:
                out.append(
                    unfolded_inference(
                        unfolded_model,
                        batch,
                        repair=False,
                        force_name="budget_aware_primal_dual_unfolded",
                    )
                )
            if "feasibility_restored_primal_dual_unfolded" in enabled:
                out.append(
                    unfolded_inference(
                        unfolded_model,
                        batch,
                        repair=True,
                        recover=False,
                        force_name="feasibility_restored_primal_dual_unfolded",
                    )
                )
        elif mode in RR_MODES:
            if "budget_aware_primal_dual_unfolded" in enabled:
                out.append(
                    unfolded_inference(
                        unfolded_model,
                        batch,
                        repair=False,
                        force_name="budget_aware_primal_dual_unfolded",
                    )
                )
            if "rate_recovered_primal_dual_unfolded" in enabled:
                out.append(
                    unfolded_inference(
                        unfolded_model,
                        batch,
                        repair=True,
                        recover=True,
                        force_name="rate_recovered_primal_dual_unfolded",
                    )
                )
        elif mode in PD_MODES:
            if "budget_aware_primal_dual_unfolded" in enabled:
                out.append(
                    unfolded_inference(
                        unfolded_model,
                        batch,
                        repair=False,
                        force_name="budget_aware_primal_dual_unfolded",
                    )
                )
        elif algo_name in enabled:
            out.append(unfolded_inference(unfolded_model, batch))
    return out


def _base_train_objective(cfg: Any, batch: WidebandBatch, w: tf.Tensor) -> Dict[str, tf.Tensor]:
    mode = unfolded_mode(cfg)
    if _mode_uses_primal_dual(mode):
        return objective_terms(
            batch=batch,
            w=w,
            fs_weight=float(twc_get(cfg, ("unfolded", "residual_fs_weight"), 1.0)),
            cvar_weight=float(twc_get(cfg, ("unfolded", "residual_cvar_weight"), 1.5)),
            power_weight=float(twc_get(cfg, ("algorithm", "power_weight"), 20.0)),
            alpha_cvar=float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95)),
        )
    return objective_terms(
        batch=batch,
        w=w,
        fs_weight=float(twc_get(cfg, ("algorithm", "fixed_fs_weight"), 15.0)),
        cvar_weight=float(twc_get(cfg, ("algorithm", "fixed_cvar_weight"), 8.0)),
        power_weight=float(twc_get(cfg, ("algorithm", "power_weight"), 20.0)),
        alpha_cvar=float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95)),
    )


def _should_distill(cfg: Any, epoch: int, global_step: int) -> bool:
    if not _mode_uses_primal_dual(unfolded_mode(cfg)):
        return False
    if not bool(twc_get(cfg, ("teacher", "distill", "enabled"), True)):
        return False
    max_epochs = int(twc_get(cfg, ("teacher", "distill", "epochs"), 0))
    if max_epochs > 0 and epoch > max_epochs:
        return False
    interval = max(1, int(twc_get(cfg, ("teacher", "distill", "interval"), 2)))
    return (global_step % interval) == 0


def _interp(epoch: int, start_epoch: int, end_epoch: int, start_val: float, end_val: float) -> float:
    if end_epoch <= start_epoch:
        return float(end_val)
    if epoch <= start_epoch:
        return float(start_val)
    if epoch >= end_epoch:
        return float(end_val)
    ratio = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start_val + ratio * (end_val - start_val))


def _curriculum_state(cfg: Any, epoch: int, total_epochs: int) -> Dict[str, float]:
    ramp_start = int(twc_get(cfg, ("unfolded", "outage_ramp_start_epoch"), max(1, total_epochs // 4)))
    ramp_end = int(twc_get(cfg, ("unfolded", "outage_ramp_end_epoch"), max(ramp_start + 1, total_epochs - 2)))

    distill_default = float(twc_get(cfg, ("teacher", "distill", "weight"), 0.5))
    return {
        "distill_weight": _interp(
            epoch,
            ramp_start,
            ramp_end,
            float(twc_get(cfg, ("teacher", "distill", "weight_start"), distill_default)),
            float(twc_get(cfg, ("teacher", "distill", "weight_end"), max(0.2, 0.5 * distill_default))),
        ),
        "outage_weight": _interp(
            epoch,
            ramp_start,
            ramp_end,
            float(twc_get(cfg, ("unfolded", "outage_weight_start"), 0.0)),
            float(twc_get(cfg, ("unfolded", "outage_weight_final"), 8.0)),
        ),
        "tail_weight": _interp(
            epoch,
            ramp_start,
            ramp_end,
            float(twc_get(cfg, ("unfolded", "tail_excess_weight_start"), 0.0)),
            float(twc_get(cfg, ("unfolded", "tail_excess_weight_final"), 4.0)),
        ),
        "excess_slack": _interp(
            epoch,
            ramp_start,
            ramp_end,
            float(twc_get(cfg, ("unfolded", "excess_slack_start"), 0.5)),
            float(twc_get(cfg, ("unfolded", "excess_slack_final"), 0.0)),
        ),
        "soft_outage_beta": _interp(
            epoch,
            ramp_start,
            ramp_end,
            float(twc_get(cfg, ("unfolded", "soft_outage_beta_start"), 8.0)),
            float(twc_get(cfg, ("unfolded", "soft_outage_beta_final"), 14.0)),
        ),
    }


def _teacher_loss_weights(cfg: Any) -> Dict[str, float]:
    return {
        "desired": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "desired"), 1.0)),
        "rate": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "rate"), 1.0)),
        "fs": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "fs"), 1.0)),
        "power": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "power"), 0.5)),
        "lambda": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "lambda"), 0.2)),
        "tail": float(twc_get(cfg, ("teacher", "distill", "loss_weights", "tail"), 1.0)),
    }


def _sample_training_batch(cfg: Any, batch_size: int) -> WidebandBatch:
    batch = build_wideband_batch(cfg, batch_size=batch_size, user_weight_profile="uniform")
    budget_values = list(twc_get(cfg, ("unfolded", "train_fs_in_target_db_values"), []))
    if budget_values:
        base_fs_budget = float(cfg.raw["fixed_service"]["in_target_db"])
        sampled_budget = float(np.random.choice(np.asarray(budget_values, dtype=float)))
        batch = _clone_batch_with_fs_budget(cfg, batch, sampled_budget, base_fs_budget)
    return batch


def _train(cfg: Any, pipe_dir: Path) -> Tuple[Optional[tf.keras.Model], pd.DataFrame]:
    enabled = bool(twc_get(cfg, ("unfolded", "enabled"), True))
    if not enabled:
        return None, pd.DataFrame()

    model, opt = train_unfolded_model(cfg)
    mode = unfolded_mode(cfg)
    epochs = int(twc_get(cfg, ("unfolded", "epochs"), 20))
    steps_per_epoch = int(twc_get(cfg, ("unfolded", "steps_per_epoch"), 8))
    train_batch_size = int(twc_get(cfg, ("unfolded", "train_batch_size"), 2))
    val_batches = int(twc_get(cfg, ("unfolded", "val_batches"), 2))
    grad_clip_norm = float(twc_get(cfg, ("unfolded", "grad_clip_norm"), 5.0))

    rows: List[Dict[str, float]] = []
    best_val = np.inf
    ckpt_dir = ensure_dir(pipe_dir / "checkpoints")
    best_path = ckpt_dir / "unfolded.weights.h5"

    global_step = 0
    strict_val_state = _curriculum_state(cfg, epochs, epochs)
    teacher_weights = _teacher_loss_weights(cfg)
    for epoch in range(1, epochs + 1):
        state = _curriculum_state(cfg, epoch, epochs)
        train_losses: List[float] = []
        train_utils: List[float] = []
        train_distill_losses: List[float] = []
        teacher_runtimes: List[float] = []
        train_soft_outages: List[float] = []
        train_tail_losses: List[float] = []

        for _ in range(steps_per_epoch):
            batch = _sample_training_batch(cfg, batch_size=train_batch_size)
            teacher = None
            if _should_distill(cfg, epoch=epoch, global_step=global_step):
                teacher = _teacher_for_mode(cfg, batch, mode)
                teacher_runtimes.append(float(teacher.runtime_s))
            with tf.GradientTape() as tape:
                w, _ = model(batch=batch, training=True)
                terms = _base_train_objective(cfg, batch, w)
                loss = terms["loss"]

                soft_out_value = tf.cast(0.0, terms["loss"].dtype)
                tail_value = tf.cast(0.0, terms["loss"].dtype)
                if _mode_uses_outage_curriculum(mode):
                    soft_out_value = soft_outage_surrogate(
                        terms["excess"],
                        beta=state["soft_outage_beta"],
                        slack=state["excess_slack"],
                    )
                    tail_value = tail_excess_penalty(terms["excess"], slack=state["excess_slack"])
                    loss = (
                        loss
                        + tf.cast(state["outage_weight"], loss.dtype) * soft_out_value
                        + tf.cast(state["tail_weight"], loss.dtype) * tail_value
                    )

                distill_loss_value = tf.cast(0.0, terms["loss"].dtype)
                if teacher is not None:
                    td = distillation_terms(
                        batch,
                        w,
                        teacher,
                        lambda_student=getattr(model, "last_lambda", None),
                        weights=teacher_weights,
                    )
                    distill_loss_value = td["total"]
                    loss = loss + tf.cast(state["distill_weight"], loss.dtype) * distill_loss_value
            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if grads_and_vars:
                grads_list, vars_list = zip(*grads_and_vars)
                grads_clip, _ = tf.clip_by_global_norm(list(grads_list), grad_clip_norm)
                opt.apply_gradients(zip(grads_clip, vars_list))
            train_losses.append(float(loss.numpy()))
            train_utils.append(float(terms["utility"].numpy()))
            train_distill_losses.append(float(distill_loss_value.numpy()))
            train_soft_outages.append(float(soft_out_value.numpy()))
            train_tail_losses.append(float(tail_value.numpy()))
            global_step += 1

        val_losses: List[float] = []
        val_utils: List[float] = []
        val_soft_outages: List[float] = []
        val_tail_losses: List[float] = []
        for _ in range(val_batches):
            batch_val = _sample_training_batch(cfg, batch_size=train_batch_size)
            terms_val = _validation_terms_for_mode(cfg, model, batch_val, mode)
            val_loss = terms_val["loss"]
            soft_out_val = tf.cast(0.0, val_loss.dtype)
            tail_val = tf.cast(0.0, val_loss.dtype)
            if _mode_uses_outage_curriculum(mode):
                soft_out_val = soft_outage_surrogate(
                    terms_val["excess"],
                    beta=strict_val_state["soft_outage_beta"],
                    slack=strict_val_state["excess_slack"],
                )
                tail_val = tail_excess_penalty(terms_val["excess"], slack=strict_val_state["excess_slack"])
                val_loss = (
                    val_loss
                    + tf.cast(strict_val_state["outage_weight"], val_loss.dtype) * soft_out_val
                    + tf.cast(strict_val_state["tail_weight"], val_loss.dtype) * tail_val
                )
            val_losses.append(float(val_loss.numpy()))
            val_utils.append(float(terms_val["utility"].numpy()))
            val_soft_outages.append(float(soft_out_val.numpy()))
            val_tail_losses.append(float(tail_val.numpy()))

        mean_train = float(np.mean(train_losses)) if train_losses else np.nan
        mean_val = float(np.mean(val_losses)) if val_losses else np.nan
        rows.append(
            {
                "epoch": epoch,
                "train_loss": mean_train,
                "train_utility": float(np.mean(train_utils)) if train_utils else np.nan,
                "train_distill_loss": float(np.mean(train_distill_losses)) if train_distill_losses else 0.0,
                "teacher_runtime_s_mean": float(np.mean(teacher_runtimes)) if teacher_runtimes else 0.0,
                "train_soft_outage": float(np.mean(train_soft_outages)) if train_soft_outages else 0.0,
                "train_tail_excess": float(np.mean(train_tail_losses)) if train_tail_losses else 0.0,
                "outage_weight": state["outage_weight"],
                "tail_weight": state["tail_weight"],
                "distill_weight": state["distill_weight"],
                "val_loss": mean_val,
                "val_utility": float(np.mean(val_utils)) if val_utils else np.nan,
                "val_soft_outage": float(np.mean(val_soft_outages)) if val_soft_outages else 0.0,
                "val_tail_excess": float(np.mean(val_tail_losses)) if val_tail_losses else 0.0,
            }
        )

        if np.isfinite(mean_val) and mean_val < best_val:
            best_val = mean_val
            model.save_weights(best_path)

    if best_path.exists():
        model.load_weights(best_path)

    hist_df = pd.DataFrame(rows)
    hist_df.to_csv(pipe_dir / "training_history.csv", index=False)
    return model, hist_df


def _parse_legacy_run_tag(path_name: str) -> Tuple[str, str]:
    parts = str(path_name).split("_")
    if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
        algo_name = "_".join(parts[:-2])
        run_tag = f"{parts[-2]}_{parts[-1]}"
        return algo_name, run_tag
    return str(path_name), str(path_name)


def _latest_legacy_metric_files(root_dir: Path) -> Dict[str, Path]:
    selected: Dict[str, Tuple[str, Path]] = {}
    for metric_path in sorted((root_dir / "legacy_baseline").glob("*/metrics.csv")):
        algo_name, run_tag = _parse_legacy_run_tag(metric_path.parent.name)
        prev = selected.get(algo_name)
        if prev is None or run_tag > prev[0]:
            selected[algo_name] = (run_tag, metric_path)
    return {algo_name: metric_path for algo_name, (_, metric_path) in selected.items()}


def _aggregate_summary_df(summary_df_raw: pd.DataFrame) -> pd.DataFrame:
    if summary_df_raw.empty:
        return summary_df_raw.copy()
    metric_cols = [c for c in summary_df_raw.columns if c not in SUMMARY_KEY_COLS]
    summary_df = summary_df_raw.groupby(SUMMARY_KEY_COLS, as_index=False)[metric_cols].mean(numeric_only=True)
    return summary_df.sort_values(SUMMARY_KEY_COLS).reset_index(drop=True)


def _legacy_i_max_watt(cfg: Any, row: Dict[str, Any]) -> float:
    base_target_db = float(cfg.raw["fixed_service"]["in_target_db"])
    target_db = _safe_float(row.get("fs_in_target_db", row.get("sweep_value", base_target_db)), base_target_db)

    if "fs_i_max_watt" in cfg.derived:
        base_i_max = _safe_float(cfg.derived.get("fs_i_max_watt"), np.nan)
        if np.isfinite(base_i_max):
            return float(base_i_max) * (10.0 ** ((target_db - base_target_db) / 10.0))

    fs_noise_watt = _safe_float(cfg.derived.get("fs_noise_watt", np.nan), np.nan)
    if np.isfinite(fs_noise_watt):
        return float(fs_noise_watt) * (10.0 ** (target_db / 10.0))

    return np.nan


def run_pipeline(cfg: Any) -> str:
    seed = int(cfg.raw["reproducibility"]["seed"])
    deterministic_tf = bool(cfg.raw["reproducibility"].get("deterministic_tf", True))
    set_global_seed(seed, deterministic_tf=deterministic_tf)

    root_dir = results_root_dir(cfg)
    pipe_dir = ensure_dir(root_dir / "twc_pipeline")
    fig_dir = ensure_dir(pipe_dir / "figures")

    save_json(
        pipe_dir / "run_metadata.json",
        {
            "seed": seed,
            "experiment_name": str(cfg.raw["output"]["experiment_name"]),
            "num_bs": int(cfg.derived["num_bs"]),
            "num_ut": int(cfg.derived["num_ut"]),
            "u_per_bs": int(cfg.derived["u_per_bs"]),
            "num_re_sim": int(cfg.derived["num_re_sim"]),
            "bs_power_budget_watt": float(cfg.derived.get("bs_power_budget_sim_watt", cfg.derived["bs_total_tx_power_watt"])),
            "enabled_algorithms": _enabled_algorithms(cfg),
            "unfolded_mode": unfolded_mode(cfg),
        },
    )

    unfolded_model, _training_hist_df = _train(cfg, pipe_dir)

    eval_batch_size = int(twc_get(cfg, ("eval", "batch_size"), 2))
    eval_num_batches = int(twc_get(cfg, ("eval", "num_batches"), 6))
    fs_budget_values = list(twc_get(cfg, ("eval", "fs_in_target_db_values"), [float(cfg.raw["fixed_service"]["in_target_db"])]))
    base_fs_budget = float(cfg.raw["fixed_service"]["in_target_db"])

    summary_rows: List[Dict[str, float]] = []
    per_sample_rows: List[Dict[str, float]] = []
    hist_frames: List[pd.DataFrame] = []

    batch_reference = build_wideband_batch(cfg, batch_size=1, user_weight_profile="uniform")

    for fs_budget in fs_budget_values:
        for _ in range(eval_num_batches):
            batch = build_wideband_batch(cfg, batch_size=eval_batch_size, user_weight_profile="uniform")
            batch = _clone_batch_with_fs_budget(
                cfg=cfg,
                batch=batch,
                fs_in_target_db=float(fs_budget),
                base_fs_in_target_db=base_fs_budget,
            )
            alg_results = _run_algorithms(cfg, batch, unfolded_model)
            for res in alg_results:
                summary, rows = summarize_algorithm(
                    batch=batch,
                    result=res,
                    sweep_tag="fs_in_target_db",
                    sweep_value=float(fs_budget),
                )
                summary_rows.append(summary)
                per_sample_rows.extend(rows)
                hist_frames.append(history_to_frame(res.history, res.name))

    summary_df_raw = pd.DataFrame(summary_rows)
    summary_df = _aggregate_summary_df(summary_df_raw)
    per_sample_df = pd.DataFrame(per_sample_rows)
    hist_frames = [df for df in hist_frames if not df.empty]
    history_df = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()

    summary_df_raw.to_csv(pipe_dir / "metrics_summary_raw.csv", index=False)
    summary_df.to_csv(pipe_dir / "metrics_summary.csv", index=False)
    per_sample_df.to_csv(pipe_dir / "metrics_per_sample.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(pipe_dir / "algorithm_history.csv", index=False)

    tone_group_sizes = list(twc_get(cfg, ("eval", "tone_group_sizes"), [1, 2, 4, 8]))
    tone_df = tone_grouping_error(batch_reference, group_sizes=tone_group_sizes)
    tone_df.to_csv(pipe_dir / "tone_grouping_error.csv", index=False)

    weight_profiles = list(
        twc_get(cfg, ("eval", "user_weight_profiles"), ["uniform", "inverse_serving_gain", "hotspot_priority", "lognormal"])
    )
    weight_rows: List[Dict[str, float]] = []
    weight_num_batches = int(twc_get(cfg, ("eval", "num_batches"), 6))
    for sample_idx in range(weight_num_batches):
        batch_ref = build_wideband_batch(cfg, batch_size=eval_batch_size, user_weight_profile="uniform")
        for prof in weight_profiles:
            weights = _weight_vector_for_profile(batch_ref, prof)
            batch_w = _clone_batch_with_user_weights(batch_ref, weights=weights, profile_name=prof)
            alg_results = _run_algorithms(cfg, batch_w, unfolded_model)
            for res in alg_results:
                summary, _ = summarize_algorithm(
                    batch=batch_w,
                    result=res,
                    sweep_tag="weight_profile",
                    sweep_value=0.0,
                )
                summary["weight_profile"] = prof
                summary["weight_sample_idx"] = sample_idx
                weight_rows.append(summary)

    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(pipe_dir / "user_weight_sensitivity.csv", index=False)

    legacy_metric_files = _latest_legacy_metric_files(root_dir)
    if legacy_metric_files:
        legacy_rows: List[Dict[str, float]] = []
        for algo_name, metric_path in sorted(legacy_metric_files.items()):
            try:
                df = pd.read_csv(metric_path)
            except Exception:
                continue
            if df.empty:
                continue
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                max_fs_interference = _safe_float(row_dict.get("max_fs_interference_watt", np.nan), np.nan)
                max_fs_violation = _safe_float(row_dict.get("max_fs_violation_watt", 0.0), 0.0)
                i_max_watt = _legacy_i_max_watt(cfg, row_dict)

                if np.isfinite(max_fs_interference) and np.isfinite(i_max_watt) and i_max_watt > 0.0:
                    worst_excess = float(max_fs_interference / i_max_watt - 1.0)
                    positive_excess = float(max(0.0, worst_excess))
                    fs_outage = float(positive_excess > 0.0)
                else:
                    worst_excess = np.nan
                    positive_excess = float(max_fs_violation > 0.0)
                    fs_outage = float(max_fs_violation > 0.0)

                legacy_rows.append(
                    {
                        "algorithm": algo_name,
                        "sweep_tag": str(row_dict.get("sweep_variable", "legacy")),
                        "sweep_value": _safe_float(row_dict.get("sweep_value", 0.0), 0.0),
                        "sum_rate_bps_per_hz_mean": _safe_float(row_dict.get("sum_rate_bps_per_hz", np.nan), np.nan),
                        "weighted_sum_rate_mean": _safe_float(row_dict.get("sum_rate_bps_per_hz", np.nan), np.nan),
                        "sum_rate_bps_per_hz_p50": _safe_float(row_dict.get("sum_rate_bps_per_hz", np.nan), np.nan),
                        "sum_rate_bps_per_hz_p05": _safe_float(row_dict.get("sum_rate_bps_per_hz", np.nan), np.nan),
                        "sum_rate_bps_per_hz_p95": _safe_float(row_dict.get("sum_rate_bps_per_hz", np.nan), np.nan),
                        "fs_outage_prob_any": fs_outage,
                        "worst_fs_excess_mean": worst_excess,
                        "mean_positive_fs_excess": positive_excess,
                        "cvar95_fs_excess": positive_excess,
                        "power_violation_mean": _safe_float(row_dict.get("max_bs_power_violation_watt", np.nan), np.nan),
                        "active_subband_fraction_mean": np.nan,
                        "runtime_ms_mean": np.nan,
                        "mean_delay_spread_ns": np.nan,
                        "blocked_subband_fraction_mean": np.nan,
                    }
                )
        if legacy_rows:
            legacy_df = pd.DataFrame(legacy_rows)
            summary_df_all = pd.concat([summary_df, legacy_df], ignore_index=True)
        else:
            summary_df_all = summary_df
    else:
        summary_df_all = summary_df

    summary_df_all.to_csv(pipe_dir / "metrics_summary_with_legacy.csv", index=False)

    generate_all_figures(
        summary_df=summary_df_all,
        per_sample_df=per_sample_df,
        history_df=history_df,
        tone_df=tone_df,
        weight_df=weight_df,
        batch_reference=batch_reference,
        fig_dir=fig_dir,
    )

    return str(pipe_dir)
