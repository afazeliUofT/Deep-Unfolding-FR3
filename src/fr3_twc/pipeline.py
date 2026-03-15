###Updated###
"""Top-level TWC pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.seeding import set_global_seed

from .algorithms import objective_terms, risk_neutral_pgd, static_notch_mf, wideband_pgd_baseline
from .config_utils import results_root_dir, twc_get
from .figures import generate_all_figures
from .io import ensure_dir, save_json
from .metrics import history_to_frame, summarize_algorithm, tone_grouping_error
from .types import WidebandBatch
from .unfolded import ScenarioAdaptiveUnfolded, train_unfolded_model, unfolded_inference
from .wideband_channel import build_wideband_batch


SUMMARY_KEY_COLS = ["algorithm", "sweep_tag", "sweep_value"]


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


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
    epsilon = tf.cast(fs_new.epsilon, real_dtype)  # [K,L]
    i_max = tf.cast(fs_new.i_max_watt, real_dtype)[None, None, :] + tf.cast(1e-15, real_dtype)
    bar_beta = tf.cast(fs_new.bar_beta, real_dtype)  # [S,B,L]
    norm_coupling = tf.reduce_max(bar_beta / i_max, axis=1)  # [S,L]
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


def _run_algorithms(cfg: Any, batch: WidebandBatch, unfolded_model: ScenarioAdaptiveUnfolded | None) -> List:
    out = [
        static_notch_mf(batch),
        risk_neutral_pgd(batch, cfg),
        wideband_pgd_baseline(batch, cfg),
    ]
    if unfolded_model is not None:
        out.append(unfolded_inference(unfolded_model, batch))
    return out


def _train(cfg: Any, pipe_dir: Path) -> Tuple[Optional[ScenarioAdaptiveUnfolded], pd.DataFrame]:
    enabled = bool(twc_get(cfg, ("unfolded", "enabled"), True))
    if not enabled:
        return None, pd.DataFrame()

    model, opt = train_unfolded_model(cfg)
    epochs = int(twc_get(cfg, ("unfolded", "epochs"), 20))
    steps_per_epoch = int(twc_get(cfg, ("unfolded", "steps_per_epoch"), 8))
    train_batch_size = int(twc_get(cfg, ("unfolded", "train_batch_size"), 2))
    val_batches = int(twc_get(cfg, ("unfolded", "val_batches"), 2))
    grad_clip_norm = float(twc_get(cfg, ("unfolded", "grad_clip_norm"), 5.0))
    power_weight = float(twc_get(cfg, ("algorithm", "power_weight"), 20.0))
    alpha_cvar = float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95))

    rows: List[Dict[str, float]] = []
    best_val = np.inf
    ckpt_dir = ensure_dir(pipe_dir / "checkpoints")
    best_path = ckpt_dir / "unfolded.weights.h5"

    for epoch in range(1, epochs + 1):
        train_losses: List[float] = []
        train_utils: List[float] = []
        for _ in range(steps_per_epoch):
            batch = build_wideband_batch(cfg, batch_size=train_batch_size, user_weight_profile="uniform")
            with tf.GradientTape() as tape:
                w, _ = model(batch=batch, training=True)
                terms = objective_terms(
                    batch=batch,
                    w=w,
                    fs_weight=float(twc_get(cfg, ("algorithm", "fixed_fs_weight"), 15.0)),
                    cvar_weight=float(twc_get(cfg, ("algorithm", "fixed_cvar_weight"), 8.0)),
                    power_weight=power_weight,
                    alpha_cvar=alpha_cvar,
                )
                loss = terms["loss"]
            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if grads_and_vars:
                grads_list, vars_list = zip(*grads_and_vars)
                grads_clip, _ = tf.clip_by_global_norm(list(grads_list), grad_clip_norm)
                opt.apply_gradients(zip(grads_clip, vars_list))
            train_losses.append(float(loss.numpy()))
            train_utils.append(float(terms["utility"].numpy()))

        val_losses: List[float] = []
        for _ in range(val_batches):
            batch_val = build_wideband_batch(cfg, batch_size=train_batch_size, user_weight_profile="uniform")
            w_val, _ = model(batch=batch_val, training=False)
            terms_val = objective_terms(
                batch=batch_val,
                w=w_val,
                fs_weight=float(twc_get(cfg, ("algorithm", "fixed_fs_weight"), 15.0)),
                cvar_weight=float(twc_get(cfg, ("algorithm", "fixed_cvar_weight"), 8.0)),
                power_weight=power_weight,
                alpha_cvar=alpha_cvar,
            )
            val_losses.append(float(terms_val["loss"].numpy()))

        mean_train = float(np.mean(train_losses)) if train_losses else np.nan
        mean_val = float(np.mean(val_losses)) if val_losses else np.nan
        rows.append(
            {
                "epoch": epoch,
                "train_loss": mean_train,
                "train_utility": float(np.mean(train_utils)) if train_utils else np.nan,
                "val_loss": mean_val,
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
    """Split legacy directory name into (algorithm_name, sortable_run_tag)."""
    parts = str(path_name).split("_")
    if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
        algo_name = "_".join(parts[:-2])
        run_tag = f"{parts[-2]}_{parts[-1]}"
        return algo_name, run_tag
    return str(path_name), str(path_name)


def _latest_legacy_metric_files(root_dir: Path) -> Dict[str, Path]:
    """Keep only the latest legacy metrics file per algorithm."""
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
    """Run the full TWC pipeline."""
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
