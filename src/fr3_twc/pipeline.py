"""Top-level TWC pipeline orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from fr3_sim.seeding import set_global_seed

from .algorithms import risk_neutral_pgd, static_notch_mf, wideband_pgd_baseline
from .config_utils import results_root_dir, twc_get
from .figures import generate_all_figures
from .io import ensure_dir, save_json
from .metrics import history_to_frame, summarize_algorithm, tone_grouping_error
from .types import WidebandBatch
from .unfolded import ScenarioAdaptiveUnfolded, train_unfolded_model, unfolded_inference
from .wideband_channel import build_wideband_batch


def _clone_batch_with_fs_budget(batch: WidebandBatch, fs_in_target_db: float, base_fs_in_target_db: float) -> WidebandBatch:
    """Scale incumbent thresholds for a budget sweep."""
    delta_db = float(fs_in_target_db) - float(base_fs_in_target_db)
    scale = 10.0 ** (delta_db / 10.0)
    fs = batch.fs_stats
    fs_new = replace(fs, i_max_watt=tf.cast(scale, fs.i_max_watt.dtype) * fs.i_max_watt)
    return replace(batch, fs_stats=fs_new)


def _clone_batch_with_user_weights(batch: WidebandBatch, weights: tf.Tensor, profile_name: str) -> WidebandBatch:
    meta = dict(batch.metadata)
    meta["weight_profile"] = profile_name
    return replace(batch, user_weights=weights, metadata=meta)


def _weight_vector_for_profile(batch: WidebandBatch, profile: str) -> tf.Tensor:
    # Build directly from the current batch to ensure the same geometry across profiles
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
    power_weight = float(twc_get(cfg, ("algorithm", "power_weight"), 20.0))
    alpha_cvar = float(twc_get(cfg, ("algorithm", "alpha_cvar"), 0.95))

    rows: List[Dict[str, float]] = []
    best_val = np.inf
    ckpt_dir = ensure_dir(pipe_dir / "checkpoints")
    best_path = ckpt_dir / "unfolded.weights.h5"

    for epoch in range(1, epochs + 1):
        train_losses = []
        train_utils = []
        for _ in range(steps_per_epoch):
            batch = build_wideband_batch(cfg, batch_size=train_batch_size, user_weight_profile="uniform")
            with tf.GradientTape() as tape:
                w, _ = model(batch, training=True)
                from .algorithms import objective_terms
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
            opt.apply_gradients(grads_and_vars)
            train_losses.append(float(loss.numpy()))
            train_utils.append(float(terms["utility"].numpy()))

        val_losses = []
        for _ in range(val_batches):
            batch_val = build_wideband_batch(cfg, batch_size=train_batch_size, user_weight_profile="uniform")
            w_val, _ = model(batch_val, training=False)
            from .algorithms import objective_terms
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


def run_pipeline(cfg: Any) -> str:
    """Run the full TWC pipeline."""
    seed = int(cfg.raw["reproducibility"]["seed"])
    deterministic_tf = bool(cfg.raw["reproducibility"].get("deterministic_tf", True))
    set_global_seed(seed, deterministic_tf=deterministic_tf)

    root_dir = results_root_dir(cfg)
    pipe_dir = ensure_dir(root_dir / "twc_pipeline")
    fig_dir = ensure_dir(pipe_dir / "figures")

    # Save high-level config metadata
    save_json(
        pipe_dir / "run_metadata.json",
        {
            "seed": seed,
            "experiment_name": str(cfg.raw["output"]["experiment_name"]),
            "num_bs": int(cfg.derived["num_bs"]),
            "num_ut": int(cfg.derived["num_ut"]),
            "u_per_bs": int(cfg.derived["u_per_bs"]),
            "num_re_sim": int(cfg.derived["num_re_sim"]),
        },
    )

    unfolded_model, training_hist_df = _train(cfg, pipe_dir)

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
            batch = _clone_batch_with_fs_budget(batch, fs_in_target_db=float(fs_budget), base_fs_in_target_db=base_fs_budget)
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

    summary_df = pd.DataFrame(summary_rows)
    per_sample_df = pd.DataFrame(per_sample_rows)
    history_df = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()

    summary_df.to_csv(pipe_dir / "metrics_summary.csv", index=False)
    per_sample_df.to_csv(pipe_dir / "metrics_per_sample.csv", index=False)
    if not history_df.empty:
        history_df.to_csv(pipe_dir / "algorithm_history.csv", index=False)

    # Tone-grouping analysis
    tone_group_sizes = list(twc_get(cfg, ("eval", "tone_group_sizes"), [1, 2, 4, 8]))
    tone_df = tone_grouping_error(batch_reference, group_sizes=tone_group_sizes)
    tone_df.to_csv(pipe_dir / "tone_grouping_error.csv", index=False)

    # Weight sensitivity
    weight_profiles = list(twc_get(cfg, ("eval", "user_weight_profiles"), ["uniform", "inverse_serving_gain", "hotspot_priority", "lognormal"]))
    weight_rows: List[Dict[str, float]] = []

    for prof in weight_profiles:
        batch = build_wideband_batch(cfg, batch_size=eval_batch_size, user_weight_profile="uniform")
        weights = _weight_vector_for_profile(batch, prof)
        batch_w = _clone_batch_with_user_weights(batch, weights=weights, profile_name=prof)
        alg_results = _run_algorithms(cfg, batch_w, unfolded_model)
        for res in alg_results:
            summary, _ = summarize_algorithm(
                batch=batch_w,
                result=res,
                sweep_tag="weight_profile",
                sweep_value=0.0,
            )
            summary["weight_profile"] = prof
            weight_rows.append(summary)

    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(pipe_dir / "user_weight_sensitivity.csv", index=False)

    # Attach legacy baseline if available
    legacy_summary_paths = sorted((root_dir / "legacy_baseline").glob("*/metrics.csv"))
    if legacy_summary_paths:
        legacy_rows = []
        for p in legacy_summary_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if df.empty:
                continue
            row = df.iloc[-1].to_dict()
            legacy_rows.append(
                {
                    "algorithm": str(p.parent.name),
                    "sweep_tag": str(df.get("sweep_variable", pd.Series(["legacy"])).iloc[-1]) if "sweep_variable" in df.columns else "legacy",
                    "sweep_value": float(df.get("sweep_value", pd.Series([0.0])).iloc[-1]) if "sweep_value" in df.columns else 0.0,
                    "sum_rate_bps_per_hz_mean": float(row.get("sum_rate_bps_per_hz", np.nan)),
                    "weighted_sum_rate_mean": float(row.get("sum_rate_bps_per_hz", np.nan)),
                    "sum_rate_bps_per_hz_p50": float(row.get("sum_rate_bps_per_hz", np.nan)),
                    "sum_rate_bps_per_hz_p05": float(row.get("sum_rate_bps_per_hz", np.nan)),
                    "sum_rate_bps_per_hz_p95": float(row.get("sum_rate_bps_per_hz", np.nan)),
                    "fs_outage_prob_any": float((row.get("max_fs_violation_watt", 0.0) > 0.0)),
                    "worst_fs_excess_mean": float(row.get("max_fs_violation_watt", np.nan)),
                    "mean_positive_fs_excess": float(row.get("mean_fs_interference_watt", np.nan)),
                    "cvar95_fs_excess": float(row.get("max_fs_violation_watt", np.nan)),
                    "power_violation_mean": float(row.get("max_bs_power_violation_watt", np.nan)),
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
