"""Figure generation for the TWC package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from .io import ensure_dir
from .types import WidebandBatch


def _save(fig: plt.Figure, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_geometry(batch: WidebandBatch, path: str | Path) -> None:
    topo = batch.topo
    fs = batch.fs_loc

    bs = topo.bs_loc[0].numpy()
    ut = topo.ut_loc[0].numpy()
    fs_xyz = fs.fs_loc[0].numpy()

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(bs[:, 0], bs[:, 1], marker="^", s=55, label="BS")
    ax.scatter(ut[:, 0], ut[:, 1], marker="o", s=12, alpha=0.6, label="UE")
    ax.scatter(fs_xyz[:, 0], fs_xyz[:, 1], marker="x", s=70, label="FS RX")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Reference geometry")
    ax.legend(loc="best")
    _save(fig, path)


def plot_subband_risk(batch: WidebandBatch, path: str | Path) -> None:
    risk = tf.reduce_mean(batch.risk_score, axis=0).numpy()
    overlap = tf.reduce_mean(batch.overlap, axis=1).numpy()
    subband_idx = np.arange(len(risk))

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(subband_idx, risk, label="Normalized risk")
    ax.plot(subband_idx, overlap, label="Mean overlap")
    ax.set_xlabel("Subband index")
    ax.set_ylabel("Score")
    ax.set_title("ISED-derived subband risk profile")
    ax.legend(loc="best")
    _save(fig, path)


def plot_summary_bars(summary_df: pd.DataFrame, fig_dir: str | Path) -> None:
    fig_dir = ensure_dir(fig_dir)

    metrics = [
        ("sum_rate_bps_per_hz_mean", "fig_rate_vs_algorithm.png", "Mean sum-rate [bps/Hz]"),
        ("fs_outage_prob_any", "fig_fs_outage_vs_algorithm.png", "FS outage probability"),
        ("cvar95_fs_excess", "fig_cvar_vs_algorithm.png", "CVaR$_{0.95}$ FS excess"),
        ("runtime_ms_mean", "fig_runtime_vs_algorithm.png", "Runtime [ms / sample]"),
    ]

    for metric, fname, ylabel in metrics:
        pivot = summary_df.groupby("algorithm", as_index=False)[metric].mean()
        fig = plt.figure(figsize=(6.8, 4.0))
        ax = fig.add_subplot(111)
        ax.bar(pivot["algorithm"], pivot[metric])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Algorithm")
        ax.set_title(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        _save(fig, Path(fig_dir) / fname)


def plot_cdfs(per_sample_df: pd.DataFrame, fig_dir: str | Path) -> None:
    fig_dir = ensure_dir(fig_dir)

    for col, fname, xlabel in [
        ("sum_rate_bps_per_hz", "fig_sum_rate_cdf.png", "Sum-rate [bps/Hz]"),
        ("worst_fs_excess", "fig_fs_excess_cdf.png", "Worst normalized FS excess"),
    ]:
        fig = plt.figure(figsize=(6.8, 4.0))
        ax = fig.add_subplot(111)
        for algo, grp in per_sample_df.groupby("algorithm"):
            x = np.sort(grp[col].to_numpy(dtype=float))
            y = np.linspace(0.0, 1.0, len(x), endpoint=True)
            ax.plot(x, y, label=algo)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(xlabel + " CDF")
        ax.legend(loc="best")
        _save(fig, Path(fig_dir) / fname)


def plot_tradeoff_scatter(summary_df: pd.DataFrame, path: str | Path) -> None:
    fig = plt.figure(figsize=(6.6, 4.4))
    ax = fig.add_subplot(111)
    for algo, grp in summary_df.groupby("algorithm"):
        ax.scatter(grp["fs_outage_prob_any"], grp["sum_rate_bps_per_hz_mean"], label=algo)
    ax.set_xlabel("FS outage probability")
    ax.set_ylabel("Mean sum-rate [bps/Hz]")
    ax.set_title("Rate vs incumbent-protection tradeoff")
    ax.legend(loc="best")
    _save(fig, path)


def plot_history(history_df: pd.DataFrame, path: str | Path) -> None:
    if history_df.empty:
        return
    fig = plt.figure(figsize=(6.6, 4.2))
    ax = fig.add_subplot(111)
    for algo, grp in history_df.groupby("algorithm"):
        ax.plot(grp["layer"], grp["utility"], marker="o", label=algo)
    ax.set_xlabel("Unfolded layer / iteration")
    ax.set_ylabel("Utility surrogate")
    ax.set_title("Convergence trace")
    ax.legend(loc="best")
    _save(fig, path)


def plot_tone_grouping(df: pd.DataFrame, path: str | Path) -> None:
    if df.empty:
        return
    fig = plt.figure(figsize=(6.6, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(df["group_size"], df["nmse"], marker="o")
    ax.set_xlabel("Tone-group size")
    ax.set_ylabel("Channel grouping NMSE")
    ax.set_title("Tone-grouping approximation error")
    _save(fig, path)


def plot_weight_sensitivity(df: pd.DataFrame, path: str | Path) -> None:
    if df.empty:
        return
    pivot = df.groupby(["weight_profile", "algorithm"], as_index=False)["sum_rate_bps_per_hz_mean"].mean()
    profiles = list(pivot["weight_profile"].drop_duplicates())
    algos = list(pivot["algorithm"].drop_duplicates())

    x = np.arange(len(profiles))
    width = 0.8 / max(1, len(algos))

    fig = plt.figure(figsize=(7.8, 4.2))
    ax = fig.add_subplot(111)
    for idx, algo in enumerate(algos):
        vals = []
        for prof in profiles:
            row = pivot[(pivot["weight_profile"] == prof) & (pivot["algorithm"] == algo)]
            vals.append(float(row["sum_rate_bps_per_hz_mean"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (idx - 0.5 * (len(algos) - 1)) * width, vals, width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=20)
    ax.set_ylabel("Mean sum-rate [bps/Hz]")
    ax.set_title("Sensitivity to user weights")
    ax.legend(loc="best")
    _save(fig, path)


def generate_all_figures(
    summary_df: pd.DataFrame,
    per_sample_df: pd.DataFrame,
    history_df: pd.DataFrame,
    tone_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    batch_reference: WidebandBatch,
    fig_dir: str | Path,
) -> None:
    fig_dir = ensure_dir(fig_dir)
    plot_reference_geometry(batch_reference, Path(fig_dir) / "fig_reference_geometry.png")
    plot_subband_risk(batch_reference, Path(fig_dir) / "fig_subband_risk_profile.png")
    plot_summary_bars(summary_df, fig_dir)
    plot_cdfs(per_sample_df, fig_dir)
    plot_tradeoff_scatter(summary_df, Path(fig_dir) / "fig_rate_fs_tradeoff_scatter.png")
    plot_history(history_df, Path(fig_dir) / "fig_convergence_unfolded.png")
    plot_tone_grouping(tone_df, Path(fig_dir) / "fig_tone_grouping_error.png")
    plot_weight_sensitivity(weight_df, Path(fig_dir) / "fig_weight_sensitivity.png")
