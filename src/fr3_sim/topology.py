"""Topology generation for FR3 coexistence simulations.

This module generates:
- Hex-grid cellular layouts (BS + UEs) via Sionna's topology helpers.
- Fixed Service (FS) receiver locations and per-receiver metadata either from:
  (a) ISED SMS CSV records (preferred), or
  (b) parametric fits / synthetic sampling (fallback), or
  (c) legacy uniform placement (legacy fallback).

Design goals:
- Keep geometry generation centralized here (channel.py does no file I/O).
- Preserve reproducibility (numpy RNG is seeded via `set_global_seed` in runner.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import math

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import ResolvedConfig, dbm_to_watt


# -----------------------
# Public data structures
# -----------------------

@dataclass(frozen=True)
class TopologyData:
    """Container for cellular topology.

    Notes on shapes:
    - bs_loc:         [batch, B, 3]
    - ut_loc:         [batch, U, 3]
    - bs_virtual_loc: [batch, B, U, 3]  (closest wrap-around image per (BS,UT))
    - ut_is_indoor:   [batch, U] boolean (if provided by Sionna)
    """
    bs_loc: tf.Tensor
    ut_loc: tf.Tensor
    bs_virtual_loc: tf.Tensor
    ut_orientations: tf.Tensor
    bs_orientations: tf.Tensor
    ut_velocities: tf.Tensor
    serving_bs: np.ndarray
    ut_is_indoor: Optional[tf.Tensor] = None


@dataclass(frozen=True)
class FixedServiceLocations:
    """FS receiver locations and optional per-receiver metadata.

    Shapes (if provided):
    - fs_loc:          [batch, L, 3]
    - fs_center_hz:    [L]
    - fs_bw_hz:        [L]
    - fs_i_max_watt:   [L]
    - fs_gain_dbi:     [L]
    - fs_beamwidth_deg:[L]
    - fs_azimuth_deg:  [L]
    """
    fs_loc: tf.Tensor
    fs_center_hz: Optional[tf.Tensor] = None
    fs_bw_hz: Optional[tf.Tensor] = None
    fs_i_max_watt: Optional[tf.Tensor] = None
    fs_gain_dbi: Optional[tf.Tensor] = None
    fs_beamwidth_deg: Optional[tf.Tensor] = None
    fs_azimuth_deg: Optional[tf.Tensor] = None


# -----------------------
# Helpers
# -----------------------

def _real_dtype(cfg: ResolvedConfig) -> tf.DType:
    prec = str(cfg.derived.get("tf_precision", "single")).lower()
    return tf.float64 if prec.startswith("double") else tf.float32


def _import_gen_hexgrid_topology():
    """Import Sionna's hex-grid topology generator with backward-compatible paths."""
    try:
        from sionna.sys.topology import gen_hexgrid_topology  # type: ignore
    except Exception:  # pragma: no cover
        from sionna.sys import gen_hexgrid_topology  # type: ignore
    return gen_hexgrid_topology


def _call_with_compatible_kwargs(fn, candidates: Tuple[Tuple[Tuple[str, ...], Any], ...]):
    """Call `fn` using the first compatible keyword for each candidate value."""
    import inspect

    sig = inspect.signature(fn)
    params = sig.parameters
    kwargs: Dict[str, Any] = {}
    for names, value in candidates:
        for name in names:
            if name in params:
                kwargs[name] = value
                break
    return fn(**kwargs)


def _resolve_path(cfg: ResolvedConfig, path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    base = Path(str(cfg.derived.get("config_dir", ".")))
    return (base / p).resolve()


@lru_cache(maxsize=8)
def _read_csv_cached(abs_path: str) -> pd.DataFrame:
    return pd.read_csv(abs_path, low_memory=False)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _station_function_mask(df: pd.DataFrame, col: str, regex: str) -> pd.Series:
    s = df[col].astype(str)
    try:
        return s.str.contains(regex, case=False, na=False, regex=True)
    except Exception:
        return s.str.contains(regex, case=False, na=False, regex=False)


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 360.0)


def _dbm_to_watt_vec(dbm: np.ndarray) -> np.ndarray:
    return np.power(10.0, (dbm.astype(float) - 30.0) / 10.0)


def _dbw_to_watt_vec(dbw: np.ndarray) -> np.ndarray:
    return np.power(10.0, dbw.astype(float) / 10.0)


def _compute_noise_dbm_vector(bw_hz: np.ndarray, temperature_K: float, noise_figure_db: float) -> np.ndarray:
    bw_hz = np.maximum(bw_hz.astype(float), 1e-9)
    t_ratio = max(float(temperature_K), 1e-9) / 290.0
    return -174.0 + 10.0 * np.log10(bw_hz) + 10.0 * math.log10(t_ratio) + float(noise_figure_db)


def _sample_from_dist_cfg(dist_cfg: Dict[str, Any], n: int) -> np.ndarray:
    """Sample n values from a simple distribution config.

    Supported (backward + YAML-friendly):
      - trunc_normal with {mu, sigma, low/high} or {mu, sigma, min/max}
      - uniform with {low/high} or {min/max}
      - lognormal with {mu_ln, sigma_ln, min/max} (rejection-sampled truncation)
      - point_mass with {value}
      - categorical with {categories, probs}
      - mixture in either format:
          (A) {weights: [...], components: [dist_cfg,...]}
          (B) {components: [{weight: w, ...dist_cfg...}, ...]}
    """
    if n <= 0:
        return np.empty((0,), dtype=float)

    dist = str(dist_cfg.get("dist", "")).lower().strip()

    # ---------- helpers ----------
    def _get_low_high(cfg: Dict[str, Any]) -> Tuple[float, float]:
        low = cfg.get("low", cfg.get("min", -np.inf))
        high = cfg.get("high", cfg.get("max", np.inf))
        return float(low), float(high)

    def _rejection(draw_fn, low: float, high: float) -> np.ndarray:
        out = np.empty((n,), dtype=float)
        k = 0
        # n is small in this simulator; simple rejection is fine
        while k < n:
            draw = draw_fn(n - k)
            if np.isfinite(low):
                draw = draw[draw >= low]
            if np.isfinite(high):
                draw = draw[draw <= high]
            if draw.size == 0:
                continue
            take = min(draw.size, n - k)
            out[k:k + take] = draw[:take]
            k += take
        return out

    # ---------- distributions ----------
    if dist in ("trunc_normal", "truncated_normal"):
        mu = float(dist_cfg["mu"])
        sigma = float(dist_cfg["sigma"])
        low, high = _get_low_high(dist_cfg)
        return _rejection(lambda m: np.random.normal(mu, sigma, size=(m,)), low, high)

    if dist == "uniform":
        low, high = _get_low_high(dist_cfg)
        # Handle infinities defensively
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("uniform dist requires finite low/high (or min/max)")
        return np.random.uniform(low, high, size=(n,)).astype(float)

    if dist == "lognormal":
        mu_ln = float(dist_cfg["mu_ln"])
        sigma_ln = float(dist_cfg["sigma_ln"])
        low, high = _get_low_high(dist_cfg)
        return _rejection(lambda m: np.random.lognormal(mu_ln, sigma_ln, size=(m,)), low, high)

    if dist == "point_mass":
        v = float(dist_cfg["value"])
        return np.full((n,), v, dtype=float)

    if dist == "categorical":
        cats = dist_cfg["categories"]
        probs = np.asarray(dist_cfg["probs"], dtype=float)
        probs = probs / probs.sum()
        idx = np.random.choice(len(cats), size=(n,), replace=True, p=probs)
        # Expect numeric categories for this simulator
        return np.asarray([float(cats[i]) for i in idx], dtype=float)

    if dist == "mixture":
        components = list(dist_cfg["components"])

        # Format A: explicit weights + components
        if "weights" in dist_cfg:
            weights = np.asarray(dist_cfg["weights"], dtype=float)
            weights = weights / weights.sum()
            comp_cfgs = components

        # Format B: each component has its own weight
        else:
            weights = np.asarray([float(c.get("weight", 0.0)) for c in components], dtype=float)
            if np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("mixture components must include positive 'weight' fields when 'weights' is absent.")
            weights = weights / weights.sum()
            comp_cfgs = []
            for c in components:
                c2 = dict(c)
                c2.pop("weight", None)
                comp_cfgs.append(c2)

        comp_idx = np.random.choice(len(comp_cfgs), size=(n,), replace=True, p=weights)
        out = np.empty((n,), dtype=float)
        for i in range(len(comp_cfgs)):
            mask = comp_idx == i
            if not np.any(mask):
                continue
            out[mask] = _sample_from_dist_cfg(comp_cfgs[i], int(mask.sum()))
        return out

    raise ValueError(
        f"Unsupported dist='{dist}'. "
        f"Supported: trunc_normal/uniform/lognormal/point_mass/categorical/mixture."
    )



def generate_pcp_topology(cfg: ResolvedConfig, batch_size: int) -> TopologyData:
    """Generate a Poisson-Cluster-Process-like (PCP) cellular deployment.

    Purpose
    -------
    Provide a *random-like* wide-area micro-cell topology over a large region (e.g., GTA),
    while keeping the downstream tensor shapes identical to the existing hex-grid pipeline.

    Key idea
    --------
    - Parents / hotspots are anchored near the *true* FS receiver locations (ISED SMS),
      so that some BSs are naturally placed close to incumbents (worst-case-like geometry).
    - Offspring are BS *sites* (each expanded to 3 sectors by default).
    - UEs are dropped around each sector within a configurable radius.

    Notes
    -----
    - This layout intentionally does NOT use wrap-around; `bs_virtual_loc` is just
      `bs_loc` repeated over users.
    - FS protection in the receiver is already simultaneous across all FS receivers
      because the solver uses the full FS set (L constraints) jointly.
    """

    topo_cfg = cfg.raw["topology"]
    pcp_cfg = topo_cfg.get("pcp", {}) or {}

    real_dtype = _real_dtype(cfg)

    # --- Derived dimensions (must match downstream expectations)
    u_per_bs = int(topo_cfg["num_ut_per_sector"])
    num_bs = int(cfg.derived.get("num_bs", 0) or 0)
    num_ut = int(cfg.derived.get("num_ut", 0) or 0)
    if num_bs <= 0 or num_ut <= 0:
        raise ValueError("Derived num_bs/num_ut must be available before PCP topology generation.")
    if num_ut != num_bs * u_per_bs:
        raise ValueError(
            f"PCP requires num_ut == num_bs * num_ut_per_sector, got num_ut={num_ut}, "
            f"num_bs={num_bs}, num_ut_per_sector={u_per_bs}."
        )

    # --- Sectorization assumptions (keeps compatibility with FS steering vectors)
    sectors_per_site = int(pcp_cfg.get("sectors_per_site", 3))
    if sectors_per_site <= 0:
        sectors_per_site = 3
    if (num_bs % sectors_per_site) != 0:
        raise ValueError(
            f"topology.pcp.sectors_per_site={sectors_per_site} does not divide num_bs={num_bs}. "
            "Keep num_bs divisible by sectors_per_site."
        )
    num_sites = num_bs // sectors_per_site

    # --- Core scale parameters
    isd_m = float(topo_cfg["isd_m"])
    bs_height = float(topo_cfg["bs_height_m"])
    min_ut_h = float(topo_cfg["min_ut_height_m"])
    max_ut_h = float(topo_cfg["max_ut_height_m"])
    indoor_probability = float(topo_cfg.get("indoor_probability", 0.0))

    # --- PCP knobs (defaults are conservative; tune in YAML)
    use_fs_as_parents = bool(pcp_cfg.get("use_fs_as_parents", True))
    hotspot_jitter_sigma_m = float(pcp_cfg.get("hotspot_jitter_sigma_m", 0.0))
    bs_cluster_sigma_m = float(pcp_cfg.get("bs_cluster_sigma_m", 3.0 * isd_m))
    bs_cluster_radius_max_m = pcp_cfg.get("bs_cluster_radius_max_m", None)
    ue_cluster_radius_m = float(pcp_cfg.get("ue_cluster_radius_m", 0.5 * isd_m))
    background_sites_fraction = float(pcp_cfg.get("background_sites_fraction", 0.0))
    background_bbox_margin_m = float(pcp_cfg.get("background_bbox_margin_m", 0.0))

    # -------------------------
    # Parent / hotspot locations
    # -------------------------
    if use_fs_as_parents:
        fs_cfg = cfg.raw.get("fixed_service", {})
        ised_cfg = fs_cfg.get("ised_sms", {})

        if not (bool(fs_cfg.get("enabled", False)) and bool(ised_cfg.get("enabled", False))):
            raise ValueError(
                "PCP with use_fs_as_parents=True requires fixed_service.enabled=True and "
                "fixed_service.ised_sms.enabled=True."
            )

        # For a true-map GTA overlay we must not rescale/clip FS to the cellular bbox.
        sel = ised_cfg.get("selection", {})
        location_mode = str(sel.get("location_mode", "rescale_to_hexgrid_bbox")).lower().strip()
        if location_mode not in ("raw_meters", "raw", "local_flat", "no_rescale"):
            raise ValueError(
                "PCP requires fixed_service.ised_sms.selection.location_mode='raw_meters' "
                "(true ISED FS locations in local meters)."
            )
        if bool(sel.get("clip_to_bbox", False)):
            raise ValueError(
                "PCP requires fixed_service.ised_sms.selection.clip_to_bbox=False "
                "(do not clip FS to the cellular bbox)."
            )

        num_fs = int(fs_cfg.get("num_receivers", 0))
        if num_fs <= 0:
            raise ValueError("fixed_service.num_receivers must be > 0 for PCP with FS parents.")

        # Use ISED loader to obtain FS receiver xy. In raw_meters mode with no clipping,
        # the cellular bbox is irrelevant, so a dummy topo is sufficient.
        dummy_bs = tf.zeros([1, 1, 3], dtype=real_dtype)
        dummy_ut = tf.zeros([1, 1, 3], dtype=real_dtype)
        dummy_bsv = tf.zeros([1, 1, 1, 3], dtype=real_dtype)
        dummy_topo = TopologyData(
            bs_loc=dummy_bs,
            ut_loc=dummy_ut,
            bs_virtual_loc=dummy_bsv,
            ut_orientations=dummy_ut,
            bs_orientations=dummy_bs,
            ut_velocities=dummy_ut,
            serving_bs=np.array([0], dtype=int),
            ut_is_indoor=tf.zeros([1, 1], dtype=tf.bool),
        )
        fs_tmp = _generate_fs_from_ised_sms(cfg, dummy_topo, batch_size=1, num_fs=num_fs)
        parent_xy = tf.cast(fs_tmp.fs_loc[0, :, 0:2], real_dtype)  # [L,2]
    else:
        # Fallback: single parent at origin
        parent_xy = tf.zeros([1, 2], dtype=real_dtype)

    L = tf.shape(parent_xy)[0]

    # Hotspot centers per batch [batch, L, 2]
    hot_xy = tf.tile(parent_xy[tf.newaxis, ...], [batch_size, 1, 1])
    if hotspot_jitter_sigma_m > 0.0:
        hot_xy = hot_xy + tf.random.normal(tf.shape(hot_xy), dtype=real_dtype) * tf.cast(
            hotspot_jitter_sigma_m, real_dtype
        )

    # -------------------------
    # Offspring: BS site process
    # -------------------------
    bg_frac = float(max(0.0, min(1.0, background_sites_fraction)))
    num_bg = int(round(num_sites * bg_frac))
    num_cluster = int(num_sites - num_bg)

    # Clustered sites
    if num_cluster > 0:
        # uniform parent assignment in {0,...,L-1}
        u = tf.random.uniform([batch_size, num_cluster], dtype=real_dtype)
        parent_idx = tf.cast(tf.floor(u * tf.cast(L, real_dtype)), tf.int32)
        parent_idx = tf.clip_by_value(parent_idx, 0, tf.maximum(L - 1, 0))

        centers = tf.gather(hot_xy, parent_idx, batch_dims=1)  # [batch,num_cluster,2]
        offs = tf.random.normal([batch_size, num_cluster, 2], dtype=real_dtype) * tf.cast(
            bs_cluster_sigma_m, real_dtype
        )

        if bs_cluster_radius_max_m is not None:
            try:
                r_max = float(bs_cluster_radius_max_m)
            except Exception:
                r_max = None
            if (r_max is not None) and (r_max > 0.0):
                r = tf.sqrt(tf.reduce_sum(tf.square(offs), axis=-1, keepdims=True))
                scale = tf.minimum(
                    tf.cast(1.0, real_dtype),
                    tf.cast(r_max, real_dtype) / tf.maximum(r, tf.cast(1e-9, real_dtype)),
                )
                offs = offs * scale

        site_xy_cluster = centers + offs  # [batch,num_cluster,2]
    else:
        site_xy_cluster = tf.zeros([batch_size, 0, 2], dtype=real_dtype)

    # Background sites uniform over FS bbox (+margin)
    if num_bg > 0:
        x_min = tf.reduce_min(parent_xy[:, 0])
        x_max = tf.reduce_max(parent_xy[:, 0])
        y_min = tf.reduce_min(parent_xy[:, 1])
        y_max = tf.reduce_max(parent_xy[:, 1])

        margin = tf.cast(background_bbox_margin_m, real_dtype)
        x_min = x_min - margin
        x_max = x_max + margin
        y_min = y_min - margin
        y_max = y_max + margin

        u_bg = tf.random.uniform([batch_size, num_bg, 2], dtype=real_dtype)
        site_xy_bg = tf.stack(
            [
                x_min + (x_max - x_min) * u_bg[..., 0],
                y_min + (y_max - y_min) * u_bg[..., 1],
            ],
            axis=-1,
        )
    else:
        site_xy_bg = tf.zeros([batch_size, 0, 2], dtype=real_dtype)

    site_xy = tf.concat([site_xy_cluster, site_xy_bg], axis=1)  # [batch,num_sites,2]
    site_xy = site_xy[:, :num_sites, :]  # safety slice


        # ------------------------------------------------------------
    # Optional: land/water half-plane constraint for BS sites
    # This is a simple geometric filter: define a line and keep sites
    # on a chosen side. Any site on the wrong side is reflected
    # across the line (fast, deterministic, preserves count).
    #
    # Config (under topology.pcp):
    #   land_constraint:
    #     enabled: true/false
    #     keep_side: "left" or "right"   (relative to p1->p2 direction)
    #     line_xy_m: [[x1,y1],[x2,y2]]   OR
    #     line_latlon_wgs84: [[lat1,lon1],[lat2,lon2]]
    # ------------------------------------------------------------
    land_cfg = pcp_cfg.get("land_constraint", {}) or {}
    if bool(land_cfg.get("enabled", False)):
        keep_side = str(land_cfg.get("keep_side", "left")).lower().strip()

        # Resolve line endpoints in local meters
        p1_xy = None
        p2_xy = None

        line_xy_m = land_cfg.get("line_xy_m", None)
        line_ll = land_cfg.get("line_latlon_wgs84", None)

        if line_xy_m is not None:
            if (not isinstance(line_xy_m, (list, tuple))) or len(line_xy_m) != 2:
                raise ValueError("topology.pcp.land_constraint.line_xy_m must be [[x1,y1],[x2,y2]]")
            x1, y1 = float(line_xy_m[0][0]), float(line_xy_m[0][1])
            x2, y2 = float(line_xy_m[1][0]), float(line_xy_m[1][1])
            p1_xy = tf.constant([x1, y1], dtype=real_dtype)
            p2_xy = tf.constant([x2, y2], dtype=real_dtype)

        elif line_ll is not None:
            if (not isinstance(line_ll, (list, tuple))) or len(line_ll) != 2:
                raise ValueError("topology.pcp.land_constraint.line_latlon_wgs84 must be [[lat1,lon1],[lat2,lon2]]")

            # Use the same local flat-earth conversion as the ISED loader (raw_meters)
            fs_cfg = cfg.raw.get("fixed_service", {}) or {}
            ised_cfg = fs_cfg.get("ised_sms", {}) or {}
            sel = ised_cfg.get("selection", {}) or {}

            origin = sel.get("origin_latlon_wgs84", None)
            if not (isinstance(origin, (list, tuple)) and len(origin) == 2):
                raise ValueError(
                    "topology.pcp.land_constraint.line_latlon_wgs84 requires "
                    "fixed_service.ised_sms.selection.origin_latlon_wgs84"
                )
            center_lat = float(origin[0])
            center_lon = float(origin[1])

            ct = ised_cfg.get("coordinate_transform", {}) or {}
            scale_m = float(ct.get("scaling_m_per_degree", 111_320.0))
            cos_lat = math.cos(math.radians(center_lat))

            def _ll_to_xy(latlon):
                lat = float(latlon[0])
                lon = float(latlon[1])
                y = (lat - center_lat) * scale_m
                x = (lon - center_lon) * scale_m * cos_lat
                return x, y

            x1, y1 = _ll_to_xy(line_ll[0])
            x2, y2 = _ll_to_xy(line_ll[1])
            p1_xy = tf.constant([x1, y1], dtype=real_dtype)
            p2_xy = tf.constant([x2, y2], dtype=real_dtype)

        else:
            raise ValueError(
                "topology.pcp.land_constraint.enabled=True requires either "
                "line_xy_m or line_latlon_wgs84"
            )

        # Validate line
	        # Validate line (eager + graph-safe)
        v = p2_xy - p1_xy
        v_norm = tf.sqrt(tf.reduce_sum(tf.square(v)))

        # Eager-friendly error
        if tf.executing_eagerly():
            dx = float(v[0].numpy())
            dy = float(v[1].numpy())
            if (dx * dx + dy * dy) < 1e-6:
                raise ValueError("topology.pcp.land_constraint line points are too close / degenerate")

        # Graph-safe assertion (covers tf.function / tf.compile)
        tf.debugging.assert_greater(
            v_norm,
            tf.cast(1.0, real_dtype),  # >= 1 meter
            message=(
                "topology.pcp.land_constraint line points are too close / degenerate. "
                "Set topology.pcp.land_constraint.line_latlon_wgs84 or line_xy_m to two distinct points."
            ),
        )

        # Unit normal pointing to the LEFT of direction (p1->p2)
        eps = tf.cast(1e-9, real_dtype)
        n_left = tf.stack([-v[1], v[0]], axis=0) / tf.maximum(v_norm, eps)  # [2]


	
	        # Signed distance to the line: d>0 => left side, d<0 => right side
        d = tf.reduce_sum((site_xy - p1_xy) * n_left, axis=-1, keepdims=True)  # [batch,num_sites,1]

        if keep_side == "left":
            bad_reflect = d < 0.0
        elif keep_side == "right":
            bad_reflect = d > 0.0
        else:
            raise ValueError("topology.pcp.land_constraint.keep_side must be 'left' or 'right'")

        # Reflect points that are on the wrong side across the line (preserves count)
        site_xy = tf.where(bad_reflect, site_xy - 2.0 * d * n_left, site_xy)

        # Optional inland buffer: push any site that ends up too close to the boundary further
        # into the kept side. This helps with curved shorelines + flat-earth overlay offsets.
        buffer_m = float(land_cfg.get("buffer_m", 0.0) or 0.0)
        if buffer_m > 0.0:
            buf = tf.cast(buffer_m, real_dtype)
            d2 = tf.reduce_sum((site_xy - p1_xy) * n_left, axis=-1, keepdims=True)

            if keep_side == "left":
                need_push = d2 < buf
                site_xy = tf.where(need_push, site_xy + (buf - d2) * n_left, site_xy)
            else:  # keep_side == "right"
                need_push = d2 > -buf
                site_xy = tf.where(need_push, site_xy + (-buf - d2) * n_left, site_xy)


	

    # Expand sites -> sectors (keep consecutive triples per site)
    bs_xy = tf.tile(site_xy[:, :, tf.newaxis, :], [1, 1, sectors_per_site, 1])
    bs_xy = tf.reshape(bs_xy, [batch_size, num_bs, 2])

    bs_z = tf.fill([batch_size, num_bs, 1], tf.cast(bs_height, real_dtype))
    bs_loc = tf.concat([bs_xy, bs_z], axis=-1)  # [batch,B,3]

    # -------------------------
    # Offspring: UE process
    # -------------------------
    R = tf.cast(ue_cluster_radius_m, real_dtype)
    u1 = tf.random.uniform([batch_size, num_bs, u_per_bs], dtype=real_dtype)
    u2 = tf.random.uniform([batch_size, num_bs, u_per_bs], dtype=real_dtype)
    r = R * tf.sqrt(u1)
    theta = tf.cast(2.0 * math.pi, real_dtype) * u2
    dx = r * tf.cos(theta)
    dy = r * tf.sin(theta)

    ut_xy = bs_xy[:, :, tf.newaxis, :] + tf.stack([dx, dy], axis=-1)  # [batch,B,u_per_bs,2]
    ut_xy = tf.reshape(ut_xy, [batch_size, num_bs * u_per_bs, 2])

    u_h = tf.random.uniform([batch_size, num_bs * u_per_bs, 1], dtype=real_dtype)
    ut_z = tf.cast(min_ut_h, real_dtype) + (tf.cast(max_ut_h, real_dtype) - tf.cast(min_ut_h, real_dtype)) * u_h
    ut_loc = tf.concat([ut_xy, ut_z], axis=-1)  # [batch,U,3]

    # No wrap-around in wide-area PCP
    bs_virtual_loc = tf.tile(bs_loc[:, :, tf.newaxis, :], [1, 1, tf.shape(ut_loc)[1], 1])

    serving_bs = np.repeat(np.arange(num_bs), u_per_bs)

    # Indoor mask (rank-2 [batch,U])
    u_ind = tf.random.uniform([batch_size, num_bs * u_per_bs], dtype=real_dtype)
    ut_is_indoor = u_ind < tf.cast(indoor_probability, real_dtype)

    ut_orient = tf.zeros_like(ut_loc)
    bs_orient = tf.zeros_like(bs_loc)
    ut_vel = tf.zeros_like(ut_loc)

    return TopologyData(
        bs_loc=bs_loc,
        ut_loc=ut_loc,
        bs_virtual_loc=bs_virtual_loc,
        ut_orientations=ut_orient,
        bs_orientations=bs_orient,
        ut_velocities=ut_vel,
        serving_bs=serving_bs,
        ut_is_indoor=ut_is_indoor,
    )



# -----------------------
# Public API
# -----------------------

def generate_hexgrid_topology(cfg: ResolvedConfig, batch_size: int) -> TopologyData:
    topo_cfg = cfg.raw["topology"]

    layout = str(topo_cfg.get("layout", topo_cfg.get("mode", "hexgrid"))).lower().strip()
    if layout in ("pcp", "poisson_cluster", "poisson_cluster_process", "cluster_process"):
        return generate_pcp_topology(cfg, batch_size)


    scenario = str(topo_cfg["scenario"])
    num_rings = int(topo_cfg["num_rings"])
    u_per_bs = int(topo_cfg["num_ut_per_sector"])
    isd_m = float(topo_cfg["isd_m"])
    bs_height = float(topo_cfg["bs_height_m"])
    min_ut_h = float(topo_cfg["min_ut_height_m"])
    max_ut_h = float(topo_cfg["max_ut_height_m"])
    indoor_probability = float(topo_cfg.get("indoor_probability", 0.0))

    enable_wrap = bool(topo_cfg.get("enable_wrap_around", True))
    downtilt = bool(topo_cfg.get("downtilt_to_sector_center", True))

    gen_hex = _import_gen_hexgrid_topology()

    out = _call_with_compatible_kwargs(
        gen_hex,
        (
            (("batch_size",), int(batch_size)),
            (("scenario",), scenario),
            (("num_rings",), int(num_rings)),
            (("num_ut_per_sector", "n_ut_per_sector"), int(u_per_bs)),
            (("cell_isd", "cell_isd_m", "isd_m", "isd"), float(isd_m)),
            (("bs_height", "bs_height_m"), float(bs_height)),
            (("min_ut_height", "min_ut_height_m"), float(min_ut_h)),
            (("max_ut_height", "max_ut_height_m"), float(max_ut_h)),
            (("indoor_probability", "indoor_prob"), float(indoor_probability)),
            (("enable_wrap_around", "wrap_around"), bool(enable_wrap)),
            (("downtilt_to_sector_center", "downtilt"), bool(downtilt)),
        ),
    )

    # Sionna's `gen_hexgrid_topology` return signature has changed across versions.
    # Some versions return 6 tensors (UT/BS locations + orientations/velocities),
    # others return extra metadata (e.g., indoor state), and some versions move
    # `bs_virtual_loc` to a different position.
    if not isinstance(out, (tuple, list)):
        raise RuntimeError(
            "gen_hexgrid_topology returned an unexpected type "
            f"{type(out)} (expected tuple/list)."
        )
    if len(out) < 2:
        raise RuntimeError(f"gen_hexgrid_topology returned too few outputs: len={len(out)} (<2)")

    out_list = list(out)

    def _is_rank3_xyz(t: Any) -> bool:
        return isinstance(t, tf.Tensor) and t.shape.rank == 3 and (t.shape[-1] in (3, None))

    def _dim_equals(t: tf.Tensor, axis: int, value: int) -> bool:
        if value is None:
            return False
        try:
            s = t.shape[axis]
            if s is not None:
                return int(s) == int(value)
        except Exception:
            pass
        try:
            if tf.executing_eagerly():
                return int(tf.shape(t)[axis].numpy()) == int(value)
        except Exception:
            pass
        return False

    exp_B = int(cfg.derived.get("num_bs", 0) or 0)
    exp_U = int(cfg.derived.get("num_ut", 0) or 0)

    # Prefer legacy order IF it matches expected dims; else search by dims.
    ut_loc = out_list[0]
    bs_loc = out_list[1]

    if not (_is_rank3_xyz(ut_loc) and (exp_U <= 0 or _dim_equals(ut_loc, 1, exp_U))):
        ut_loc = None
        for t in out_list:
            if _is_rank3_xyz(t) and exp_U > 0 and _dim_equals(t, 1, exp_U):
                ut_loc = t
                break

    if not (_is_rank3_xyz(bs_loc) and (exp_B <= 0 or _dim_equals(bs_loc, 1, exp_B))):
        bs_loc = None
        for t in out_list:
            if _is_rank3_xyz(t) and exp_B > 0 and _dim_equals(t, 1, exp_B):
                bs_loc = t
                break

    if ut_loc is None:
        ut_loc = out_list[0]
    if bs_loc is None:
        bs_loc = out_list[1]

    if not _is_rank3_xyz(ut_loc) or not _is_rank3_xyz(bs_loc):
        raise RuntimeError(
            "Could not identify (ut_loc, bs_loc) as rank-3 xyz tensors from gen_hexgrid_topology outputs. "
            f"Got ut_loc rank={getattr(getattr(ut_loc, 'shape', None), 'rank', None)}, "
            f"bs_loc rank={getattr(getattr(bs_loc, 'shape', None), 'rank', None)}."
        )

    # Resolve B/U from selected tensors (eager-safe)
    B = int(bs_loc.shape[1]) if bs_loc.shape[1] is not None else int(tf.shape(bs_loc)[1].numpy())
    U = int(ut_loc.shape[1]) if ut_loc.shape[1] is not None else int(tf.shape(ut_loc)[1].numpy())

    # --- bs_virtual_loc: normalize to [batch, B, U, 3]
    bs_virtual_loc = None
    for cand in out_list:
        if not isinstance(cand, tf.Tensor):
            continue
        if cand.shape.rank != 4:
            continue
        if cand.shape[-1] not in (3, None):
            continue

        # Common conventions:
        #   [batch, B, U, 3]  (expected downstream)
        #   [batch, U, B, 3]  (swap B/U)
        if _dim_equals(cand, 1, B) and _dim_equals(cand, 2, U):
            bs_virtual_loc = cand
            break
        if _dim_equals(cand, 1, U) and _dim_equals(cand, 2, B):
            bs_virtual_loc = tf.transpose(cand, [0, 2, 1, 3])
            break

    if bs_virtual_loc is None:
        # Fallback when Sionna doesn't provide virtual locations:
        bs_virtual_loc = tf.tile(bs_loc[:, :, tf.newaxis, :], [1, 1, tf.shape(ut_loc)[1], 1])

        # ------------------------------------------------------------
    # Optional: translate the entire cellular layout in the (x,y) plane.
    # Keeps ISD / cell radius unchanged; only changes absolute placement.
    # ------------------------------------------------------------
    xy_offset = topo_cfg.get("xy_offset_m", None)
    if xy_offset is not None:
        try:
            dx = float(xy_offset[0])
            dy = float(xy_offset[1])
        except Exception:
            dx, dy = 0.0, 0.0

        if (dx != 0.0) or (dy != 0.0):
            off = tf.constant([dx, dy, 0.0], dtype=bs_loc.dtype)
            bs_loc = bs_loc + off
            ut_loc = ut_loc + tf.cast(off, ut_loc.dtype)
            bs_virtual_loc = bs_virtual_loc + tf.cast(off, bs_virtual_loc.dtype)


    # --- Optional indoor-state tensor (rank-2 [batch,U] or rank-3 [batch,U,1])
    in_state = None
    for cand in out_list:
        if not isinstance(cand, tf.Tensor):
            continue
        if cand is ut_loc or cand is bs_loc:
            continue
        r = cand.shape.rank
        if r not in (2, 3):
            continue
        if not _dim_equals(cand, 1, U):
            continue
        if r == 3 and (cand.shape[-1] not in (1, None)):
            continue
        try:
            if tf.executing_eagerly() and r == 3 and int(tf.shape(cand)[-1].numpy()) != 1:
                continue
        except Exception:
            pass
        in_state = cand
        break

    num_bs = int(B)
    serving_bs = np.repeat(np.arange(num_bs), u_per_bs)

    # Always provide a `ut_is_indoor` tensor for downstream O2I reproducibility.
    if in_state is not None:
        t = tf.cast(in_state, tf.bool)
        if t.shape.rank == 3 and int(t.shape[-1]) == 1:
            t = tf.squeeze(t, axis=-1)
        ut_is_indoor = t
    else:
        real_dtype = _real_dtype(cfg)
        shape = tf.stack([tf.constant(int(batch_size), dtype=tf.int32), tf.shape(ut_loc)[1]])
        u = tf.random.uniform(shape, dtype=real_dtype)
        ut_is_indoor = u < tf.cast(indoor_probability, real_dtype)

    # Preserve any additional rank-3 [batch,U,3] / [batch,B,3] tensors if present; else safe defaults.
    ut_orient = tf.zeros_like(ut_loc)
    bs_orient = tf.zeros_like(bs_loc)
    ut_vel = tf.zeros_like(ut_loc)

    # If Sionna returned additional rank-3 xyz tensors, reuse them (best-effort)
    rank3_xyz = [t for t in out_list if _is_rank3_xyz(t)]
    ut_xyz = [t for t in rank3_xyz if _dim_equals(t, 1, U) and t is not ut_loc]
    bs_xyz = [t for t in rank3_xyz if _dim_equals(t, 1, B) and t is not bs_loc]
    if len(ut_xyz) >= 1:
        ut_orient = ut_xyz[0]
    if len(bs_xyz) >= 1:
        bs_orient = bs_xyz[0]
    if len(ut_xyz) >= 2:
        ut_vel = ut_xyz[1]

    return TopologyData(
        bs_loc=bs_loc,
        ut_loc=ut_loc,
        bs_virtual_loc=bs_virtual_loc,
        ut_orientations=ut_orient,
        bs_orientations=bs_orient,
        ut_velocities=ut_vel,
        serving_bs=serving_bs,
        ut_is_indoor=ut_is_indoor,
    )





def generate_fixed_service_locations(cfg: ResolvedConfig, topo: TopologyData, batch_size: int) -> FixedServiceLocations:
    fs_cfg = cfg.raw.get("fixed_service", {})
    if not bool(fs_cfg.get("enabled", False)):
        raise ValueError("fixed_service.enabled is False, but FS generation was requested.")

    num_fs = int(fs_cfg.get("num_receivers", 0))
    if num_fs <= 0:
        raise ValueError("fixed_service.num_receivers must be > 0 when fixed_service.enabled=True")

    ised_cfg = fs_cfg.get("ised_sms", {})
    if bool(ised_cfg.get("enabled", False)):
        return _generate_fs_from_ised_sms(cfg, topo, batch_size, num_fs)

    params_cfg = fs_cfg.get("parameters", {})
    if str(params_cfg.get("source", "")).lower().strip() not in ("", "none", "null"):
        return _generate_fs_from_parameter_fits(cfg, topo, batch_size, num_fs)

    return _generate_fs_uniform_in_bbox(cfg, topo, batch_size, num_fs)


def _get_sim_bbox_xy(topo: TopologyData, margin_m: float) -> Tuple[float, float, float, float]:
    bs_xy = topo.bs_loc[0, :, :2].numpy()
    x_min = float(np.min(bs_xy[:, 0])) - float(margin_m)
    x_max = float(np.max(bs_xy[:, 0])) + float(margin_m)
    y_min = float(np.min(bs_xy[:, 1])) - float(margin_m)
    y_max = float(np.max(bs_xy[:, 1])) + float(margin_m)
    return x_min, x_max, y_min, y_max


def _generate_fs_uniform_in_bbox(cfg: ResolvedConfig, topo: TopologyData, batch_size: int, num_fs: int) -> FixedServiceLocations:
    fs_cfg = cfg.raw["fixed_service"]
    placement = fs_cfg.get("placement", {})
    margin_m = float(placement.get("margin_m", 0.0))
    x_min, x_max, y_min, y_max = _get_sim_bbox_xy(topo, margin_m)

    real_dtype = _real_dtype(cfg)
    x = tf.random.uniform([num_fs], minval=x_min, maxval=x_max, dtype=real_dtype)
    y = tf.random.uniform([num_fs], minval=y_min, maxval=y_max, dtype=real_dtype)
    z = tf.fill([num_fs], tf.cast(20.0, real_dtype))

    fs_one = tf.stack([x, y, z], axis=-1)
    fs_loc = tf.tile(fs_one[tf.newaxis, ...], [batch_size, 1, 1])
    return FixedServiceLocations(fs_loc=fs_loc)


def _generate_fs_from_ised_sms(cfg: ResolvedConfig, topo: TopologyData, batch_size: int, num_fs: int) -> FixedServiceLocations:
    """Generate FS receivers by sampling real ISED SMS records.

    Key behaviors (all driven by YAML under fixed_service.ised_sms.selection)
    -----------------------------------------------------------------------
    - GTA bbox filter (lat/lon)
    - station_function regex filter (RX)
    - optional frequency range filter (freq_range_mhz)
    - optional overlap-with-gNB-band filter
    - location_mode:
        * rescale_to_hexgrid_bbox (default): map GTA points to sim bbox (scale+translate)
        * raw_meters: local flat-earth meters about origin_latlon_wgs84 (no rescale)
    - sample_mode:
        * random_without_replacement (default)
        * random_with_replacement
    - Fallbacks for missing FS record fields:
        * if fixed_service.parameters.<field> distribution exists, sample from it
        * else use conservative defaults (keeps sim running)
    - TI objective:
        * if radio_reference_csv present, prefer deterministic join on (authorization_number, reference_id)
        * fallback to sampling its empirical distribution
        * final fallback to fixed_service.parameters.ti_objective_db or fixed_service.i_max.fallback_ti_objective_db
    - Optional antenna_reference_csv:
        * used to fill missing antenna_gain_dbi and beamwidth_3db_deg (horizontal_beamwidth_deg) if available
    """

    fs_cfg = cfg.raw["fixed_service"]
    ised_cfg = fs_cfg["ised_sms"]
    fields = ised_cfg.get("fields", {}) or {}
    sel = ised_cfg.get("selection", {}) or {}

    params_cfg = fs_cfg.get("parameters", {}) or {}
    i_cfg = fs_cfg.get("i_max", {}) or {}

    fs_csv = _resolve_path(cfg, str(ised_cfg["fixed_service_csv"]))
    rr_csv = _resolve_path(cfg, str(ised_cfg.get("radio_reference_csv", ""))) if ised_cfg.get("radio_reference_csv") else None
    ant_csv = _resolve_path(cfg, str(ised_cfg.get("antenna_reference_csv", ""))) if ised_cfg.get("antenna_reference_csv") else None

    fs_df = _read_csv_cached(str(fs_csv)).copy()

    # ---- Column names (overrideable via fixed_service.ised_sms.fields)
    lat_col = str(fields.get("latitude_wgs84", "latitude_wgs84"))
    lon_col = str(fields.get("longitude_wgs84", "longitude_wgs84"))
    sf_col = str(fields.get("station_function", "station_function"))
    freq_col = str(fields.get("frequency_mhz", "frequency_mhz"))
    bw_col = str(fields.get("occupied_bw_khz", "occupied_bw_khz"))
    h_col = str(fields.get("height_agl_m", "height_agl_m"))
    gain_col = str(fields.get("antenna_gain_dbi", "antenna_gain_dbi"))
    bwdeg_col = str(fields.get("beamwidth_3db_deg", "beamwidth_3db_deg"))
    az_col = str(fields.get("azimuth_deg", "azimuth_deg"))
    rxthr_col = str(fields.get("rx_threshold_dbw_ber1e3", "rx_threshold_dbw_ber1e3"))
    auth_col = str(fields.get("authorization_number", "authorization_number"))
    ref_col = str(fields.get("reference_id", "reference_id"))
    man_col = str(fields.get("antenna_manufacturer", "antenna_manufacturer"))
    model_col = str(fields.get("antenna_model", "antenna_model"))

    # ---- Basic numeric coercions
    if lat_col in fs_df.columns:
        fs_df[lat_col] = _to_numeric(fs_df[lat_col])
    if lon_col in fs_df.columns:
        fs_df[lon_col] = _to_numeric(fs_df[lon_col])
    if freq_col in fs_df.columns:
        fs_df[freq_col] = _to_numeric(fs_df[freq_col])
    if bw_col in fs_df.columns:
        fs_df[bw_col] = _to_numeric(fs_df[bw_col])

    # ---- GTA bbox filter
    bbox = sel.get("gta_bbox", {}) or {}
    lat_min = float(bbox.get("lat_min", -90.0))
    lat_max = float(bbox.get("lat_max", 90.0))
    lon_min = float(bbox.get("lon_min", -180.0))
    lon_max = float(bbox.get("lon_max", 180.0))

    if (lat_col in fs_df.columns) and (lon_col in fs_df.columns):
        mask_bbox = fs_df[lat_col].between(lat_min, lat_max) & fs_df[lon_col].between(lon_min, lon_max)
    else:
        raise RuntimeError(f"ISED FS CSV missing required lat/lon columns: '{lat_col}', '{lon_col}'")

    # ---- RX filter (regex)
    rx_regex = str(sel.get("station_function_rx_regex", sel.get("station_function_regex", "RX")))
    if sf_col in fs_df.columns:
        mask_rx = _station_function_mask(fs_df, sf_col, rx_regex)
    else:
        mask_rx = True

    df = fs_df.loc[mask_bbox & mask_rx].copy()
    if len(df) == 0:
        raise RuntimeError("No FS records after GTA bbox + RX filter")

    # ---- Optional frequency range filter (MHz)
    fr = sel.get("freq_range_mhz", None)
    if fr is not None and freq_col in df.columns:
        try:
            fmin = float(fr[0])
            fmax = float(fr[1])
            if fmax < fmin:
                fmin, fmax = fmax, fmin
            df = df.loc[df[freq_col].between(fmin, fmax)].copy()
        except Exception:
            pass

    if len(df) == 0:
        raise RuntimeError("No FS records left after freq_range_mhz filter")

    # ---- Overlap filter (optional)
    if bool(sel.get("overlap_with_gnb_band", False)) and (freq_col in df.columns) and (bw_col in df.columns):
        bw_total = float(cfg.derived["bandwidth_total_hz"])
        f_c = float(cfg.raw["system_model"]["carrier_frequency_hz"])
        gnb_min = f_c - bw_total / 2.0
        gnb_max = f_c + bw_total / 2.0

        f_hz = df[freq_col].to_numpy(dtype=float) * 1e6
        bw_hz = df[bw_col].to_numpy(dtype=float) * 1e3
        fs_min = f_hz - bw_hz / 2.0
        fs_max = f_hz + bw_hz / 2.0
        df = df.loc[(fs_max >= gnb_min) & (fs_min <= gnb_max)].copy()

    if len(df) == 0:
        raise RuntimeError("No FS records left after overlap-with-gNB-band filter")

    # ---- Convert lat/lon to local x/y meters
    origin = sel.get("origin_latlon_wgs84", None)
    if isinstance(origin, (list, tuple)) and len(origin) == 2:
        center_lat = float(origin[0])
        center_lon = float(origin[1])
    else:
        center_lat = 0.5 * (lat_min + lat_max)
        center_lon = 0.5 * (lon_min + lon_max)

    ct = ised_cfg.get("coordinate_transform", {}) or {}
    scale_m = float(ct.get("scaling_m_per_degree", 111_320.0))
    cos_lat = math.cos(math.radians(center_lat))

    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    y_raw = (lat - center_lat) * scale_m
    x_raw = (lon - center_lon) * scale_m * cos_lat

    placement = fs_cfg.get("placement", {}) or {}
    margin_m = float(placement.get("margin_m", 0.0))
    x_min_sim, x_max_sim, y_min_sim, y_max_sim = _get_sim_bbox_xy(topo, margin_m)

    location_mode = str(sel.get("location_mode", "rescale_to_hexgrid_bbox")).lower().strip()
    if location_mode in ("raw_meters", "raw", "local_flat", "no_rescale"):
        x_m, y_m = x_raw, y_raw
    else:
        x_min_d, x_max_d = float(np.min(x_raw)), float(np.max(x_raw))
        y_min_d, y_max_d = float(np.min(y_raw)), float(np.max(y_raw))
        w_d = max(x_max_d - x_min_d, 1e-9)
        h_d = max(y_max_d - y_min_d, 1e-9)
        w_s = max(x_max_sim - x_min_sim, 1e-9)
        h_s = max(y_max_sim - y_min_sim, 1e-9)
        scale = min(w_s / w_d, h_s / h_d)

        x_c_d = 0.5 * (x_min_d + x_max_d)
        y_c_d = 0.5 * (y_min_d + y_max_d)
        x_c_s = 0.5 * (x_min_sim + x_max_sim)
        y_c_s = 0.5 * (y_min_sim + y_max_sim)

        x_m = (x_raw - x_c_d) * scale + x_c_s
        y_m = (y_raw - y_c_d) * scale + y_c_s

    if bool(sel.get("clip_to_bbox", True)):
        x_m = np.clip(x_m, x_min_sim, x_max_sim)
        y_m = np.clip(y_m, y_min_sim, y_max_sim)

    # ---- Sampling mode
    sample_mode = str(sel.get("sample_mode", "")).lower().strip()
    if sample_mode in ("random_with_replacement", "with_replacement", "bootstrap", "replace"):
        replace = True
    elif sample_mode in ("random_without_replacement", "without_replacement", "wout_replacement", "no_replace"):
        replace = False
    else:
        replace = bool(sel.get("sample_with_replacement", False))
        if ("random_sample_without_replacement" in sel) and ("sample_with_replacement" not in sel):
            replace = not bool(sel.get("random_sample_without_replacement", True))

    import warnings  # <-- add at top of file if not already imported

# If user asked for "without replacement" but the filtered dataset is too small,
# do NOT silently duplicate the same FS record unless explicitly allowed.
    if (not replace) and (num_fs > len(df)):
        allow = bool(sel.get("allow_replacement_when_insufficient_records", False))
        msg = (
            f"ISED FS selection produced only {len(df)} eligible records after filters, "
            f"but fixed_service.num_receivers={num_fs} and sample_mode=without_replacement. "
            "Either relax filters (e.g., overlap_with_gnb_band / bbox / freq_range_mhz), "
            "or set fixed_service.ised_sms.selection.allow_replacement_when_insufficient_records=true."
        )
        if not allow:
            raise RuntimeError(msg)
        warnings.warn(msg + " Proceeding with sampling WITH replacement.", RuntimeWarning)
        replace = True


    seed = sel.get("random_seed", None)
    if seed is None:
        idx = np.random.choice(len(df), size=(num_fs,), replace=replace)
    else:
        try:
            rng = np.random.default_rng(int(seed))
        except Exception:
            rng = np.random.default_rng()
        idx = rng.choice(len(df), size=(num_fs,), replace=replace)

    x_sel = x_m[idx]
    y_sel = y_m[idx]

    # ---- Extract numeric arrays (selected), allowing NaNs for later fallback
    def _get_sel(col: str) -> np.ndarray:
        if col in df.columns:
            return _to_numeric(df[col]).to_numpy(dtype=float)[idx]
        return np.full((num_fs,), np.nan, dtype=float)

    fs_freq_mhz = _get_sel(freq_col)
    fs_bw_khz = _get_sel(bw_col)
    fs_h_m = _get_sel(h_col)
    fs_gain_dbi = _get_sel(gain_col)
    fs_bwdeg = _get_sel(bwdeg_col)
    fs_az = _get_sel(az_col)
    fs_rxthr_dbw = _get_sel(rxthr_col)

    # ---- Optional antenna reference fill (gain + horizontal beamwidth) if missing in FS record
    if ant_csv is not None and ant_csv.exists() and (man_col in df.columns) and (model_col in df.columns):
        ant_df = _read_csv_cached(str(ant_csv)).copy()

        if (man_col in ant_df.columns) and (model_col in ant_df.columns):
            left = df.iloc[idx][[man_col, model_col]].copy()
            left[man_col] = left[man_col].astype(str).str.strip()
            left[model_col] = left[model_col].astype(str).str.strip()

            right_cols = [man_col, model_col]
            if "antenna_gain_dbi" in ant_df.columns:
                right_cols.append("antenna_gain_dbi")
            if "horizontal_beamwidth_deg" in ant_df.columns:
                right_cols.append("horizontal_beamwidth_deg")

            right = ant_df[right_cols].copy()
            right[man_col] = right[man_col].astype(str).str.strip()
            right[model_col] = right[model_col].astype(str).str.strip()
            if "antenna_gain_dbi" in right.columns:
                right["antenna_gain_dbi"] = _to_numeric(right["antenna_gain_dbi"])
            if "horizontal_beamwidth_deg" in right.columns:
                right["horizontal_beamwidth_deg"] = _to_numeric(right["horizontal_beamwidth_deg"])

            right = right.drop_duplicates(subset=[man_col, model_col], keep="first")
            joined = left.merge(right, on=[man_col, model_col], how="left")

            if "antenna_gain_dbi" in joined.columns:
                g_ref = joined["antenna_gain_dbi"].to_numpy(dtype=float)
                bad = ~np.isfinite(fs_gain_dbi)
                fill = bad & np.isfinite(g_ref)
                fs_gain_dbi[fill] = g_ref[fill]

            if "horizontal_beamwidth_deg" in joined.columns:
                bw_ref = joined["horizontal_beamwidth_deg"].to_numpy(dtype=float)
                bad = ~np.isfinite(fs_bwdeg)
                fill = bad & np.isfinite(bw_ref)
                fs_bwdeg[fill] = bw_ref[fill]

    # ---- Fallback behavior: fill missing values using fixed_service.parameters fits (if provided)
    def _fill_invalid(vals: np.ndarray, key: str, default: float) -> np.ndarray:
        vals = np.asarray(vals, dtype=float)
        bad = ~np.isfinite(vals)
        if not np.any(bad):
            return vals
        if isinstance(params_cfg, dict) and (key in params_cfg) and isinstance(params_cfg[key], dict) and ("dist" in params_cfg[key]):
            vals[bad] = _sample_from_dist_cfg(params_cfg[key], int(bad.sum()))
        else:
            vals[bad] = float(default)
        return vals

    fs_freq_mhz = _fill_invalid(fs_freq_mhz, "frequency_mhz", float(cfg.raw["system_model"]["carrier_frequency_hz"]) / 1e6)
    fs_bw_khz = _fill_invalid(fs_bw_khz, "occupied_bw_khz", float(fs_cfg.get("rx_bandwidth_hz", 1e6)) / 1e3)
    fs_h_m = _fill_invalid(fs_h_m, "height_agl_m", 20.0)
    fs_gain_dbi = _fill_invalid(fs_gain_dbi, "antenna_gain_dbi", 0.0)
    fs_bwdeg = _fill_invalid(fs_bwdeg, "beamwidth_3db_deg", 30.0)
    fs_az = _fill_invalid(fs_az, "azimuth_deg", np.random.uniform(0.0, 360.0))
    fs_az = _wrap_deg(fs_az)

    fs_rxthr_dbw = np.asarray(fs_rxthr_dbw, dtype=float)

    # ---- TI objective (prefer deterministic join with radio reference, then fallback to sampling)
    ti_obj = np.full((num_fs,), np.nan, dtype=float)
    ti_source = str(i_cfg.get("ti_objective_source", "radio_reference_csv")).lower().strip()

    rr_df = None
    if (ti_source in ("radio_reference_csv", "radio_reference", "radio", "rr")) and (rr_csv is not None) and rr_csv.exists():
        rr_df = _read_csv_cached(str(rr_csv)).copy()

        if ("ti_objective_db" in rr_df.columns) and (auth_col in df.columns) and (ref_col in df.columns) and (auth_col in rr_df.columns) and (ref_col in rr_df.columns):
            left = df.iloc[idx][[auth_col, ref_col]].copy()
            left[auth_col] = left[auth_col].astype(str)
            left[ref_col] = left[ref_col].astype(str)

            right = rr_df[[auth_col, ref_col, "ti_objective_db"]].copy()
            right[auth_col] = right[auth_col].astype(str)
            right[ref_col] = right[ref_col].astype(str)
            right["ti_objective_db"] = _to_numeric(right["ti_objective_db"])
            right = right.dropna(subset=["ti_objective_db"])
            right = right.drop_duplicates(subset=[auth_col, ref_col], keep="first")

            joined = left.merge(right, on=[auth_col, ref_col], how="left")
            ti_obj = _to_numeric(joined["ti_objective_db"]).to_numpy(dtype=float)

        missing = ~np.isfinite(ti_obj)
        if np.any(missing) and ("ti_objective_db" in rr_df.columns):
            ti_vals = _to_numeric(rr_df["ti_objective_db"]).dropna().to_numpy(dtype=float)
            if ti_vals.size > 0:
                ti_obj[missing] = np.random.choice(ti_vals, size=int(missing.sum()), replace=True)

    missing = ~np.isfinite(ti_obj)
    if np.any(missing):
        if isinstance(params_cfg, dict) and ("ti_objective_db" in params_cfg) and isinstance(params_cfg["ti_objective_db"], dict) and ("dist" in params_cfg["ti_objective_db"]):
            ti_obj[missing] = _sample_from_dist_cfg(params_cfg["ti_objective_db"], int(missing.sum()))
        else:
            ti_obj[missing] = float(i_cfg.get("fallback_ti_objective_db", 36.0))

        # ---- I_max (Watt)
    override_dbm = fs_cfg.get("i_max_dbm_override", None)
    if override_dbm is not None:
        i_max_watt = np.full((num_fs,), float(dbm_to_watt(float(override_dbm))), dtype=float)
    else:
        mode = str(i_cfg.get("mode", "rx_threshold_minus_ti_objective")).lower().strip()

        # Allow common aliases for the noise+I/N based mode
        noise_modes = {"noise_plus_in_target", "in_target_db", "noise_plus_in", "noise_in"}

        if mode == "rx_threshold_minus_ti_objective":
            # Use record-specific rx_threshold and TI objective if present
            i_dbw = fs_rxthr_dbw - ti_obj
            i_max_watt = _dbw_to_watt_vec(i_dbw)

            # If any field is missing, fall back to noise + I/N target (ITU-R F.758)
            invalid = ~np.isfinite(i_max_watt)
            if np.any(invalid):
                temp_k = float(cfg.raw["noise"]["temperature_K"])
                nf_db = float(fs_cfg.get("receiver_noise_figure_db", fs_cfg.get("noise_figure_db", 0.0)))
                noise_pow_dbm_cfg = i_cfg.get("noise_power_dbm", None)

                bw_hz = fs_bw_khz[invalid] * 1e3
                bad_bw = ~np.isfinite(bw_hz) | (bw_hz <= 0)
                if np.any(bad_bw):
                    bw_hz = np.asarray(bw_hz, dtype=float)
                    bw_hz[bad_bw] = float(fs_cfg.get("rx_bandwidth_hz", 1e6))

                if noise_pow_dbm_cfg is not None:
                    noise_dbm = np.full((int(invalid.sum()),), float(noise_pow_dbm_cfg))
                else:
                    noise_dbm = _compute_noise_dbm_vector(bw_hz, temp_k, nf_db)

                in_target_db = float(i_cfg.get("in_target_db", -10.0))
                i_max_watt[invalid] = _dbm_to_watt_vec(noise_dbm + in_target_db)

        elif mode in noise_modes:
            # Direct noise + I/N mode
            temp_k = float(cfg.raw["noise"]["temperature_K"])
            nf_db = float(fs_cfg.get("receiver_noise_figure_db", fs_cfg.get("noise_figure_db", 0.0)))
            noise_pow_dbm_cfg = i_cfg.get("noise_power_dbm", None)

            bw_hz = fs_bw_khz * 1e3
            bad_bw = ~np.isfinite(bw_hz) | (bw_hz <= 0)
            if np.any(bad_bw):
                bw_hz = np.asarray(bw_hz, dtype=float)
                bw_hz[bad_bw] = float(fs_cfg.get("rx_bandwidth_hz", 1e6))

            if noise_pow_dbm_cfg is not None:
                noise_dbm = np.full((num_fs,), float(noise_pow_dbm_cfg))
            else:
                noise_dbm = _compute_noise_dbm_vector(bw_hz, temp_k, nf_db)

            in_target_db = float(i_cfg.get("in_target_db", -10.0))
            i_max_watt = _dbm_to_watt_vec(noise_dbm + in_target_db)

        else:
            raise ValueError(f"Unsupported fixed_service.i_max.mode='{mode}'")


    # ---- Build tensors
    real_dtype = _real_dtype(cfg)

    fs_one = tf.constant(np.stack([x_sel, y_sel, fs_h_m], axis=-1), dtype=real_dtype)
    fs_loc = tf.tile(fs_one[tf.newaxis, ...], [batch_size, 1, 1])

    return FixedServiceLocations(
        fs_loc=fs_loc,
        fs_center_hz=tf.constant(fs_freq_mhz * 1e6, dtype=real_dtype),
        fs_bw_hz=tf.constant(fs_bw_khz * 1e3, dtype=real_dtype),
        fs_i_max_watt=tf.constant(i_max_watt, dtype=real_dtype),
        fs_gain_dbi=tf.constant(fs_gain_dbi, dtype=real_dtype),
        fs_beamwidth_deg=tf.constant(fs_bwdeg, dtype=real_dtype),
        fs_azimuth_deg=tf.constant(fs_az, dtype=real_dtype),
    )



def _generate_fs_from_parameter_fits(cfg: ResolvedConfig, topo: TopologyData, batch_size: int, num_fs: int) -> FixedServiceLocations:
    fs_cfg = cfg.raw["fixed_service"]
    params = fs_cfg.get("parameters", {})
    if not isinstance(params, dict):
        raise ValueError("fixed_service.parameters must be a mapping")

    placement = fs_cfg.get("placement", {})
    margin_m = float(placement.get("margin_m", 0.0))
    x_min, x_max, y_min, y_max = _get_sim_bbox_xy(topo, margin_m)

    x_sel = np.random.uniform(x_min, x_max, size=(num_fs,))
    y_sel = np.random.uniform(y_min, y_max, size=(num_fs,))

    def _maybe_sample(key: str, default: float) -> np.ndarray:
        if key in params and isinstance(params[key], dict) and "dist" in params[key]:
            return _sample_from_dist_cfg(params[key], num_fs)
        return np.full((num_fs,), float(default), dtype=float)

    fs_freq_mhz = _maybe_sample("frequency_mhz", cfg.raw["system_model"]["carrier_frequency_hz"] / 1e6)
    fs_bw_khz = _maybe_sample("occupied_bw_khz", fs_cfg.get("rx_bandwidth_hz", 1e6) / 1e3)
    fs_gain_dbi = _maybe_sample("antenna_gain_dbi", 0.0)
    fs_bwdeg = _maybe_sample("beamwidth_3db_deg", 30.0)
    fs_az = _wrap_deg(_maybe_sample("azimuth_deg", 0.0))
    fs_h = _maybe_sample("height_agl_m", 20.0)
    fs_rxthr_dbw = _maybe_sample("rx_threshold_dbw_ber1e3", np.nan)

    ti_obj = None
    if "ti_objective_db" in params:
        ti_obj = _sample_from_dist_cfg(params["ti_objective_db"], num_fs)
    if ti_obj is None:
        ti_obj = np.full((num_fs,), float(fs_cfg.get("i_max", {}).get("fallback_ti_objective_db", 36.0)))

    i_cfg = fs_cfg.get("i_max", {})
    override_dbm = fs_cfg.get("i_max_dbm_override", None)
    if override_dbm is not None:
        i_max_watt = np.full((num_fs,), float(dbm_to_watt(float(override_dbm))), dtype=float)
    else:
        mode = str(i_cfg.get("mode", "rx_threshold_minus_ti_objective")).lower().strip()
        noise_modes = {"noise_plus_in_target", "in_target_db", "noise_plus_in", "noise_in"}

        if mode == "rx_threshold_minus_ti_objective":
            i_dbw = fs_rxthr_dbw - ti_obj
            i_max_watt = _dbw_to_watt_vec(i_dbw)

            invalid = ~np.isfinite(i_max_watt)
            if np.any(invalid):
                temp_k = float(cfg.raw["noise"]["temperature_K"])
                nf_db = float(fs_cfg.get("receiver_noise_figure_db", fs_cfg.get("noise_figure_db", 0.0)))
                noise_pow_dbm_cfg = i_cfg.get("noise_power_dbm", None)

                bw_hz = fs_bw_khz[invalid] * 1e3
                bad_bw = ~np.isfinite(bw_hz) | (bw_hz <= 0)
                if np.any(bad_bw):
                    bw_hz = np.asarray(bw_hz, dtype=float)
                    bw_hz[bad_bw] = float(fs_cfg.get("rx_bandwidth_hz", 1e6))

                if noise_pow_dbm_cfg is not None:
                    noise_dbm = np.full((int(invalid.sum()),), float(noise_pow_dbm_cfg))
                else:
                    noise_dbm = _compute_noise_dbm_vector(bw_hz, temp_k, nf_db)

                in_target_db = float(i_cfg.get("in_target_db", -10.0))
                i_max_watt[invalid] = _dbm_to_watt_vec(noise_dbm + in_target_db)

        elif mode in noise_modes:
            temp_k = float(cfg.raw["noise"]["temperature_K"])
            nf_db = float(fs_cfg.get("receiver_noise_figure_db", fs_cfg.get("noise_figure_db", 0.0)))
            noise_pow_dbm_cfg = i_cfg.get("noise_power_dbm", None)

            bw_hz = fs_bw_khz * 1e3
            bad_bw = ~np.isfinite(bw_hz) | (bw_hz <= 0)
            if np.any(bad_bw):
                bw_hz = np.asarray(bw_hz, dtype=float)
                bw_hz[bad_bw] = float(fs_cfg.get("rx_bandwidth_hz", 1e6))

            if noise_pow_dbm_cfg is not None:
                noise_dbm = np.full((num_fs,), float(noise_pow_dbm_cfg))
            else:
                noise_dbm = _compute_noise_dbm_vector(bw_hz, temp_k, nf_db)

            in_target_db = float(i_cfg.get("in_target_db", -10.0))
            i_max_watt = _dbm_to_watt_vec(noise_dbm + in_target_db)

        else:
            raise ValueError(f"Unsupported fixed_service.i_max.mode='{mode}'")


    real_dtype = _real_dtype(cfg)
    fs_one = tf.constant(np.stack([x_sel, y_sel, fs_h], axis=-1), dtype=real_dtype)
    fs_loc = tf.tile(fs_one[tf.newaxis, ...], [batch_size, 1, 1])

    return FixedServiceLocations(
        fs_loc=fs_loc,
        fs_center_hz=tf.constant(fs_freq_mhz * 1e6, dtype=real_dtype),
        fs_bw_hz=tf.constant(fs_bw_khz * 1e3, dtype=real_dtype),
        fs_i_max_watt=tf.constant(i_max_watt, dtype=real_dtype),
        fs_gain_dbi=tf.constant(fs_gain_dbi, dtype=real_dtype),
        fs_beamwidth_deg=tf.constant(fs_bwdeg, dtype=real_dtype),
        fs_azimuth_deg=tf.constant(fs_az, dtype=real_dtype),
    )
