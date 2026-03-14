"""Channel model wrappers.

Responsibilities
---------------
1) UE channels (BS->UT):
   - 3GPP TR 38.901 UMi-StreetCanyon large-scale pathloss
   - optional shadow fading
   - i.i.d. Rayleigh small-scale fading (baseline in the paper)

2) Fixed Service (FS) statistics (BS->FS):
   - free-space large-scale model for average gains \bar{\beta}
   - spectral overlap weights \epsilon between NR RE groups and FS bands
   - I_max thresholds

This module must NOT:
- Implement the receiver algorithm
- Do file I/O
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import tensorflow as tf

from .config import ResolvedConfig, dbm_to_watt
from .topology import FixedServiceLocations, TopologyData


# -----------------
# Data containers
# -----------------


@dataclass(frozen=True)
class FsStats:
    """Statistics needed for the average FS interference constraint.

    correlation:
      - "identity":  Rbar = I_M  -> tr(Rbar Q)=tr(Q) (depends only on per-BS power)
      - "steering_rank1": Rbar = a a^H -> tr(Rbar Q)=a^H Q a (directional, enables nulling)
    """
    bar_beta: tf.Tensor         # [batch,B,L]
    epsilon: tf.Tensor          # [T,L]
    delta: tf.Tensor            # [B,T]
    i_max_watt: tf.Tensor       # [L]
    correlation: str = "identity"
    a_bs_fs: Optional[tf.Tensor] = None   # [batch,B,L,M] complex (only for steering_rank1)



# -----------------
# UE channel model
# -----------------


def umi_los_probability(d_2d_m: tf.Tensor) -> tf.Tensor:
    """3GPP TR 38.901 UMi-StreetCanyon LoS probability.

    d_2d_m: 2D distance in meters.
    Returns probability in [0,1].
    """
    d = tf.maximum(d_2d_m, tf.constant(1e-3, dtype=d_2d_m.dtype))
    term1 = tf.minimum(18.0 / d, 1.0)
    p_los = term1 * (1.0 - tf.exp(-d / 36.0)) + tf.exp(-d / 36.0)
    return tf.clip_by_value(p_los, 0.0, 1.0)


def _log10(x: tf.Tensor) -> tf.Tensor:
    return tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))


def _o2i_penetration_loss_db(
    cfg: ResolvedConfig,
    ut_is_indoor: tf.Tensor,
    real_dtype: tf.DType,
) -> tf.Tensor:
    """Compute O2I penetration loss (dB) per UE using 3GPP TR 38.901.

    This implements the 3GPP O2I "low-loss / high-loss building" model
    (TR 38.901) and returns a per-UE additional loss (in dB). The loss is
    applied *only* to indoor UEs (outdoor UEs get 0 dB).

    YAML compatibility
    ------------------
    Supports both:

        channel_model:
          o2i:
            inside_loss_coeff_db_per_m: 0.5
            shadow_fading_std_db:
              low_loss: 4.4
              high_loss: 6.5

    and older scalar keys:

        channel_model:
          o2i:
            inside_loss_coeff: 0.5
            sigma_p_low_db: 4.4
            sigma_p_high_db: 6.5
    """

    o2i_cfg = cfg.raw.get("channel_model", {}).get("o2i", {}) or {}

    ut_is_indoor = tf.cast(ut_is_indoor, tf.bool)
    if ut_is_indoor.shape.rank == 2 and int(ut_is_indoor.shape[-1]) == 1:
        ut_is_indoor = tf.squeeze(ut_is_indoor, axis=-1)

    batch = tf.shape(ut_is_indoor)[0]
    U = tf.shape(ut_is_indoor)[1]

    # Frequency in GHz (TR 38.901 formulas use GHz)
    f_ghz = tf.cast(float(cfg.raw["system_model"]["carrier_frequency_hz"]) / 1e9, real_dtype)

    # --- External wall losses (TR 38.901)
    l_glass = tf.cast(2.02, real_dtype) + tf.cast(0.2, real_dtype) * f_ghz
    l_conc = tf.cast(5.0, real_dtype) + tf.cast(4.0, real_dtype) * f_ghz
    l_irr = tf.cast(23.0, real_dtype) + tf.cast(3.0, real_dtype) * f_ghz

    pl_npi_db = float(o2i_cfg.get("pl_npi_db", 0.0))
    pl_npi = tf.cast(pl_npi_db, real_dtype)

    ten = tf.cast(10.0, real_dtype)

    term_low = tf.cast(0.3, real_dtype) * tf.pow(ten, -l_glass / 10.0) + tf.cast(0.7, real_dtype) * tf.pow(
        ten, -l_conc / 10.0
    )
    pl_tw_low_db = tf.cast(5.0, real_dtype) - tf.cast(10.0, real_dtype) * _log10(term_low) + pl_npi

    term_high = tf.cast(0.7, real_dtype) * tf.pow(ten, -l_irr / 10.0) + tf.cast(0.3, real_dtype) * tf.pow(
        ten, -l_conc / 10.0
    )
    pl_tw_high_db = tf.cast(5.0, real_dtype) - tf.cast(10.0, real_dtype) * _log10(term_high) + pl_npi

    # --- Building type sampling (low-loss vs high-loss)
    probs = o2i_cfg.get("building_type_probs", {}) or {}
    p_low = float(probs.get("low_loss", 0.8))
    p_high = float(probs.get("high_loss", 0.2))
    s = p_low + p_high
    if s <= 0:
        p_low = 1.0
        p_high = 0.0
    else:
        p_low /= s
        p_high /= s

    u = tf.random.uniform([batch, U], dtype=real_dtype)
    is_low = u < tf.cast(p_low, real_dtype)
    pl_tw_db = tf.where(is_low, pl_tw_low_db, pl_tw_high_db)

    # --- Indoor distance (m)
    dist_cfg = o2i_cfg.get("indoor_distance_m", {}) or {}
    dist = str(dist_cfg.get("dist", "uniform")).lower().strip()

    low_m = float(dist_cfg.get("low_m", dist_cfg.get("low", dist_cfg.get("min", 0.0))))
    high_m = float(dist_cfg.get("high_m", dist_cfg.get("high", dist_cfg.get("max", 25.0))))
    if high_m < low_m:
        low_m, high_m = high_m, low_m

    if dist in ("min_of_two_uniform", "min_of_2_uniform", "min2"):
        d1 = tf.random.uniform([batch, U], minval=low_m, maxval=high_m, dtype=real_dtype)
        d2 = tf.random.uniform([batch, U], minval=low_m, maxval=high_m, dtype=real_dtype)
        d_in = tf.minimum(d1, d2)
    else:
        d_in = tf.random.uniform([batch, U], minval=low_m, maxval=high_m, dtype=real_dtype)

    # --- Internal loss coefficient (dB per meter)
    inside_loss_coeff = o2i_cfg.get("inside_loss_coeff_db_per_m", None)
    if inside_loss_coeff is None:
        inside_loss_coeff = o2i_cfg.get("inside_loss_coeff", 0.5)
    inside_loss_coeff = float(inside_loss_coeff)
    pl_in_db = tf.cast(inside_loss_coeff, real_dtype) * d_in

    # --- Penetration loss shadowing (TR 38.901)
    sh_cfg = o2i_cfg.get("shadow_fading_std_db", {}) or {}
    sigma_low = sh_cfg.get("low_loss", sh_cfg.get("low", None))
    if sigma_low is None:
        sigma_low = o2i_cfg.get("sigma_p_low_db", 4.4)
    sigma_low = float(sigma_low)

    sigma_high = sh_cfg.get("high_loss", sh_cfg.get("high", None))
    if sigma_high is None:
        sigma_high = o2i_cfg.get("sigma_p_high_db", 6.5)
    sigma_high = float(sigma_high)

    sigma_p = tf.where(is_low, tf.cast(sigma_low, real_dtype), tf.cast(sigma_high, real_dtype))
    sf_pen_db = tf.random.normal([batch, U], mean=0.0, stddev=1.0, dtype=real_dtype) * sigma_p

    pl_o2i_db = pl_tw_db + pl_in_db + sf_pen_db

    # Only indoor users get O2I loss
    return pl_o2i_db * tf.cast(ut_is_indoor, real_dtype)




def umi_breakpoint_distance_m(
    f_c_hz: float,
    h_bs_m: tf.Tensor,
    h_ut_m: tf.Tensor,
    h_e_m: float,
) -> tf.Tensor:
    """Compute the UMi breakpoint distance d_BP (meters)."""
    c = 299792458.0  # speed of light
    f = tf.cast(f_c_hz, h_bs_m.dtype)
    return 4.0 * (h_bs_m - h_e_m) * (h_ut_m - h_e_m) * f / c


def umi_pathloss_db(
    d_2d_m: tf.Tensor,
    d_3d_m: tf.Tensor,
    f_c_hz: float,
    h_bs_m: tf.Tensor,
    h_ut_m: tf.Tensor,
    h_e_m: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute UMi LoS and NLoS pathloss (dB) per TR 38.901.

    Returns
    -------
    pl_los_db, pl_nlos_db : tf.Tensor
        Both have the same shape as d_2d_m.
    """
    dtype = d_2d_m.dtype
    d_2d = tf.maximum(d_2d_m, tf.constant(1e-3, dtype=dtype))
    d_3d = tf.maximum(d_3d_m, tf.constant(1e-3, dtype=dtype))

    f_ghz = tf.cast(f_c_hz / 1e9, dtype)

    d_bp = umi_breakpoint_distance_m(f_c_hz, h_bs_m, h_ut_m, h_e_m)
    log10 = lambda x: tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=dtype))

    # LoS
    pl_los_1 = 32.4 + 21.0 * log10(d_3d) + 20.0 * log10(f_ghz)
    pl_los_2 = (
        32.4
        + 40.0 * log10(d_3d)
        + 20.0 * log10(f_ghz)
        - 9.5 * log10(d_bp**2 + (h_bs_m - h_ut_m) ** 2)
    )
    pl_los = tf.where(d_2d <= d_bp, pl_los_1, pl_los_2)

    # NLoS
    pl_nlos_base = (
        22.4
        + 35.3 * log10(d_3d)
        + 21.3 * log10(f_ghz)
        - 0.3 * (h_ut_m - 1.5)
    )
    pl_nlos = tf.maximum(pl_los, pl_nlos_base)

    return pl_los, pl_nlos


def generate_ue_channels(cfg: ResolvedConfig, topo: TopologyData, batch_size: int) -> tf.Tensor:
    """Generate BS->UT channels H[b,u] with i.i.d. Rayleigh small-scale fading.

    Output shape
    ------------
    H: [batch, B, U, N_rx, M_tx]
    """
    precision = cfg.derived["tf_precision"]
    real_dtype = tf.float32 if precision == "single" else tf.float64
    complex_dtype = tf.complex64 if precision == "single" else tf.complex128

    num_bs = int(cfg.derived["num_bs"])
    num_ut = int(cfg.derived["num_ut"])
    m_tx = int(cfg.raw["channel_model"]["num_bs_ant"])
    n_rx = int(cfg.raw["channel_model"]["num_ut_ant"])

    f_c_hz = float(cfg.raw["system_model"]["carrier_frequency_hz"])
    h_e_m = float(cfg.raw["channel_model"].get("h_e_m", 1.0))

    # Distances with wrap-around BS locations (recommended)
    # bs_virtual_loc: [batch, B, U, 3]
    bs_v = tf.cast(topo.bs_virtual_loc, real_dtype)
    ut = tf.cast(topo.ut_loc, real_dtype)

        # Defensive normalization: depending on the installed Sionna version,
    # `bs_virtual_loc` may come back as:
    #   - [batch, B, 3]       (no per-UE virtual locations)
    #   - [batch, U, B, 3]    (B/U swapped)
    # This normalizes to the expected [batch, B, U, 3].
    if bs_v.shape.rank == 3:
        bs_v = tf.tile(bs_v[:, :, tf.newaxis, :], [1, 1, num_ut, 1])
    elif bs_v.shape.rank == 4:
        try:
            if tf.executing_eagerly():
                if int(tf.shape(bs_v)[1].numpy()) == num_ut and int(tf.shape(bs_v)[2].numpy()) == num_bs:
                    bs_v = tf.transpose(bs_v, [0, 2, 1, 3])
        except Exception:
            # If dynamic validation fails, keep as-is.
            pass
    else:
        raise ValueError(f"topo.bs_virtual_loc must have rank 3 or 4, got rank={bs_v.shape.rank}")


    # Compute 2D and 3D distances
    diff_xy = bs_v[:, :, :, :2] - ut[:, tf.newaxis, :, :2]  # [batch,B,U,2]
    d_2d = tf.sqrt(tf.reduce_sum(diff_xy**2, axis=-1) + 1e-12)
    diff_z = bs_v[:, :, :, 2] - ut[:, tf.newaxis, :, 2]
    d_3d = tf.sqrt(d_2d**2 + diff_z**2)

    h_bs = bs_v[:, :, :, 2]
    h_ut = ut[:, tf.newaxis, :, 2]

    # LoS probability + sampling
    p_los = umi_los_probability(d_2d)

    try:
        from sionna.phy.utils import sample_bernoulli  # type: ignore

        los = sample_bernoulli(p_los)
    except Exception:
        # Fallback: TF uniform
        u = tf.random.uniform(tf.shape(p_los), dtype=real_dtype)
        los = u < p_los

    pl_los_db, pl_nlos_db = umi_pathloss_db(d_2d, d_3d, f_c_hz, h_bs, h_ut, h_e_m)
    pl_db = tf.where(los, pl_los_db, pl_nlos_db)

    # Optional shadow fading (disabled in baseline)
    if bool(cfg.raw["channel_model"].get("shadow_fading_enable", False)):
        sigma_los = float(cfg.raw["channel_model"].get("shadow_fading_std_los_db", 4.0))
        sigma_nlos = float(cfg.raw["channel_model"].get("shadow_fading_std_nlos_db", 7.82))
        sigma = tf.where(los, tf.cast(sigma_los, real_dtype), tf.cast(sigma_nlos, real_dtype))
        sf = tf.random.normal(tf.shape(pl_db), mean=0.0, stddev=sigma, dtype=real_dtype)
        pl_db = pl_db + sf

        # O2I penetration loss (adds to pathloss for indoor UEs only)
    o2i_cfg = cfg.raw.get("channel_model", {}).get("o2i", {})
    if bool(o2i_cfg.get("enabled", False)):
        ut_is_indoor = topo.ut_is_indoor
        if ut_is_indoor is None:
            p_ind = float(cfg.raw.get("topology", {}).get("indoor_probability", 0.0))
            ut_is_indoor = tf.random.uniform([batch_size, num_ut], dtype=real_dtype) < tf.cast(p_ind, real_dtype)
        if ut_is_indoor.shape.rank == 3 and ut_is_indoor.shape[-1] == 1:
            ut_is_indoor = tf.squeeze(ut_is_indoor, axis=-1)
        ut_is_indoor = tf.cast(ut_is_indoor, tf.bool)

        pl_o2i_db = _o2i_penetration_loss_db(cfg, ut_is_indoor, real_dtype)  # [batch,U]
        pl_db = pl_db + pl_o2i_db[:, tf.newaxis, :]


    beta = tf.pow(tf.constant(10.0, dtype=real_dtype), -pl_db / 10.0)  # linear power gain

    # Small-scale fading: i.i.d Rayleigh
    try:
        from sionna.phy.utils import complex_normal  # type: ignore

        h_tilde = complex_normal([batch_size, num_bs, num_ut, n_rx, m_tx], dtype=complex_dtype)
    except Exception:
        # Fallback CN(0,1)
        re = tf.random.normal([batch_size, num_bs, num_ut, n_rx, m_tx], dtype=real_dtype)
        im = tf.random.normal([batch_size, num_bs, num_ut, n_rx, m_tx], dtype=real_dtype)
        h_tilde = tf.complex(re, im) / tf.cast(tf.sqrt(2.0), complex_dtype)

    h = tf.cast(tf.sqrt(beta), complex_dtype)[..., tf.newaxis, tf.newaxis] * h_tilde
    return h


# -----------------
# FS stats
# -----------------


def _compute_epsilon_overlap(cfg: ResolvedConfig, fs_center_hz: tf.Tensor, fs_bw_hz) -> tf.Tensor:
    """Compute spectral overlap epsilon_{t,l} between tone-groups and FS bands.

    Args:
        fs_center_hz: [L] or [1,L]
        fs_bw_hz:     scalar or [L] or [1,L]
    Returns:
        epsilon: [T, L] with entries in [0,1]
    """
    real_dtype = tf.float64 if str(cfg.derived["tf_precision"]).lower().startswith("double") else tf.float32

    bw_total = float(cfg.derived["bandwidth_total_hz"])
    f_c = float(cfg.raw["system_model"]["carrier_frequency_hz"])
    gnb_min = f_c - bw_total / 2.0
    gnb_max = f_c + bw_total / 2.0

    # Tone groups
    T = int(cfg.derived["num_re_sim"])
    tone_bw = bw_total / T
    tone_edges = tf.linspace(tf.cast(gnb_min, real_dtype), tf.cast(gnb_max, real_dtype), T + 1)  # [T+1]
    tone_lo = tone_edges[:-1]  # [T]
    tone_hi = tone_edges[1:]   # [T]

    # FS band edges
    fc = tf.cast(fs_center_hz, real_dtype)
    if fc.shape.rank == 1:
        fc = fc[tf.newaxis, :]  # [1,L]

    if isinstance(fs_bw_hz, tf.Tensor):
        bw = tf.cast(fs_bw_hz, real_dtype)
    else:
        bw = tf.cast(float(fs_bw_hz), real_dtype)

    if not isinstance(bw, tf.Tensor):
        bw = tf.fill([tf.shape(fc)[1]], bw)

    if bw.shape.rank == 1:
        bw = bw[tf.newaxis, :]  # [1,L]

    fs_min = fc - bw / 2.0  # [1,L]
    fs_max = fc + bw / 2.0  # [1,L]

    # Broadcast to [T,L]
    lo = tone_lo[:, tf.newaxis]
    hi = tone_hi[:, tf.newaxis]
    overlap = tf.nn.relu(tf.minimum(hi, fs_max) - tf.maximum(lo, fs_min))
    eps = overlap / tf.cast(tone_bw, real_dtype)
    return tf.clip_by_value(eps, 0.0, 1.0)



def _compute_bs_fs_steering_vectors(
    cfg: ResolvedConfig,
    bs_xyz: tf.Tensor,   # [batch,B,3] real
    fs_xyz: tf.Tensor,   # [batch,L,3] real
) -> tf.Tensor:
    """Compute geometry-based BS steering vectors a_{b->ell} with ||a||^2 = M.

    Assumption (documented):
      - URA panel in the local y-z plane (broadside along +x of a sector frame)
      - element spacing = (element_spacing_lambda * wavelength)
      - optional sector azimuth rotation (default enabled)

    NOTE:
      This implementation infers B from `bs_xyz` (not cfg.derived["num_bs"]) so that
      unit tests / custom topologies do not crash with shape mismatches.
    """
    precision = cfg.derived["tf_precision"]
    real_dtype = tf.float32 if precision == "single" else tf.float64
    complex_dtype = tf.complex64 if precision == "single" else tf.complex128

    M = int(cfg.raw["channel_model"]["num_bs_ant"])

    ant_cfg = cfg.raw["fixed_service"].get("antenna", {})
    steer_cfg = ant_cfg.get("steering", {})

    rows = int(steer_cfg.get("array_rows", steer_cfg.get("rows", 8)))
    cols = int(steer_cfg.get("array_cols", steer_cfg.get("cols", 8)))
    pol  = int(steer_cfg.get("polarizations", 2))
    elem_spacing_lambda = float(steer_cfg.get("element_spacing_lambda", 0.5))

    # Fallback if M does not match requested URA geometry
    if rows * cols * pol != M:
        rows, cols, pol = M, 1, 1

    # Direction from BS to FS (unit vector)
    diff = fs_xyz[:, tf.newaxis, :, :] - bs_xyz[:, :, tf.newaxis, :]   # [batch,B,L,3]
    d = tf.sqrt(tf.reduce_sum(diff**2, axis=-1, keepdims=True) + 1e-12)
    u = diff / d  # [batch,B,L,3]

    # Optional sector azimuth rotation (so each sector has a different x-axis / boresight)
    include_sector_az = bool(steer_cfg.get("include_sector_azimuth", True))
    if include_sector_az:
        # Default: [30,150,270] repeating
        az_deg = steer_cfg.get("sector_azimuths_deg", [30.0, 150.0, 270.0])
        if not (isinstance(az_deg, (list, tuple)) and len(az_deg) == 3):
            az_deg = [30.0, 150.0, 270.0]

        pi = tf.cast(3.141592653589793, real_dtype)
        phi3 = tf.cast(tf.constant(az_deg), real_dtype) * (pi / tf.cast(180.0, real_dtype))  # [3]

        # Infer B from the actual tensor (robust to unit tests)
        B_static = bs_xyz.shape[1]
        if B_static is None:
            B_dyn = tf.shape(bs_xyz)[1]
            idx = tf.range(B_dyn, dtype=tf.int32)
            phi_b = tf.gather(phi3, tf.math.mod(idx, 3))  # [B_dyn]
            cos_phi = tf.reshape(tf.cos(phi_b), tf.stack([1, B_dyn, 1, 1]))
            sin_phi = tf.reshape(tf.sin(phi_b), tf.stack([1, B_dyn, 1, 1]))
        else:
            B_int = int(B_static)
            phi_b = tf.tile(phi3, [max(B_int // 3, 1)])[:B_int]  # [B_int]
            cos_phi = tf.reshape(tf.cos(phi_b), [1, B_int, 1, 1])
            sin_phi = tf.reshape(tf.sin(phi_b), [1, B_int, 1, 1])

        u_x = u[..., 0:1]
        u_y = u[..., 1:2]
        u_z = u[..., 2:3]

        # rotate XY components into sector frame
        u_y_loc = -u_x * sin_phi + u_y * cos_phi
        u_z_loc = u_z
    else:
        u_y_loc = u[..., 1:2]
        u_z_loc = u[..., 2:3]

    # Wavelength
    c = 299792458.0
    fc = float(cfg.raw["system_model"]["carrier_frequency_hz"])
    lam = c / fc
    d_elem = elem_spacing_lambda * lam
    k = tf.cast(2.0 * 3.141592653589793 / lam, real_dtype)

    # URA element coordinates (y,z)
    y = (tf.cast(tf.range(cols), real_dtype) - tf.cast((cols - 1) / 2.0, real_dtype)) * tf.cast(d_elem, real_dtype)  # [cols]
    z = (tf.cast(tf.range(rows), real_dtype) - tf.cast((rows - 1) / 2.0, real_dtype)) * tf.cast(d_elem, real_dtype)  # [rows]
    Z, Y = tf.meshgrid(z, y, indexing="ij")  # [rows,cols]

    Y = Y[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # [1,1,1,rows,cols]
    Z = Z[tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    u_y_loc5 = u_y_loc[..., tf.newaxis, tf.newaxis]  # [batch,B,L,1,1]
    u_z_loc5 = u_z_loc[..., tf.newaxis, tf.newaxis]  # [batch,B,L,1,1]
    phase = k * (u_y_loc5 * Y + u_z_loc5 * Z)  # [batch,B,L,rows,cols]

    # exp(-j*phase)
    a2 = tf.complex(tf.cos(phase), -tf.sin(phase))  # [batch,B,L,rows,cols]
    a_flat = tf.reshape(
        a2,
        [tf.shape(bs_xyz)[0], tf.shape(bs_xyz)[1], tf.shape(fs_xyz)[1], rows * cols],
    )  # [batch,B,L,rows*cols]

    # replicate for polarizations (co-located pols => same steering)
    if pol > 1:
        a_flat = tf.tile(a_flat, [1, 1, 1, pol])  # [batch,B,L,M]

    return tf.cast(a_flat, complex_dtype)




def _fs_rx_pattern_atten_db(
    cfg: ResolvedConfig,
    bs_loc: tf.Tensor,  # [batch,B,3]
    fs_loc: tf.Tensor,  # [batch,L,3]
    fs_beamwidth_deg: Optional[tf.Tensor],  # [L]
    fs_azimuth_deg: Optional[tf.Tensor],    # [L]
    real_dtype: tf.DType,
) -> tf.Tensor:
    """FS Rx antenna attenuation (dB) for each (batch,BS,FS) link."""
    rx_pat = cfg.raw.get("fixed_service", {}).get("antenna", {}).get("rx_pattern", {}) or {}
    if not bool(rx_pat.get("enabled", False)):
        return tf.zeros([tf.shape(bs_loc)[0], tf.shape(bs_loc)[1], tf.shape(fs_loc)[1]], dtype=real_dtype)

    pattern = str(rx_pat.get("pattern", "3gpp_tr38901")).lower().strip()
    if pattern not in ("3gpp_tr38901", "tr38901", "3gpp"):
        return tf.zeros([tf.shape(bs_loc)[0], tf.shape(bs_loc)[1], tf.shape(fs_loc)[1]], dtype=real_dtype)

    if fs_beamwidth_deg is None or fs_azimuth_deg is None:
        return tf.zeros([tf.shape(bs_loc)[0], tf.shape(bs_loc)[1], tf.shape(fs_loc)[1]], dtype=real_dtype)

    a_max_db = float(rx_pat.get("a_max_db", 30.0))
    a_max = tf.cast(a_max_db, real_dtype)

    bw = tf.reshape(tf.cast(fs_beamwidth_deg, real_dtype), [1, 1, -1])  # [1,1,L]
    az0 = tf.reshape(tf.cast(fs_azimuth_deg, real_dtype), [1, 1, -1])   # [1,1,L]

    bs_xy = bs_loc[:, :, tf.newaxis, :2]  # [batch,B,1,2]
    fs_xy = fs_loc[:, tf.newaxis, :, :2]  # [batch,1,L,2]
    diff = bs_xy - fs_xy                   # [batch,B,L,2] direction FS->BS

    bearing = tf.math.atan2(diff[..., 1], diff[..., 0]) * (180.0 / np.pi)
    bearing = tf.math.floormod(bearing, 360.0)  # [0,360)

    delta = tf.math.floormod(bearing - az0 + 180.0, 360.0) - 180.0
    delta = tf.abs(delta)

    omni = bw >= tf.cast(180.0, real_dtype)
    a = tf.cast(12.0, real_dtype) * tf.square(delta / tf.maximum(bw, tf.cast(1e-3, real_dtype)))
    atten = tf.minimum(a, a_max)
    atten = tf.where(omni, tf.zeros_like(atten), atten)
    return atten


def generate_fs_stats(cfg: ResolvedConfig, topo: TopologyData, fs: FixedServiceLocations, batch_size: int) -> FsStats:
    """Generate FS statistics (epsilon, delta, bar_beta, i_max) for WMMSE.

    Uses per-FS metadata if present (ISED mode), else falls back to legacy.
    """
    fs_cfg = cfg.raw.get("fixed_service", {}) or {}
    real_dtype = tf.float64 if str(cfg.derived["tf_precision"]).lower().startswith("double") else tf.float32
    complex_dtype = tf.complex128 if real_dtype == tf.float64 else tf.complex64

    B = int(topo.bs_loc.shape[1])
    L = int(fs.fs_loc.shape[1])
    T = int(cfg.derived["num_re_sim"])

    # FS center / BW (prefer metadata)
    if fs.fs_center_hz is not None:
        fs_center_hz = tf.cast(fs.fs_center_hz, real_dtype)  # [L]
    else:
        fs_center_hz = tf.fill([L], tf.cast(float(cfg.raw["system_model"]["carrier_frequency_hz"]), real_dtype))

    if fs.fs_bw_hz is not None:
        fs_bw_hz = tf.cast(fs.fs_bw_hz, real_dtype)  # [L]
    else:
        fs_bw_hz = tf.fill([L], tf.cast(float(fs_cfg.get("rx_bandwidth_hz", 1e6)), real_dtype))

    # Frequency in GHz for ABG / FSPL
    fc_ghz = fs_center_hz / tf.cast(1e9, real_dtype)            # [L]
    fc_ghz_3d = tf.reshape(fc_ghz, [1, 1, L])                   # [1,1,L] for broadcast

    epsilon = _compute_epsilon_overlap(cfg, fs_center_hz=fs_center_hz, fs_bw_hz=fs_bw_hz)  # [T,L]
    delta = tf.ones([B, T], dtype=real_dtype)

    # I_max (prefer per-FS metadata)
    if fs.fs_i_max_watt is not None:
        i_max = tf.cast(fs.fs_i_max_watt, real_dtype)  # [L]
    else:
        i_max = tf.fill([L], tf.cast(float(cfg.derived["fs_i_max_watt"]), real_dtype))

    # Correlation / steering
    ant_cfg = fs_cfg.get("antenna", {}) or {}
    correlation = str(ant_cfg.get("correlation", "identity")).lower().strip()

    a_bs_fs = None
    if correlation == "steering_rank1":
        # NOTE: _compute_bs_fs_steering_vectors expects (cfg, bs_xyz, fs_xyz).
        a_bs_fs = tf.cast(
            _compute_bs_fs_steering_vectors(cfg, bs_xyz=topo.bs_loc, fs_xyz=fs.fs_loc),
            complex_dtype,
        )

    # Distances [batch,B,L]
    bs = tf.cast(topo.bs_loc, real_dtype)
    fs_loc = tf.cast(fs.fs_loc, real_dtype)
    diff = fs_loc[:, tf.newaxis, :, :] - bs[:, :, tf.newaxis, :]
    d3d = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1))
    d2d = tf.sqrt(tf.reduce_sum(tf.square(diff[..., :2]), axis=-1))

    path_cfg = fs_cfg.get("pathloss", {}) or {}
    model = str(path_cfg.get("model", "free_space")).lower().strip()

    d_ref = float(path_cfg.get("d_ref_m", 1.0))
    d_ref_t = tf.cast(d_ref, real_dtype)
    d3d = tf.maximum(d3d, d_ref_t)
    log_term = _log10(d3d / d_ref_t)

    # Reference loss at d_ref
    fspl_const_db = path_cfg.get("fspl_const_db", None)
    if fspl_const_db is None:
        const_db = float(32.4 + 20.0 * np.log10(d_ref))
        l0_db_fs = tf.cast(const_db, real_dtype) + tf.cast(20.0, real_dtype) * _log10(fc_ghz)  # [L]
        l0_db = tf.reshape(l0_db_fs, [1, 1, L])  # broadcast to [batch,B,L]
    else:
        l0_db = tf.cast(float(fspl_const_db) + 20.0 * np.log10(d_ref), real_dtype)

    # Antenna coupling constants
    kappa_ant = float(path_cfg.get("kappa_ant", ant_cfg.get("kappa_ant", 1.0)))
    l_ant_db = float(path_cfg.get("l_ant_db", ant_cfg.get("l_ant_db", 0.0)))
    ant_lin = tf.cast(kappa_ant * (10.0 ** (-l_ant_db / 10.0)), real_dtype)

    # FS Rx gain + pattern attenuation
    atten_db = _fs_rx_pattern_atten_db(
        cfg,
        bs_loc=bs,
        fs_loc=fs_loc,
        fs_beamwidth_deg=getattr(fs, "fs_beamwidth_deg", None),
        fs_azimuth_deg=getattr(fs, "fs_azimuth_deg", None),
        real_dtype=real_dtype,
    )

    if getattr(fs, "fs_gain_dbi", None) is not None:
        gain_db = tf.reshape(tf.cast(fs.fs_gain_dbi, real_dtype), [1, 1, L])
    else:
        gain_db = tf.zeros([1, 1, L], dtype=real_dtype)

    rx_gain_lin = tf.pow(tf.cast(10.0, real_dtype), (gain_db - atten_db) / 10.0)

    # Shadow-fading expectation factors (lognormal mean)
    ln10 = float(np.log(10.0))
    sigma_los = float(path_cfg.get("shadow_fading_std_los_db", 0.0))
    sigma_nlos = float(path_cfg.get("shadow_fading_std_nlos_db", 0.0))
    kappa_los = float(np.exp((ln10 * ln10) * (sigma_los * sigma_los) / 200.0))
    kappa_nlos = float(np.exp((ln10 * ln10) * (sigma_nlos * sigma_nlos) / 200.0))

    # LoS probability (used by models that mix LoS/NLoS)
    los_model = str(path_cfg.get("los_probability_model", "always_los")).lower().strip()
    if los_model in ("always_los", "true", "1"):
        p_los = tf.ones_like(d2d, dtype=real_dtype)
    elif los_model in ("never_los", "false", "0"):
        p_los = tf.zeros_like(d2d, dtype=real_dtype)
    else:
        p_los = umi_los_probability(d2d)

    if model in ("free_space", "log_distance"):
        alpha = float(path_cfg.get("alpha", 2.0))
        pl_db = l0_db + tf.cast(10.0 * alpha, real_dtype) * log_term
        beta = tf.pow(tf.cast(10.0, real_dtype), -pl_db / 10.0)
        bar_beta = beta * ant_lin * rx_gain_lin

    elif model == "umi_los_nlos_avg":
        sub = path_cfg.get("umi_los_nlos_avg", {}) or {}
        alpha_los = float(sub.get("alpha_los", 2.0))
        alpha_nlos = float(sub.get("alpha_nlos", 3.0))

        pl_los_db = l0_db + tf.cast(10.0 * alpha_los, real_dtype) * log_term
        pl_nlos_db = l0_db + tf.cast(10.0 * alpha_nlos, real_dtype) * log_term

        beta_los = tf.pow(tf.cast(10.0, real_dtype), -pl_los_db / 10.0) * tf.cast(kappa_los, real_dtype)
        beta_nlos = tf.pow(tf.cast(10.0, real_dtype), -pl_nlos_db / 10.0) * tf.cast(kappa_nlos, real_dtype)

        beta = p_los * beta_los + (tf.cast(1.0, real_dtype) - p_los) * beta_nlos
        bar_beta = beta * ant_lin * rx_gain_lin

    elif model == "itu_p1411_above_rooftop_avg":
        p1411 = path_cfg.get("itu_p1411_abg", {}) or {}

        alpha_los = float(p1411.get("alpha_los", 2.29))
        beta_los_const = float(p1411.get("beta_los_const", p1411.get("beta_los", 28.6)))
        gamma_los = float(p1411.get("gamma_los", 1.96))

        alpha_nlos = float(p1411.get("alpha_nlos", 4.39))
        beta_nlos_const = float(p1411.get("beta_nlos_const", p1411.get("beta_nlos", -6.27)))
        gamma_nlos = float(p1411.get("gamma_nlos", 2.30))

        clamp_to_fspl = bool(p1411.get("clamp_to_fspl", True))
        d_min_los = float(p1411.get("d_min_los_m", 55.0))
        d_min_nlos = float(p1411.get("d_min_nlos_m", 260.0))

        # Validity floors for ABG model
        d_los = tf.maximum(d3d, tf.cast(d_min_los, real_dtype))
        d_nlos = tf.maximum(d3d, tf.cast(d_min_nlos, real_dtype))

        # ABG formula: PL = 10*alpha*log10(d[m]) + beta + 10*gamma*log10(f[GHz])
        a_los = tf.cast(10.0 * alpha_los, real_dtype)
        b_los = tf.cast(beta_los_const, real_dtype)
        g_los = tf.cast(10.0 * gamma_los, real_dtype)

        a_nlos = tf.cast(10.0 * alpha_nlos, real_dtype)
        b_nlos = tf.cast(beta_nlos_const, real_dtype)
        g_nlos = tf.cast(10.0 * gamma_nlos, real_dtype)

        pl_los_db = a_los * _log10(d_los) + b_los + g_los * _log10(fc_ghz_3d)
        pl_nlos_db = a_nlos * _log10(d_nlos) + b_nlos + g_nlos * _log10(fc_ghz_3d)

        # Do not allow ABG to be better than free-space (physical clamp)
        if clamp_to_fspl:
            fspl_db = tf.cast(32.4, real_dtype) \
                + tf.cast(20.0, real_dtype) * _log10(d3d) \
                + tf.cast(20.0, real_dtype) * _log10(fc_ghz_3d)
            pl_los_db = tf.maximum(pl_los_db, fspl_db)
            pl_nlos_db = tf.maximum(pl_nlos_db, fspl_db)

        # Convert to linear, include lognormal mean factors (kappa_*)
        kappa_los_t = tf.cast(kappa_los, real_dtype)
        kappa_nlos_t = tf.cast(kappa_nlos, real_dtype)

        beta_los = tf.pow(tf.cast(10.0, real_dtype), -pl_los_db / tf.cast(10.0, real_dtype)) * kappa_los_t
        beta_nlos = tf.pow(tf.cast(10.0, real_dtype), -pl_nlos_db / tf.cast(10.0, real_dtype)) * kappa_nlos_t

        beta = p_los * beta_los + (tf.cast(1.0, real_dtype) - p_los) * beta_nlos

        # Final "average" large-scale gain used by the FS constraint:
        # includes antenna + polarization terms
        bar_beta = beta * ant_lin * rx_gain_lin

    else:
        raise ValueError(f"Unsupported fixed_service.pathloss.model='{model}'")

    return FsStats(
        bar_beta=bar_beta,
        epsilon=epsilon,
        delta=delta,
        i_max_watt=i_max,
        correlation=correlation,
        a_bs_fs=a_bs_fs,
    )



