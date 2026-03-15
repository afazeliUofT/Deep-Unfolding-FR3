"""Receiver / beamformer design algorithms.

Scope in this repo
------------------
Implements the *classical* WMMSE baseline with dual updates for
- per-BS total power constraints
- average FS interference constraints

Deep unfolding (learned step sizes / mixing coefficients / unrolled layers) is
explicitly out of scope, but the architecture leaves a clean extension point via
`ReceiverBase`.

This module must NOT:
- Generate topology or channels
- Do file I/O
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import tensorflow as tf

from .config import ResolvedConfig, dbm_to_watt
from .channel import FsStats
from .processing import MmseOutput, mmse_combiners_and_mse


@dataclass(frozen=True)
class ReceiverResult:
    """Outputs of a receiver/beamformer design run."""

    w: tf.Tensor  # [batch, T, B, M, U_per_bs]
    mmse: MmseOutput  # computed on the final iterate (flattened tones merged into batch)
    mu: tf.Tensor  # [batch, B]
    lam: tf.Tensor  # [batch, L] (empty if FS disabled)
    num_iter: int
    converged: bool
    history: Dict[str, tf.Tensor]


class ReceiverBase(ABC):
    """Interface for pluggable receivers.

    A future deep-unfolded receiver should implement the same interface.
    """

    @abstractmethod
    def solve(
        self,
        cfg: ResolvedConfig,
        H: tf.Tensor,
        noise_var_watt: float,
        fs: Optional[FsStats],
        bs_total_tx_power_watt: Optional[float] = None,
    ) -> ReceiverResult:
        raise NotImplementedError


def _complex_normal(shape, dtype: tf.dtypes.DType) -> tf.Tensor:
    """Best-effort complex normal CN(0,1)."""
    try:
        from sionna.phy.utils import complex_normal  # type: ignore

        return complex_normal(shape, dtype=dtype)
    except Exception:
        real_dtype = tf.float32 if dtype == tf.complex64 else tf.float64
        re = tf.random.normal(shape, dtype=real_dtype)
        im = tf.random.normal(shape, dtype=real_dtype)
        return tf.complex(re, im) / tf.cast(tf.sqrt(2.0), dtype)


def _extract_self_channels(H: tf.Tensor, u_per_bs: int) -> tf.Tensor:
    """Extract channels from each BS to its *own* served users.

    Parameters
    ----------
    H : [S, B_tx, U, Nr, M]
    u_per_bs : number of users per BS (single stream each)

    Returns
    -------
    H_self : [S, B, u_per_bs, Nr, M]
        H_self[s,b,u] = H[s, b_tx=b, u_global=b*u_per_bs+u]
    """
    S = tf.shape(H)[0]
    B = tf.shape(H)[1]
    Nr = tf.shape(H)[3]
    M = tf.shape(H)[4]

    # Reshape user dim: U = B * u_per_bs
    H_rs = tf.reshape(H, [S, B, B, u_per_bs, Nr, M])
    # Bring (B_tx,B_serving) to last for diag extraction
    H_perm = tf.transpose(H_rs, [0, 3, 4, 5, 1, 2])  # [S, u_per_bs, Nr, M, B, B]
    H_diag = tf.linalg.diag_part(H_perm)  # [S, u_per_bs, Nr, M, B]
    H_self = tf.transpose(H_diag, [0, 4, 1, 2, 3])  # [S, B, u_per_bs, Nr, M]
    return H_self

def _max_fs_interference_watt_rank1(
    w_full: tf.Tensor,
    fs: FsStats,
    re_scaling: float,
    real_dtype: tf.DType,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute FS interference for steering-rank1 correlation.

    Parameters
    ----------
    w_full : [batch, T, B, M, u_per_bs] complex beamformer
    fs     : FsStats with steering vectors and coupling terms
    re_scaling : float scaling from simulated REs to full band
    real_dtype : tf.float32 or tf.float64

    Returns
    -------
    I_fs_max : scalar tensor (worst-case over batch and receivers)
    I_fs     : [batch, L] tensor
    """
    a = tf.cast(fs.a_bs_fs, w_full.dtype)  # [batch,B,L,M]
    T = tf.shape(w_full)[1]
    a_eff = tf.tile(a[:, tf.newaxis, ...], [1, T, 1, 1, 1])  # [batch,T,B,L,M]

    proj = tf.einsum(
        "stblm,stbmu->stblu",
        tf.math.conj(a_eff),
        w_full,
        optimize=True,
    )  # [batch,T,B,L,u_per_bs]

    p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)  # [batch,T,B,L]

    delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])  # [T,B]
    p_dir = p_dir * delta_tb[tf.newaxis, :, :, tf.newaxis]
    epsilon = tf.cast(fs.epsilon, real_dtype)  # [T,L]
    bar_beta = tf.cast(fs.bar_beta, real_dtype)  # [batch,B,L]

    I_fs = tf.cast(re_scaling, real_dtype) * tf.einsum(
        "stbl,tl,sbl->sl",
        p_dir,
        epsilon,
        bar_beta,
        optimize=True,
    )  # [batch,L]

    return tf.reduce_max(I_fs), I_fs



class WmmseReceiver(ReceiverBase):
    """Classical WMMSE beamformer design with dual updates."""

    def solve(
        self,
        cfg: ResolvedConfig,
        H: tf.Tensor,
        noise_var_watt: float,
        fs: Optional[FsStats] = None,
        bs_total_tx_power_watt: Optional[float] = None,
        # Internal knobs (used by fs_lambda_search / unit tests)
        _lam_scalar_override: Optional[float] = None,
        _freeze_lambda: bool = False,
        _force_aggressive_fs_nulling: Optional[bool] = None,
        _w_init_override: Optional[tf.Tensor] = None,
        _skip_fs_lambda_search: bool = False,
    ) -> ReceiverResult:

        precision = cfg.derived["tf_precision"]
        real_dtype = tf.float32 if precision == "single" else tf.float64
        complex_dtype = tf.complex64 if precision == "single" else tf.complex128

        # Dimensions
        B = int(cfg.derived["num_bs"])
        u_per_bs = int(cfg.raw["topology"]["num_ut_per_sector"])
        U = int(cfg.derived["num_ut"])
        M = int(cfg.raw["channel_model"]["num_bs_ant"])
        Nr = int(cfg.raw["channel_model"]["num_ut_ant"])
        T = int(cfg.derived["num_re_sim"])

        batch = tf.shape(H)[0]

        # If H doesn't have tone dimension, replicate across tones
        if len(H.shape) == 5:
            H_t = tf.tile(H[:, tf.newaxis, ...], [1, T, 1, 1, 1, 1])  # [batch,T,B,U,Nr,M]
        elif len(H.shape) == 6:
            H_t = H
        else:
            raise ValueError("H must have shape [batch,B,U,Nr,M] or [batch,T,B,U,Nr,M]")

        # Flatten tones into batch dimension for MMSE computation
        H_eff = tf.reshape(H_t, [-1, B, U, Nr, M])  # [batch*T, B, U, Nr, M]
        S_eff = tf.shape(H_eff)[0]

        # --- Config parameters
        w_cfg = cfg.raw["receiver"]["wmmse"]


        # ---- FS enforcement mode (optional, backward-compatible)
        # Priority: receiver.wmmse.fs_enforcement (new) > receiver.wmmse.aggressive_fs_nulling (legacy)
        fs_mode = str(w_cfg.get("fs_enforcement", "")).lower().strip()
        if fs_mode == "":
            fs_mode = "hard_null" if bool(w_cfg.get("aggressive_fs_nulling", False)) else "none"

        # Make a local copy so we can safely override behavior without mutating cfg.raw
        w_cfg = dict(w_cfg)

        if fs_mode == "hard_null":
            w_cfg["aggressive_fs_nulling"] = True
        elif fs_mode in ("budget_dual", "hybrid", "none"):
            w_cfg["aggressive_fs_nulling"] = False

        if _force_aggressive_fs_nulling is not None:
            w_cfg["aggressive_fs_nulling"] = bool(_force_aggressive_fs_nulling)

        # ---- Optional: automatic lambda search ("use" the FS interference budget) for steering-rank1 FS model
        if (
            (not _skip_fs_lambda_search)
            and (fs is not None)
            and (fs_mode in ("budget_dual", "hybrid"))
            and bool(w_cfg.get("fs_lambda_search", False))
        ):
            return self._solve_budget_dual_with_lambda_search(
                cfg=cfg,
                H=H,
                noise_var_watt=noise_var_watt,
                fs=fs,
                bs_total_tx_power_watt=bs_total_tx_power_watt,
                fs_mode=fs_mode,
            )






                # ---- Hyperparameters
        # FS dual penalty is enabled only for budget_dual / hybrid modes.
        # In "none" or "hard_null" modes, we still allow computing FS *metrics* (fs is not None),
        # but we force lambda=0 and disable lambda updates so there is no soft FS constraint.
        fs_dual_enabled = (fs is not None) and (fs_mode in ("budget_dual", "hybrid"))

        K_iter = int(w_cfg.get("num_iterations", 20))
        tol = float(w_cfg.get("convergence_tol", 1e-6))
        ridge = float(w_cfg.get("ridge_regularization", 1e-9))
        rho_mu = float(w_cfg.get("dual_step_mu", 0.02))
        rho_lam = 0.0 if (not fs_dual_enabled) or _freeze_lambda else float(w_cfg.get("dual_step_lambda", 0.2))


        damping = float(w_cfg.get("damping_w", 1.0))
        verbose = bool(w_cfg.get("verbose", False))

        # Total power constraint (per BS, across full band)
        # NOTE: cfg.derived uses the key "bs_total_tx_power_watt".
        p_tot_watt = float(bs_total_tx_power_watt) if bs_total_tx_power_watt is not None else float(cfg.derived["bs_total_tx_power_watt"])
        re_scaling = float(cfg.derived["re_scaling"])

        # User weights xi
        # (Paper notation: \xi_{b,u}). Config key is system_model.user_weight_xi.
        # Keep a backward-compatible fallback for older configs.
        xi_val = float(
            cfg.raw.get("system_model", {}).get(
                "user_weight_xi",
                cfg.raw.get("transmitter", {}).get("weights_xi", 1.0),
            )
        )
        xi = tf.fill([S_eff, U], tf.cast(xi_val, real_dtype))

        # --- Initialize beamformers W
        init_type = str(w_cfg.get("init", "random_scaled"))
                # --- Initialize beamformers W
        

        if _w_init_override is not None:
            w = tf.cast(_w_init_override, complex_dtype)
        else:
            w0 = _complex_normal([S_eff, B, M, u_per_bs], dtype=complex_dtype)
            # Scale to satisfy power roughly (per tone-group, per BS)
            pow_b = tf.reduce_sum(tf.abs(w0) ** 2, axis=[2, 3])  # [S_eff,B]
            target = tf.cast(p_tot_watt / (re_scaling * T), real_dtype)
            scale = tf.sqrt(target / (tf.cast(pow_b, real_dtype) + 1e-12))  # [S_eff,B]
            w = w0 * tf.cast(scale[:, :, tf.newaxis, tf.newaxis], complex_dtype)


        


                # --- Dual variables
        # Draft baseline: mu_b^(0)=1.0, lambda^(0)=0.0
        mu_init = float(w_cfg.get("mu_init", 1.0))
        lam_init = float(w_cfg.get("lambda_init", 0.0))

        mu = tf.fill([batch, B], tf.cast(mu_init, real_dtype))
        if fs is not None:
            L = int(fs.bar_beta.shape[-1])
            lam_val = float(_lam_scalar_override) if _lam_scalar_override is not None else lam_init
            lam = tf.fill([batch, L], tf.cast(lam_val, real_dtype))
            if not fs_dual_enabled:
                lam = tf.zeros_like(lam)
        else:
            L = 0
            # Baseline "No FS protection": define an empty lambda with shape [batch, 0]
            # so downstream code and plots can safely read res.lam.
            lam = tf.zeros_like(mu[:, :0])               
            












        # --- History tensors (collected as python lists then stacked)
        obj_hist = []
        pow_violation_hist = []
        fs_violation_hist = []
        w_delta_hist = []

        converged = False

        mmse = mmse_combiners_and_mse(H_eff, w, noise_var_watt=tf.cast(noise_var_watt, real_dtype))

        for k in range(K_iter):
            w_prev = w

            # 1) MMSE weights for current iterate
            q = xi / tf.cast(mmse.mse, real_dtype)  # [S_eff,U]

            # 2) Beamformer update (per tone-group, flattened)
            # Reshape q and v per BS
            q_bu = tf.reshape(q, [S_eff, B, u_per_bs])  # [S_eff,B,U_per_bs]
            v_bu = tf.reshape(mmse.v, [S_eff, B, u_per_bs, Nr])  # [S_eff,B,U_per_bs,Nr]

            # Self-channels H_bb for each bs to its users
         # --- Keep RHS for served users (d_{b,u}) using self channels
            H_self = _extract_self_channels(H_eff, u_per_bs=u_per_bs)  # [S_eff,B,U_per_bs,Nr,M]
            a_self = tf.einsum(
                "sburm, sbur -> sbum",
                tf.math.conj(H_self),
                v_bu,
                optimize=True
            )  # [S_eff,B,U_per_bs,M]

            # --- FIX: Build D_signal using ALL users (matches paper C_b[n] sum over all (b',u'))
            # mmse.v is [S_eff, U, Nr]
            v_all = tf.cast(mmse.v, complex_dtype)  # [S_eff,U,Nr]
            a_all = tf.einsum(
                "sburm, sur -> sbum",
                tf.math.conj(H_eff),
                v_all,
                optimize=True
            )  # [S_eff,B,U,M]  where a_all[s,b,u] = H_{b->u}^H v_u

            sqrt_q_all = tf.sqrt(tf.maximum(tf.cast(q, real_dtype), 0.0))  # [S_eff,U]
            x_all = (
                tf.cast(sqrt_q_all[:, tf.newaxis, :, tf.newaxis], complex_dtype)
                * tf.cast(a_all, complex_dtype)
            )  # [S_eff,B,U,M]

            # D_signal = sum_u q_u a_all a_all^H
            D_signal = tf.einsum("sbum, sbun -> sbmn", x_all, tf.math.conj(x_all), optimize=True)  # [S_eff,B,M,M]

            # RHS: C = q_{b,u} a_self for served users
            C = tf.einsum("sbu, sbum -> sbmu", tf.cast(q_bu, a_self.dtype), a_self, optimize=True)  # [S_eff,B,M,U_per_bs]


                        # ------------------------
            # FS penalty term
            #   - identity: diagonal scalar f_eff -> adds (f_eff)*I
            #   - steering_rank1: full matrix R_fs -> adds R_fs
            # ------------------------
            R_fs = None
            f_eff = None

            if fs is not None:
                bar_beta = tf.cast(fs.bar_beta, real_dtype)   # [batch,B,L]
                epsilon  = tf.cast(fs.epsilon, real_dtype)    # [T,L]
                delta    = tf.cast(fs.delta, real_dtype)      # [B,T]

                # g[s,b,t,l] = lambda[s,l] * delta[b,t] * epsilon[t,l] * bar_beta[s,b,l]
                g = tf.cast(re_scaling, real_dtype) * tf.einsum("sl,bt,tl,sbl->sbtl", lam, delta, epsilon, bar_beta, optimize=True)  # [batch,B,T,L]
                g_eff = tf.reshape(tf.transpose(g, [0, 2, 1, 3]), [S_eff, B, L])  # [batch*T,B,L]

                corr = getattr(fs, "correlation", "identity")
                if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
                    a = tf.cast(fs.a_bs_fs, complex_dtype)  # [batch,B,L,M]
                    a_eff = tf.reshape(tf.tile(a[:, tf.newaxis, ...], [1, T, 1, 1, 1]), [S_eff, B, L, M])

                    sqrt_g = tf.sqrt(tf.maximum(g_eff, 0.0))  # [S_eff,B,L]
                    u = tf.cast(sqrt_g[..., tf.newaxis], complex_dtype) * a_eff  # [S_eff,B,L,M]

                    # R_fs[s,b] = sum_l g[s,b,l] * a a^H  (rank-L sum)
                    R_fs = tf.einsum("sblm,sbln->sbmn", u, tf.math.conj(u), optimize=True)  # [S_eff,B,M,M]
                else:
                    # identity correlation => diagonal penalty
                    f_eff = tf.reduce_sum(g_eff, axis=-1)  # [S_eff,B]
            else:
                f_eff = tf.zeros([S_eff, B], dtype=real_dtype)


            # mu replicated across tones: [batch*T,B]
            mu_eff = tf.reshape(tf.tile(mu[:, tf.newaxis, :], [1, T, 1]), [S_eff, B])

            rs = tf.cast(re_scaling, real_dtype)
            diag_add = rs * mu_eff + tf.cast(ridge, real_dtype)
            I = tf.eye(M, batch_shape=tf.shape(D_signal)[:-2], dtype=D_signal.dtype)
            D = D_signal + tf.cast(diag_add[:, :, tf.newaxis, tf.newaxis], D_signal.dtype) * I

            if fs is not None:
                if R_fs is not None:
                    D = D + R_fs
                else:
                    # f_eff is defined in the identity-correlation branch
                    D = D + tf.cast(f_eff[:, :, tf.newaxis, tf.newaxis], D_signal.dtype) * I


            # C: [S_eff,B,M,U_per_bs]
            
            # Solve D w = C
            w_new = tf.linalg.solve(D, C)  # [S_eff,B,M,U_per_bs]

            # Damping
            w = tf.cast(damping, complex_dtype) * w_new + tf.cast(1.0 - damping, complex_dtype) * w_prev

            # --- Aggressive FS steering nulling (optional, per config) ---
            aggr_null = bool(w_cfg.get("aggressive_fs_nulling", False))
            if aggr_null and (fs is not None):
                corr = getattr(fs, "correlation", "identity")
                if (corr == "steering_rank1") and (getattr(fs, "a_bs_fs", None) is not None) and (L > 0):
                    # Build a_eff: [S_eff,B,L,M]
                    a = tf.cast(fs.a_bs_fs, complex_dtype)  # [batch,B,L,M]
                    a_eff = tf.reshape(
                        tf.tile(a[:, tf.newaxis, ...], [1, T, 1, 1, 1]),
                        [S_eff, B, L, M],
                    )

                    # Rows are a_l^H, i.e., conj(a)^T
                    # Gram matrix G = A A^H = [S_eff,B,L,L] with entries a_l^H a_k
                    G = tf.einsum(
                        "sblm,sbkm->sblk",
                        tf.math.conj(a_eff),
                        a_eff,
                        optimize=True,
                    )

                    # Regularize for numerical stability (esp. if steering vectors are nearly dependent)
                    reg = tf.cast(float(w_cfg.get("aggressive_fs_nulling_reg", 1e-6)), real_dtype)

                    I_L = tf.eye(L, batch_shape=[S_eff, B], dtype=complex_dtype)
                    G_reg = tf.cast(G, complex_dtype) + tf.cast(reg, complex_dtype) * I_L

                    # Compute A w = [S_eff,B,L,U_per_bs]
                    Aw = tf.einsum(
                        "sblm,sbmu->sblu",
                        tf.math.conj(a_eff),
                        w,
                        optimize=True,
                    )

                    # Solve (A A^H) x = (A w), then subtract A^H x
                    x = tf.linalg.lstsq(G_reg, Aw, fast=False)  # [S_eff,B,L,U_per_bs]
                    correction = tf.einsum(
                        "sblm,sblu->sbmu",
                        a_eff,
                        x,
                        optimize=True,
                    )

                    # Project onto nullspace: w <- (I - A^H (A A^H)^-1 A) w
                    w = w - correction
            # --- end aggressive nulling ---


                        # 3) Dual updates (on aggregated full-band constraints)
            # Reshape w back to [batch,T,B,M,U_per_bs] for constraint sums
            w_full = tf.reshape(w, [batch, T, B, M, u_per_bs])

            # Power per BS (NO delta weighting): re_scaling * sum_t sum_{m,u} |w|^2
            pow_bt = tf.reduce_sum(tf.abs(w_full) ** 2, axis=[3, 4])  # [batch,T,B]
            pow_bt = tf.cast(pow_bt, real_dtype)
            pow_b_raw = tf.cast(re_scaling, real_dtype) * tf.reduce_sum(pow_bt, axis=1)  # [batch,B]

            # --- Dual update uses the *raw* (pre-projection) violation ---
            # If we update mu using post-projection power, mu can never increase.
            p_tot = tf.cast(p_tot_watt, real_dtype)
            mu = tf.maximum(0.0, mu + tf.cast(rho_mu, real_dtype) * (pow_b_raw - p_tot))

            # --- Per-BS power projection safeguard (scale DOWN only) ---
            eps_pow = tf.cast(1e-12, real_dtype)
            scale = tf.sqrt(p_tot / tf.maximum(pow_b_raw, eps_pow))
            scale = tf.minimum(scale, 1.0)
            # Safety margin to avoid tiny positive violation from float32 roundoff
            scale = tf.where(pow_b_raw > p_tot, scale * tf.cast(1.0 - 1e-6, real_dtype), scale)

            w_full = w_full * tf.cast(scale[:, tf.newaxis, :, tf.newaxis, tf.newaxis], complex_dtype)
            w = tf.reshape(w_full, [S_eff, B, M, u_per_bs])

            # Recompute power after projection (for history/metrics)
            pow_bt = tf.reduce_sum(tf.abs(w_full) ** 2, axis=[3, 4])  # [batch,T,B]
            pow_bt = tf.cast(pow_bt, real_dtype)
            pow_b = tf.cast(re_scaling, real_dtype) * tf.reduce_sum(pow_bt, axis=1)  # [batch,B]

            # FS interference per receiver + lambda update


           
            

            # FS interference per receiver + lambda update
            if fs is not None:
                corr = getattr(fs, "correlation", "identity")

                if corr == "steering_rank1" and getattr(fs, "a_bs_fs", None) is not None:
                    # Directional: I_fs depends on beam directions via |a^H w|^2
                    a = tf.cast(fs.a_bs_fs, complex_dtype)  # [batch,B,L,M]

                    proj = tf.einsum(
                        "sblm,stbmu->stblu",
                        tf.math.conj(a),
                        w_full,
                        optimize=True,
                    )  # [batch,T,B,L,U_per_bs]

                    p_dir = tf.reduce_sum(tf.abs(proj) ** 2, axis=-1)  # [batch,T,B,L]

                    # apply delta activity
                    delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])  # [T,B]
                    p_dir = tf.cast(p_dir, real_dtype) * delta_tb[tf.newaxis, :, :, tf.newaxis]

                    I_fs = re_scaling * tf.einsum(
                        "stbl,tl,sbl->sl",
                        p_dir,
                        tf.cast(fs.epsilon, real_dtype),
                        tf.cast(fs.bar_beta, real_dtype),
                        optimize=True,
                    )  # [batch,L]
                else:
                    # Legacy trace model (apply delta activity here; pow_bt is RAW for power accounting)
                    delta_tb = tf.transpose(tf.cast(fs.delta, real_dtype), [1, 0])
                    pow_bt_fs = pow_bt * delta_tb[tf.newaxis, :, :] 
                    I_fs = re_scaling * tf.einsum(
                    "stb,tl,sbl->sl",
                    pow_bt_fs,  # <-- FIX: must use pow_bt_fs (not pow_bt)
                    tf.cast(fs.epsilon, real_dtype),
                    tf.cast(fs.bar_beta, real_dtype),
                    optimize=True,
                    ) 

                i_max = tf.cast(fs.i_max_watt[tf.newaxis, :], real_dtype)

                # Lambda (FS dual) update.
                # NOTE: With Watt-scale constraints (I_fs, I_max ~ 1e-14..1e-10 W), the raw violation
                # (I_fs - I_max) can be extremely small, which makes dual ascent steps vanish unless
                # dual_step_lambda is enormous. Allow an optional normalized update.
                lam_update_mode = str(w_cfg.get("lambda_update_mode", "watt")).lower().strip()
                if lam_update_mode in ("ratio", "normalized", "norm", "relative"):
                    eps_i = tf.cast(1e-30, real_dtype)
                    viol = (I_fs - i_max) / tf.maximum(i_max, eps_i)  # dimensionless
                else:
                    viol = (I_fs - i_max)  # legacy Watt-scale update

                lam = tf.maximum(0.0, lam + tf.cast(rho_lam, real_dtype) * viol)





            else:
                I_fs = tf.zeros([batch, 0], dtype=real_dtype)
                i_max = tf.zeros([batch, 0], dtype=real_dtype)

            # 4) Convergence checks / history
            # w_delta: max relative change
            # NOTE: tf.linalg.norm() on complex tensors can return a complex dtype in some TF versions.
            # We therefore take tf.abs() to ensure a real-valued norm for the relative-change test.
            num = tf.abs(tf.linalg.norm(w - w_prev))
            den = tf.abs(tf.linalg.norm(w_prev))
            den = tf.maximum(den, tf.cast(1e-12, den.dtype))
            w_delta = tf.cast(num / den, real_dtype)

            # Objective proxy: sum-rate (averaged over tones by construction)
            # sum_rate_bpsphz = (1/T) * sum_u log2(1+sinr)
                        # Objective proxy: sum-rate for the POST-update iterate.
            # To keep MMSE work at ~1x per iteration, compute MMSE here for the new beamformers
            # and carry it into the next iteration.
            mmse_next = mmse_combiners_and_mse(H_eff, w, noise_var_watt=tf.cast(noise_var_watt, real_dtype))

            # sum_rate_bpsphz = (1/T) * sum_u log2(1+sinr)
            try:
                from sionna.phy.utils import log2  # type: ignore

                rate = log2(1.0 + tf.cast(mmse_next.sinr, real_dtype))
            except Exception:
                rate = tf.math.log(1.0 + tf.cast(mmse_next.sinr, real_dtype)) / tf.math.log(tf.constant(2.0, dtype=real_dtype))
            sum_rate = tf.reduce_mean(tf.reduce_sum(rate, axis=-1))  # scalar average over batch*T

            obj_hist.append(sum_rate)
            pow_violation_hist.append(tf.reduce_max(pow_b - tf.cast(p_tot_watt, real_dtype)))
            if fs is not None:
                fs_violation_hist.append(tf.reduce_max(I_fs - i_max))
            else:
                fs_violation_hist.append(tf.cast(0.0, real_dtype))
            w_delta_hist.append(w_delta)

            # Carry MMSE state into the next iteration
            mmse = mmse_next


            if verbose:
                tf.print(
                    "iter", k,
                    "sum_rate", sum_rate,
                    "w_delta", w_delta,
                    "max_pow_violation", pow_violation_hist[-1],
                    "max_fs_violation", fs_violation_hist[-1],
                )

            if float(w_delta.numpy()) < tol:
                converged = True
                break

        # Recompute MMSE metrics for the final iterate (consistent with returned beamformers)
        mmse_final = mmse

        # Final reshape for output
        w_out = tf.reshape(w, [batch, T, B, M, u_per_bs])

        history = {
            "sum_rate": tf.stack(obj_hist, axis=0),
            "max_power_violation_watt": tf.stack(pow_violation_hist, axis=0),
            "max_fs_violation_watt": tf.stack(fs_violation_hist, axis=0),
            "w_delta": tf.stack(w_delta_hist, axis=0),
        }

        return ReceiverResult(
            w=w_out,
            mmse=mmse_final,
            mu=mu,
            lam=lam,
            num_iter=len(obj_hist),
            converged=converged,
            history=history,
        )
    
    def _solve_budget_dual_with_lambda_search(
        self,
        cfg: ResolvedConfig,
        H: tf.Tensor,
        noise_var_watt: float,
        fs: FsStats,
        bs_total_tx_power_watt: Optional[float],
        fs_mode: str,
    ) -> ReceiverResult:
        """Budget-dual with automatic scalar-λ search (bisection) + safeguard.

        Notes
        -----
        - Requires fs.correlation == "steering_rank1" (directional steering model).
        - Searches λ in [fs_lambda_min, fs_lambda_max] to get max(I_fs)/I_max in [ratio_min, 1].
          where ratio_min = 10^(-tol_db/10), i.e. within tol_db *below* the limit.
        - If the bracket cannot find a feasible λ:
            * fs_mode == "hybrid": fall back to hard-null (aggressive_fs_nulling ON)
            * fs_mode == "budget_dual": raise RuntimeError
        """
        w_cfg = dict(cfg.raw.get("receiver", {}).get("wmmse", {}))

        if getattr(fs, "correlation", None) != "steering_rank1":
            raise ValueError(
                "fs_lambda_search requires fs.correlation == 'steering_rank1' (directional steering model)."
            )

        # Dtypes
        precision = cfg.derived.get("tf_precision", "single")
        real_dtype = tf.float32 if precision == "single" else tf.float64
        complex_dtype = tf.complex64 if precision == "single" else tf.complex128

        # Dimensions (match solve())
        B = int(cfg.derived["num_bs"])
        u_per_bs = int(cfg.raw["topology"]["num_ut_per_sector"])
        U = int(cfg.derived["num_ut"])
        M = int(cfg.raw["channel_model"]["num_bs_ant"])
        Nr = int(cfg.raw["channel_model"]["num_ut_ant"])
        T = int(cfg.derived["num_re_sim"])
        batch = tf.shape(H)[0]

        # Power + scaling
        p_tot_watt = float(bs_total_tx_power_watt) if bs_total_tx_power_watt is not None else float(cfg.derived["bs_total_tx_power_watt"])
        re_scaling = float(cfg.derived["re_scaling"])

        # Search params
        lam_lo = float(w_cfg.get("fs_lambda_min", 0.0))
        lam_hi = float(w_cfg.get("fs_lambda_max", 1e16))
        tol_db = float(w_cfg.get("fs_lambda_search_tol_db", 0.2))
        max_iter = int(w_cfg.get("fs_lambda_search_max_iter", 25))

        # Target: within tol_db BELOW the limit (never above)
        ratio_min = 10.0 ** (-tol_db / 10.0)

        # IMPORTANT: use fs.i_max_watt actually passed in (runner may override per sweep point)
        i_max_vec = tf.cast(fs.i_max_watt, real_dtype)  # [L]
        i_max_vec = tf.maximum(i_max_vec, tf.cast(1e-30, real_dtype))  # avoid divide-by-zero
        i_max_min_for_msg = float(tf.reduce_min(i_max_vec).numpy())

        # ---- Build one fixed init beamformer (same for all λ evaluations)
        if len(H.shape) == 5:
            H_t = tf.tile(H[:, tf.newaxis, ...], [1, T, 1, 1, 1, 1])  # [batch,T,B,U,Nr,M]
        elif len(H.shape) == 6:
            H_t = H
        else:
            raise ValueError("H must have shape [batch,B,U,Nr,M] or [batch,T,B,U,Nr,M]")

        H_eff = tf.reshape(H_t, [-1, B, U, Nr, M])  # [batch*T, B, U, Nr, M]
        S_eff = tf.shape(H_eff)[0]

        # random init scaled to roughly meet per-BS budget per tone-group
        w0 = _complex_normal([S_eff, B, M, u_per_bs], dtype=complex_dtype)
        pow_b0 = tf.reduce_sum(tf.abs(w0) ** 2, axis=[2, 3])  # [S_eff,B]
        target = tf.cast(p_tot_watt / (re_scaling * T), real_dtype)
        scale0 = tf.sqrt(target / (tf.cast(pow_b0, real_dtype) + tf.cast(1e-12, real_dtype)))  # [S_eff,B]
        w_init = w0 * tf.cast(scale0[:, :, tf.newaxis, tf.newaxis], complex_dtype)

        def eval_lam(lam_scalar: float) -> Tuple[float, float, ReceiverResult]:
            """Return (ratio=max_{batch,l} I_fs/I_max, max_I_watt, ReceiverResult)."""
            res = self.solve(
                cfg=cfg,
                H=H,
                noise_var_watt=noise_var_watt,
                fs=fs,
                bs_total_tx_power_watt=bs_total_tx_power_watt,
                _lam_scalar_override=lam_scalar,
                _freeze_lambda=True,
                _force_aggressive_fs_nulling=False,
                _w_init_override=w_init,
                _skip_fs_lambda_search=True,
            )

            I_fs_max_tensor, I_fs = _max_fs_interference_watt_rank1(
                res.w, fs=fs, re_scaling=re_scaling, real_dtype=real_dtype
            )  # scalar, [batch,L]

            I_fs_max_val = float(I_fs_max_tensor.numpy())
            ratio = tf.reduce_max(I_fs / i_max_vec[tf.newaxis, :])
            return float(ratio.numpy()), I_fs_max_val, res

        # --- Bracket check
        ratio_lo, I_lo, res_lo = eval_lam(lam_lo)
        if ratio_lo <= 1.0:
            # Already satisfies at the minimum λ: constraint is not binding.
            hist = dict(res_lo.history)
            hist["fs_lambda_search_lambda"] = tf.constant(lam_lo, dtype=real_dtype)
            hist["fs_lambda_search_ratio"] = tf.constant(ratio_lo, dtype=real_dtype)
            return ReceiverResult(
                w=res_lo.w,
                mmse=res_lo.mmse,
                mu=res_lo.mu,
                lam=res_lo.lam,
                num_iter=res_lo.num_iter,
                converged=res_lo.converged,
                history=hist,
            )

        ratio_hi, I_hi, res_hi = eval_lam(lam_hi)
        if ratio_hi > 1.0:
            msg = (
                f"fs_lambda_search failed: even λ={lam_hi:g} gives max_I={I_hi:.3e} W > "
                f"I_max(min)={i_max_min_for_msg:.3e} W. "
                "Increase receiver.wmmse.fs_lambda_max or use fs_enforcement='hard_null'/'hybrid'."
            )
            if fs_mode == "hybrid":
                return self.solve(
                    cfg=cfg,
                    H=H,
                    noise_var_watt=noise_var_watt,
                    fs=fs,
                    bs_total_tx_power_watt=bs_total_tx_power_watt,
                    _freeze_lambda=True,
                    _force_aggressive_fs_nulling=True,
                    _w_init_override=w_init,
                    _skip_fs_lambda_search=True,
                )
            raise RuntimeError(msg)

        # --- Bisection: keep smallest feasible λ (ratio <= 1)
        best_res = res_hi
        best_ratio = ratio_hi
        best_lam = lam_hi

        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            ratio_mid, _, res_mid = eval_lam(lam_mid)

            if ratio_mid > 1.0:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
                best_lam = lam_mid
                best_res = res_mid
                best_ratio = ratio_mid

            # Stop once we're within tol_db of I_max (from below)
            if best_ratio >= ratio_min:
                break

        # --- Safeguard: if somehow still violating, push λ up; if that fails, fall back
        if best_ratio > 1.0 + 1e-6:
            ratio_try, _, res_try = eval_lam(best_lam * 10.0)
            if ratio_try <= 1.0:
                best_res = res_try
                best_ratio = ratio_try
                best_lam = best_lam * 10.0
            elif fs_mode == "hybrid":
                best_res = self.solve(
                    cfg=cfg,
                    H=H,
                    noise_var_watt=noise_var_watt,
                    fs=fs,
                    bs_total_tx_power_watt=bs_total_tx_power_watt,
                    _freeze_lambda=True,
                    _force_aggressive_fs_nulling=True,
                    _w_init_override=w_init,
                    _skip_fs_lambda_search=True,
                )
                # In hard-null fallback, FS will be well below limit.
                best_ratio = 0.0
            else:
                raise RuntimeError(
                    "fs_lambda_search ended with FS violation; use fs_enforcement='hybrid' for a hard-null safeguard."
                )

        # Annotate history for debugging/plots
        hist = dict(best_res.history)
        hist["fs_lambda_search_lambda"] = tf.constant(best_lam, dtype=real_dtype)
        hist["fs_lambda_search_ratio"] = tf.constant(best_ratio, dtype=real_dtype)
        return ReceiverResult(
            w=best_res.w,
            mmse=best_res.mmse,
            mu=best_res.mu,
            lam=best_res.lam,
            num_iter=best_res.num_iter,
            converged=best_res.converged,
            history=hist,
        )


