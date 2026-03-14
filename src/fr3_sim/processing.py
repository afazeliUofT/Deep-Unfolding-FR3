"""Shared signal-processing computations.

This module contains reusable tensor operations used by both the receiver
and metrics layers:
- Effective channel computation E_u = [H_0u W_0, ..., H_{B-1,u} W_{B-1}]
- MMSE combining vectors and MSE/SINR extraction

This module must NOT:
- Implement WMMSE iterations (handled in receiver.py)
- Do file I/O
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf


@dataclass(frozen=True)
class MmseOutput:
    """MMSE combiner outputs."""

    v: tf.Tensor  # [batch, U, Nr]
    mse: tf.Tensor  # [batch, U]
    sinr: tf.Tensor  # [batch, U]
    desired: tf.Tensor  # [batch, U, Nr]


def compute_effective_channel(H: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    """Compute the per-user effective channel matrix E.

    Parameters
    ----------
    H: tf.Tensor
        BS->UT channels with shape [batch, B, U, Nr, M].
    W: tf.Tensor
        Beamformers with shape [batch, B, M, U_per_bs].

    Returns
    -------
    tf.Tensor
        Effective channel E with shape [batch, U, Nr, B*U_per_bs].
        The stream ordering is BS-major, then local-stream index.
    """
    # A: [batch, U, Nr, B, U_per_bs]
    # H: [s,b,u,r,m], W: [s,b,m,k] -> A: [s,u,r,b,k]
    A = tf.einsum("sburm, sbmk -> surbk", H, W, optimize=True)
    # Flatten streams: [batch, U, Nr, B*U_per_bs]
    shape = tf.shape(A)
    batch = shape[0]
    U = shape[1]
    Nr = shape[2]
    B = shape[3]
    K = shape[4]
    E = tf.reshape(A, [batch, U, Nr, B * K])
    return E


def extract_desired_vectors(E: tf.Tensor) -> tf.Tensor:
    """Extract the desired stream vector for each user from E.

    Assumes that the global stream index equals the global user index.
    This holds when users are ordered BS-major and each user has 1 stream.

    Parameters
    ----------
    E: tf.Tensor
        Effective channel [batch, U, Nr, U].

    Returns
    -------
    tf.Tensor
        desired vectors with shape [batch, U, Nr].
    """
    # Move Nr first to use diag_part on the last two dims
    E_t = tf.transpose(E, [0, 2, 1, 3])  # [batch, Nr, U, U]
    diag = tf.linalg.diag_part(E_t)  # [batch, Nr, U]
    return tf.transpose(diag, [0, 2, 1])  # [batch, U, Nr]


def mmse_combiners_and_mse(H: tf.Tensor, W: tf.Tensor, noise_var_watt: tf.Tensor, mse_floor: float = 1e-12) -> MmseOutput:
    """Compute MMSE combiners and corresponding MSE/SINR.

    Parameters
    ----------
    H: tf.Tensor
        Channels [batch, B, U, Nr, M]
    W: tf.Tensor
        Beamformers [batch, B, M, U_per_bs]
    noise_var_watt: tf.Tensor
        Noise variance per UT antenna (per RE) in Watt. Shape broadcastable to [batch, U].
    mse_floor: float
        Minimum MSE for numerical stability.

    Returns
    -------
    MmseOutput
        v, mse, sinr, and desired vectors.
    """
    E = compute_effective_channel(H, W)  # [batch, U, Nr, streams]
    desired = extract_desired_vectors(E)  # [batch, U, Nr]

    # Covariance of received vector y_u: R = E E^H + sigma^2 I
    R = tf.matmul(E, E, adjoint_b=True)  # [batch, U, Nr, Nr]

    Nr = tf.shape(R)[-1]
    eye = tf.eye(Nr, batch_shape=tf.shape(R)[:-2], dtype=R.dtype)

    sigma2 = tf.cast(noise_var_watt, R.dtype)
    # Broadcast sigma2 to [batch, U]
    batch = tf.shape(H)[0]
    U = tf.shape(H)[2]
    sigma2 = tf.broadcast_to(sigma2, [batch, U])
    R = R + sigma2[:, :, tf.newaxis, tf.newaxis] * eye

    # Solve R v = desired
    v = tf.linalg.solve(R, desired[..., tf.newaxis])  # [batch,U,Nr,1]
    v = tf.squeeze(v, axis=-1)  # [batch,U,Nr]

    # MMSE MSE: e = 1 - desired^H v (real)
    inner = tf.reduce_sum(tf.math.conj(desired) * v, axis=-1)
    mse = 1.0 - tf.math.real(inner)
    mse = tf.clip_by_value(mse, mse_floor, 1e9)

    sinr = tf.clip_by_value(1.0 / mse - 1.0, 0.0, 1e12)

    return MmseOutput(v=v, mse=mse, sinr=sinr, desired=desired)
