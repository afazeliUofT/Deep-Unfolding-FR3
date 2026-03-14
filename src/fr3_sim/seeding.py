"""Deterministic seeding utilities.

This module centralizes all RNG seeding to make experiments reproducible.
It seeds:
- Python's `random`
- NumPy
- TensorFlow
- (optionally) Sionna's global seed/config if available

This module must NOT:
- Import project-specific modules (to avoid circular imports)
- Perform any I/O
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import random

import numpy as np


def _try_seed_sionna(seed: int) -> None:
    """Best-effort seeding of Sionna (works across multiple versions).

    Important
    ---------
    When TensorFlow op determinism is enabled, Sionna must *not* initialize its
    internal TF RNG via `tf.random.Generator.from_non_deterministic_state()`,
    or TensorFlow will raise a RuntimeError.

    We therefore proactively set Sionna's internal RNG objects (if present)
    using `from_seed(seed)`.
    """
    try:
        import tensorflow as tf

        # We must target the *actual* singleton used by Sionna blocks.
        # In Sionna 1.2.x this is `sionna.phy.config.config` (instance of Config).
        cfg_obj = None
        try:
            from sionna.phy.config import config as cfg_obj  # type: ignore
        except Exception:
            # Fallback: some versions re-export `config` from the package.
            try:
                from sionna.phy import config as cfg_obj  # type: ignore
            except Exception:
                cfg_obj = None

        if cfg_obj is None:
            return

        # 1) If a public seed exists, set it (best effort)
        if hasattr(cfg_obj, "seed"):
            try:
                cfg_obj.seed = int(seed)
            except Exception:
                pass

        # 2) Force deterministic RNG objects so Sionna never falls back to
        #    `from_non_deterministic_state()` under TF determinism.
        try:
            if hasattr(cfg_obj, "_tf_rng"):
                cfg_obj._tf_rng = tf.random.Generator.from_seed(int(seed))
            if hasattr(cfg_obj, "_np_rng"):
                cfg_obj._np_rng = np.random.default_rng(int(seed))
        except Exception:
            pass
    except Exception:
        pass


def set_global_seed(seed: int, deterministic_tf: bool = True) -> None:
    """Set seeds for Python/NumPy/TensorFlow (best effort deterministic).

    Notes
    -----
    Full determinism on GPU can require additional env vars (see README).
    """
    seed = int(seed)

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow (import here to allow env vars to be set earlier if desired)
    import tensorflow as tf

    tf.random.set_seed(seed)

    # Best-effort deterministic ops
    if deterministic_tf:
        # Recommended by TF; some ops may still be non-deterministic depending on GPU/TF version.
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            # Older TF versions may not have this
            pass

    # Sionna
    _try_seed_sionna(seed)


@dataclass(frozen=True)
class Precision:
    """TensorFlow dtypes used across the project."""

    real: "tensorflow.dtypes.DType"
    complex: "tensorflow.dtypes.DType"


def get_precision(precision: str) -> Precision:
    """Return TensorFlow dtypes for a given precision string."""
    import tensorflow as tf

    if precision == "single":
        return Precision(real=tf.float32, complex=tf.complex64)
    if precision == "double":
        return Precision(real=tf.float64, complex=tf.complex128)
    raise ValueError("precision must be 'single' or 'double'")
