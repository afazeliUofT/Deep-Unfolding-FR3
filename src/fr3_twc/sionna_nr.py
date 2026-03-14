"""Best-effort Sionna NR/OFDM resource-grid construction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import math
import numpy as np

from .config_utils import twc_get
from .types import NrGridContext


def _try_import_resource_grid():
    """Import ResourceGrid from the most common Sionna paths."""
    candidates = [
        ("sionna.phy.ofdm", "ResourceGrid"),
        ("sionna.ofdm", "ResourceGrid"),
    ]
    last_err: Optional[Exception] = None
    for module_name, cls_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as exc:  # pragma: no cover - import path depends on installed Sionna version
            last_err = exc
    raise ImportError(f"Could not import Sionna ResourceGrid. Last error: {last_err}")


def _try_import_carrier_config():
    """Import CarrierConfig if available."""
    candidates = [
        ("sionna.nr", "CarrierConfig"),
        ("sionna.phy.nr", "CarrierConfig"),
    ]
    for module_name, cls_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception:
            continue
    return None


def build_nr_grid(cfg: Any) -> NrGridContext:
    """Build an NR-like OFDM grid.

    Notes
    -----
    The TWC pipeline uses the repository's `system_model.num_re_sim` as the
    number of *simulated subbands* for coexistence optimization. The Sionna
    `ResourceGrid` is used for NR/OFDM numerology metadata, while the actual
    coexistence subbands are aligned with the repository's reduced-band RE
    groups so that the incumbent-overlap matrix `epsilon` remains consistent.
    """
    num_subbands = int(twc_get(cfg, ("nr", "num_subbands"), cfg.derived["num_re_sim"]))
    num_symbols = int(twc_get(cfg, ("nr", "num_ofdm_symbols"), 14))
    guard = int(twc_get(cfg, ("nr", "guard_subcarriers"), 2))
    dc_null = bool(twc_get(cfg, ("nr", "dc_null"), True))
    pilot_pattern = twc_get(cfg, ("nr", "pilot_pattern"), None)
    pilot_syms = twc_get(cfg, ("nr", "pilot_ofdm_symbol_indices"), None)

    # Keep the coexistence subbands consistent with the repository's reduced-band model
    total_bandwidth_hz = float(cfg.derived["bandwidth_total_hz"])
    carrier_frequency_hz = float(cfg.raw["system_model"]["carrier_frequency_hz"])
    subband_bw_hz = total_bandwidth_hz / float(num_subbands)

    # Build center frequencies of the simulated subbands
    freq_offsets = (np.arange(num_subbands, dtype=np.float64) - 0.5 * (num_subbands - 1)) * subband_bw_hz
    subband_center_hz = carrier_frequency_hz + freq_offsets

    fft_size = 1 << int(math.ceil(math.log2(max(16, num_subbands + 2 * guard + (1 if dc_null else 0)))))

    metadata: Dict[str, Any] = {
        "carrier_config_available": False,
        "resource_grid_backend": "fallback",
    }

    resource_grid: Any = None
    try:
        ResourceGrid = _try_import_resource_grid()
        kwargs: Dict[str, Any] = dict(
            num_ofdm_symbols=num_symbols,
            fft_size=fft_size,
            subcarrier_spacing=float(cfg.raw["system_model"]["subcarrier_spacing_hz"]),
            num_tx=1,
            num_streams_per_tx=1,
            dc_null=dc_null,
            num_guard_carriers=[guard, guard],
        )
        if pilot_pattern is not None:
            kwargs["pilot_pattern"] = pilot_pattern
        if pilot_syms is not None:
            kwargs["pilot_ofdm_symbol_indices"] = list(pilot_syms)
        resource_grid = ResourceGrid(**kwargs)
        metadata["resource_grid_backend"] = str(ResourceGrid.__module__)
    except Exception:
        resource_grid = None

    CarrierConfig = _try_import_carrier_config()
    if CarrierConfig is not None:
        try:
            # Best effort only. Different Sionna versions expose slightly different fields.
            scs_khz = int(round(float(cfg.raw["system_model"]["subcarrier_spacing_hz"]) / 1e3))
            n_size_grid = max(1, int(math.ceil(total_bandwidth_hz / float(cfg.raw["system_model"]["subcarrier_spacing_hz"]) / 12.0)))
            carrier_cfg = CarrierConfig(n_size_grid=n_size_grid, subcarrier_spacing=scs_khz)
            metadata["carrier_config_available"] = True
            metadata["carrier_config_type"] = str(type(carrier_cfg))
        except Exception:
            pass

    return NrGridContext(
        resource_grid=resource_grid,
        num_subbands=num_subbands,
        num_ofdm_symbols=num_symbols,
        subband_bw_hz=float(subband_bw_hz),
        subband_center_hz=subband_center_hz,
        carrier_frequency_hz=float(carrier_frequency_hz),
        total_bandwidth_hz=float(total_bandwidth_hz),
        fft_size=int(fft_size),
        guard_subcarriers=int(guard),
        metadata=metadata,
    )
