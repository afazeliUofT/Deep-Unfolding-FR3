"""Dataclasses shared across the TWC pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from fr3_sim.channel import FsStats
from fr3_sim.topology import FixedServiceLocations, TopologyData


@dataclass
class NrGridContext:
    """NR-like frequency grid metadata used by the wideband simulator."""

    resource_grid: Any
    num_subbands: int
    num_ofdm_symbols: int
    subband_bw_hz: float
    subband_center_hz: np.ndarray
    carrier_frequency_hz: float
    total_bandwidth_hz: float
    fft_size: int
    guard_subcarriers: int
    metadata: Dict[str, Any]


@dataclass
class WidebandBatch:
    """A single wideband coexistence mini-batch."""

    topo: TopologyData
    fs_loc: FixedServiceLocations
    fs_stats: FsStats
    grid: NrGridContext
    h_wb: tf.Tensor                      # [S,K,B,U,Nr,M]
    h_eff: tf.Tensor                     # [S,K,B,U,M]
    serving_bs: np.ndarray               # [U]
    user_weights: tf.Tensor              # [U]
    noise_var_watt: float
    bs_power_budget_watt: float
    risk_score: tf.Tensor                # [S,K]
    overlap: tf.Tensor                   # [K,L]
    static_gate: tf.Tensor               # [S,K]
    delay_spread_ns: tf.Tensor           # [S,B,U]
    los_mask: tf.Tensor                  # [S,B,U]
    scenario_features: tf.Tensor         # [S,F]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmResult:
    """Output of one algorithm on one mini-batch."""

    name: str
    w: tf.Tensor                         # [S,K,B,M,U_per_bs]
    runtime_s: float
    history: Dict[str, List[float]]
    extra: Dict[str, Any] = field(default_factory=dict)
