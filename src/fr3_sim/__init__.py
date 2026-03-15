"""FR3 WMMSE + Sionna simulation package (baseline, no deep unfolding).

The codebase is intentionally modular:
- config loading/validation
- topology generation (hexgrid)
- channel generation (UMi pathloss + i.i.d. Rayleigh)
- receiver baseline (classical WMMSE with FS protection multipliers)
- runner/orchestration, metrics, plotting

"""

from .config import load_config

__all__ = ["load_config"]
