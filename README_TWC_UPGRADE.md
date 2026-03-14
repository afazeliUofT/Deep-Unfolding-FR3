# TWC Upgrade Package for `Deep-Unfolding-FR3`

This package adds a **wideband, frequency-selective, risk-aware, scenario-adaptive** FR3 coexistence pipeline on top of the current repository.

## Main technical upgrade

1. **Sionna-based 5G NR numerology**
   - Builds an NR-like OFDM resource grid using Sionna.
   - Uses the repository's 3GPP-style topology generation and ISED fixed-service loader.

2. **Frequency-selective FR3 channel instead of flat tone-groups**
   - Generates wideband MIMO channels across subbands.
   - Adds a delay-spread driven frequency-selective model and reports tone-group approximation error.

3. **Hybrid coexistence instead of spatial-only coexistence**
   - Uses the actual ISED FS center-frequency / bandwidth data to build a per-subband incumbent-overlap mask.
   - Combines **subband-aware cognitive avoidance** with **spatial precoding**.

4. **Risk-aware deep unfolding**
   - Adds a learnable unfolded projected-gradient precoder.
   - Uses CVaR-style tail-risk penalties for incumbent protection.
   - Conditions the layer parameters on scenario features (geometry, overlap intensity, risk histogram).

5. **TWC-style analysis outputs**
   - Legacy baseline comparison.
   - Static notch baseline.
   - Fixed-parameter wideband PGD baseline.
   - Proposed scenario-adaptive unfolded baseline.
   - Runtime/convergence plots.
   - Tone-group error and coherence-bandwidth proxy plots.
   - User-weight sensitivity plots.
   - Reference-geometry plot.
