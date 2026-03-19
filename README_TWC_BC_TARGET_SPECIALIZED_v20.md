# TWC BC target-specialized v20

This package adds one final **target-specialized** check around the strongest v19 learned-feasible family.

Why v20 is justified:
- v19 resolved the engineering/runtime issues.
- Variant **D** was the best learned-feasible **full -10 dB** point, but it still trailed the classical repaired+recovered feasible baseline.
- All current learned BC models are trained jointly across `[-6, -10, -14] dB` FS targets.
- The cleanest final question is whether the remaining gap is mainly caused by **multi-target joint training**.
- So v20 adds **single-target specialized** versions of the strongest v19 families for the central `-10 dB` benchmark.

What this package does:
- no core `src/` code is changed
- adds a tiny config-override workflow
- adds six specialized jobs:
  - D10 smoke/full
  - E10 smoke/full
  - F10 smoke/full

Interpretation goal:
- If a specialized learned-feasible model now beats the current classical feasible baseline at `-10 dB`, or matches it with lower runtime, then the learned mechanism is strong enough to headline.
- If not, the paper is still publishable, but the **flagship** should be the classical feasible solver + wideband framework, with the learned track as a strong secondary result.
