# TWC BC tuning sweep v18

This package adds a targeted tuning sweep on top of the already-correct BC core logic.

Why this sweep:
- v17 fixed the BC code sync and the jobs are now valid.
- The remaining gap is scientific, not a bug:
  - smoke learned feasible method is already near parity with the classical recovered baseline,
  - full learned feasible method is still below the classical recovered baseline,
  - learned feasible runtime is still not better than the classical recovered baseline.
- So the right next step is a focused BC hyperparameter sweep, not another architecture rewrite.

Variants:
- A: balanced rate/feasibility
- B: stronger rate bias and longer training
- C: fast-feasible recovery

All result roots and job names are unique, so the slurms can run simultaneously.
