# TWC BC Core Sync v17

This package fixes the BC-track repository sync issue.

What happened:
- The repo already contains the BC configs, BC slurms, BC README, and BC result bundles.
- But the core BC logic in `src/fr3_twc/algorithms.py` and `src/fr3_twc/pipeline.py`
  was not updated correctly.
- As a result, the latest BC results were produced with only a partial BC update.

This package replaces exactly:
- `src/fr3_twc/algorithms.py`
- `src/fr3_twc/pipeline.py`

Main fixes restored by this package:
1. Boundary-aware outage surrogate in `algorithms.py`
2. Straight-through repaired/recovered training consistency in `pipeline.py`
3. RR teacher stage switch in `pipeline.py`

Do NOT remove the current BC configs/slurms. They are already in the repo.
After applying this package, rerun only:
- `slurm/twc_bc_smoke.slurm`
- `slurm/twc_bc_full.slurm`

These two jobs are safe to run simultaneously because they write to different result roots.
