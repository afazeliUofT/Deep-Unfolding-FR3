# TWC Boundary-Consistent RR Next Step v16

This package adds the next recovery-consistent training step for the RC track.

Main changes:
1. Boundary-aware outage surrogate: feasible points close to the FS boundary are no longer penalized like ~50% outages.
2. Straight-through postprocessed training for RR/FR modes: during training, the loss is evaluated on the repaired/recovered output using a stop-gradient surrogate, so training matches evaluation much better.
3. RR teacher schedule: early epochs use the stronger rate-oriented budget-dual teacher; later epochs switch to the repaired+recovered teacher.
4. New parallel configs/slurms:
   - configs/twc_bc_smoke.yaml
   - configs/twc_bc_full.yaml
   - slurm/twc_bc_smoke.slurm
   - slurm/twc_bc_full.slurm

Replace:
- src/fr3_twc/algorithms.py
- src/fr3_twc/pipeline.py

Add:
- configs/twc_bc_smoke.yaml
- configs/twc_bc_full.yaml
- slurm/twc_bc_smoke.slurm
- slurm/twc_bc_full.slurm
- README_TWC_BC_NEXTSTEP_v16.md

Run only the new BC jobs for this step.
