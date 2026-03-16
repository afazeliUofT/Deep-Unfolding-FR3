# TWC next-step package v10: budget-aware primal-dual unfolding

This package adds the next algorithmic step after the fairness-fixed TWC pipeline:

- **New baseline:** `budgeted_primal_dual_pgd`
- **New proposed method:** `budget_aware_primal_dual_unfolded`
- **Teacher guidance:** optional distillation from a classical `budget_dual` WMMSE teacher solved on the **same wideband batch**
- **Separate configs and Slurm jobs:** `twc_pd_smoke` and `twc_pd_full`

## Main idea

The previous learned method used fixed penalty surrogates and learned risk gating, but it did not explicitly carry the FS dual variables. This package changes the optimization model to a **primal-dual unfolded architecture**:

1. Explicit per-FS dual variables are initialized from ISED/geometry-induced coupling risk.
2. Beamformers are updated by primal gradient steps on a Lagrangian-style objective.
3. Dual variables are updated layer-by-layer from the normalized FS budget excess.
4. Training is stabilized by a light **teacher distillation** term from the classical `budget_dual` WMMSE, solved on the exact same wideband batch with matched simulated-band power budget and `re_scaling=1`.

## Files in this package

### Replace
- `src/fr3_twc/algorithms.py`
- `src/fr3_twc/unfolded.py`
- `src/fr3_twc/pipeline.py`
- `scripts/twc_verify_outputs.py`

### Add
- `src/fr3_twc/teacher.py`
- `configs/twc_pd_smoke.yaml`
- `configs/twc_pd_full.yaml`
- `slurm/twc_pd_smoke.slurm`
- `slurm/twc_pd_full.slurm`
- `README_TWC_PD_NEXTSTEP_v10.md`

## Simultaneous execution safety

The new Slurm jobs are safe to run simultaneously because they write to **different result roots**:

- smoke: `results/twc_pd_smoke_bundle/`
- full: `results/twc_pd_full_bundle/`

They also produce different Slurm output names:

- `fr3_twc_pd_smoke-<JOBID>.out`
- `fr3_twc_pd_full-<JOBID>.out`

## Expected new algorithms in metrics_summary.csv

- `static_notch_mf`
- `risk_neutral_pgd`
- `wideband_pgd`
- `budgeted_primal_dual_pgd`
- `budget_aware_primal_dual_unfolded`

Legacy comparisons are still generated under `legacy_baseline/` by the same runner.
