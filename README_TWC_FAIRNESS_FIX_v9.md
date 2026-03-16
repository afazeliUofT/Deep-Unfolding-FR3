# TWC Fairness Fix v9

This patch fixes a methodological issue in the legacy comparison protocol.

Problem fixed:
- The legacy baseline runner was hard-forcing `experiment.freeze_topology=true`.
- The TWC smoke/full pipeline configs use `experiment.freeze_topology=false`.
- Therefore, `metrics_summary_with_legacy.csv` was mixing TWC results averaged over random topologies with legacy results evaluated on a single frozen geometry. That is not an apples-to-apples comparison.

Files replaced:
- `scripts/twc_run_legacy_baseline.py`
- `configs/twc_legacy_baseline.yaml`

What changes:
- The legacy runner now inherits the config value by default.
- `configs/twc_legacy_baseline.yaml` is aligned to `freeze_topology: false`, matching `configs/twc_full.yaml`.
- Smoke already inherits `configs/twc_smoke.yaml`, which already has `freeze_topology: false`.

What to rerun:
1. `sbatch slurm/twc_smoke.slurm`
2. `sbatch slurm/twc_full.slurm`

After rerun, push back:
- `results/twc_smoke_bundle/`
- `results/twc_full_bundle/`
- `fr3_twc_smoke-*.out`
- `fr3_twc_full-*.out`
