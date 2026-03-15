This patch must be applied before running slurm/twc_full.slurm.

Files replaced:
- src/fr3_twc/pipeline.py
- src/fr3_twc/unfolded.py
- src/fr3_twc/wideband_channel.py

Why this patch is needed:
1. The TWC batch builder was using full-band BS power (`bs_total_tx_power_watt`) instead of the simulated-band power budget (`bs_power_budget_sim_watt`).
2. The TWC FS-budget sweep was only scaling `i_max_watt`; it was not refreshing `risk_score`, `static_gate`, and `scenario_features`, so the -6 / -10 / -14 dB sweep in twc_full was not self-consistent.
3. The merged `metrics_summary_with_legacy.csv` was mixing TWC normalized FS-excess ratios with legacy watt-domain FS metrics.
4. The unfolded model produced many `complex64 -> float32` TensorFlow warnings; this patch rewrites the real-parameter scaling path to avoid those warnings and adds gradient clipping.
5. `metrics_summary.csv` is now saved as an aggregated summary across evaluation batches, and the raw unaggregated rows are saved to `metrics_summary_raw.csv`.

Run order:
1. Apply patch in repo root.
2. Commit and push.
3. On Narval: git pull.
4. Re-run smoke first: sbatch slurm/twc_smoke.slurm
5. Only if smoke completes cleanly, run full: sbatch slurm/twc_full.slurm
