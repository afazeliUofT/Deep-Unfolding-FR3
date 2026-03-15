What this patch does
- Re-applies the safe unfolded-model call path (`model(batch=..., training=...)`).
- Keeps the user-weight sensitivity comparison on the same geometry within each sample.
- Keeps only the latest legacy-baseline metrics file per legacy algorithm, so repeated legacy reruns do not duplicate rows in `metrics_summary_with_legacy.csv`.
- Adds an automatic verifier script and makes both Slurm jobs fail if the expected TWC outputs are missing.
- Clears only the old `twc_pipeline` folder at job start, so the rerun is clean but legacy history remains available.

Files in this patch
- src/fr3_twc/pipeline.py
- src/fr3_twc/unfolded.py
- scripts/twc_verify_outputs.py
- slurm/twc_smoke.slurm
- slurm/twc_full.slurm

Exactly what to do
1. Extract this zip into the root of your GitHub repo and overwrite the existing files.
2. Commit and push to GitHub.
3. On Narval:
   cd /home/rsadve1/FR3_DeepUnfolding
   git pull
4. Recreate the venv cleanly:
   bash scripts/setup_twc_venv.sh fr3_twc_venv
5. Submit smoke:
   sbatch slurm/twc_smoke.slurm
6. Check status:
   sacct -j <SMOKE_JOBID> --format=JobID,State,Elapsed
7. After smoke is COMPLETED, push these back to GitHub:
   results/twc_smoke_bundle/
   fr3_twc_smoke-<SMOKE_JOBID>.out
8. Submit full:
   sbatch slurm/twc_full.slurm
9. Check status:
   sacct -j <FULL_JOBID> --format=JobID,State,Elapsed
10. After full is COMPLETED, push these back to GitHub:
   results/twc_full_bundle/
   fr3_twc_full-<FULL_JOBID>.out

What should exist after a successful run
- results/twc_smoke_bundle/twc_pipeline/training_history.csv
- results/twc_smoke_bundle/twc_pipeline/metrics_summary.csv
- results/twc_smoke_bundle/twc_pipeline/metrics_summary_with_legacy.csv
- results/twc_smoke_bundle/twc_pipeline/metrics_per_sample.csv
- results/twc_smoke_bundle/twc_pipeline/tone_grouping_error.csv
- results/twc_smoke_bundle/twc_pipeline/user_weight_sensitivity.csv
- results/twc_smoke_bundle/twc_pipeline/algorithm_history.csv
- results/twc_smoke_bundle/twc_pipeline/figures/
- The same set under results/twc_full_bundle/twc_pipeline/

What to send me next
- The updated GitHub contents.
- The two new Slurm log files.
- The full smoke and full result folders.
