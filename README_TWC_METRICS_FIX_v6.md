# TWC metrics dtype fix v6

## Files in this patch
- `src/fr3_twc/metrics.py`
- `src/fr3_twc/pipeline.py`

## What this fixes
1. Fixes the crash in `tone_grouping_error()` caused by dividing `tf.complex64` by `tf.float32`.
2. Removes the current `pd.concat` future-warning by skipping empty history frames before concatenation.

## Exact steps
1. Extract this zip into the **root** of the GitHub repo and overwrite existing files.
2. Commit and push:
   ```bash
   git add src/fr3_twc/metrics.py src/fr3_twc/pipeline.py README_TWC_METRICS_FIX_v6.md
   git commit -m "Fix TWC tone-grouping dtype bug and clean history concat"
   git push
   ```
3. On Narval:
   ```bash
   cd /home/rsadve1/FR3_DeepUnfolding
   git pull
   sbatch slurm/twc_smoke.slurm
   ```
4. After smoke finishes, push back:
   ```bash
   git add results/twc_smoke_bundle fr3_twc_smoke-*.out
   git commit -m "Add completed TWC smoke outputs after metrics fix"
   git push
   ```
5. Then run full:
   ```bash
   cd /home/rsadve1/FR3_DeepUnfolding
   sbatch slurm/twc_full.slurm
   ```
6. After full finishes, push back:
   ```bash
   git add results/twc_full_bundle fr3_twc_full-*.out
   git commit -m "Add completed TWC full outputs after metrics fix"
   git push
   ```

## Success condition
After rerun, each `results/twc_*_bundle/twc_pipeline/` must contain:
- `run_metadata.json`
- `metrics_summary.csv`
- `metrics_summary_with_legacy.csv`
- `metrics_per_sample.csv`
- `tone_grouping_error.csv`
- `user_weight_sensitivity.csv`
- `algorithm_history.csv`
- `training_history.csv`
- `figures/`

If smoke or full still fails, send back the new `.out` file and the complete `results/twc_*_bundle/twc_pipeline/` folder.
