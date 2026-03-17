# TWC RR Fix v13

## What this fixes
Both latest v12 jobs crashed in the new rate-recovery stage:

- smoke log: `fr3_twc_rr_smoke-57887432.out`
- full log: `fr3_twc_rr_full-57887433.out`

Exact crash path:
- `scripts/twc_run_pipeline.py`
- `src/fr3_twc/pipeline.py` -> `_run_algorithms(...)`
- `src/fr3_twc/algorithms.py` -> `budgeted_primal_dual_pgd_repair_recover(...)`
- `src/fr3_twc/algorithms.py` -> `recover_rate_with_feasible_mask(...)`

Root cause:
`tf.sequence_mask(min_k, maxlen=K)` was created from a scalar `min_k`, so it became a 1-D mask of shape `[K]`, but it was then used as if it were a 2-D per-sample mask over a tensor of shape `[S, K, 2]`. That is why TensorFlow raised:
- smoke: `ValueError: Shapes (1,) and (12,) are incompatible`
- full: `ValueError: Shapes (2,) and (24,) are incompatible`

## Files in this patch
Replace:
- `src/fr3_twc/algorithms.py`

Add:
- `README_TWC_RR_FIX_v13.md`

## Exact steps

1. Extract this zip into the **root** of your GitHub repo and overwrite files.

2. Commit and push:
```bash
git add src/fr3_twc/algorithms.py README_TWC_RR_FIX_v13.md
git commit -m "Fix TWC RR feasible-mask shape bug"
git push
```

3. On Narval:
```bash
cd /home/rsadve1/FR3_DeepUnfolding
git pull
source /home/rsadve1/FR3_DeepUnfolding/fr3_twc_venv/bin/activate
rm -rf results/twc_rr_smoke_bundle/twc_pipeline
rm -rf results/twc_rr_full_bundle/twc_pipeline
```

4. Run both jobs. They are safe to run simultaneously:
```bash
SMOKE=$(sbatch slurm/twc_rr_smoke.slurm | awk '{print $4}')
FULL=$(sbatch slurm/twc_rr_full.slurm | awk '{print $4}')
echo $SMOKE
echo $FULL
```

5. Monitor:
```bash
sacct -j "$SMOKE","$FULL" --format=JobID,State,Elapsed
```

6. After both finish, check only real blockers:
```bash
grep -E "Traceback|ValueError: Shapes .* incompatible|nan|NaN|INF|Inf" "fr3_twc_rr_smoke-${SMOKE}.out"
grep -E "Traceback|ValueError: Shapes .* incompatible|nan|NaN|INF|Inf" "fr3_twc_rr_full-${FULL}.out"
```

7. Verify outputs:
```bash
python scripts/twc_verify_outputs.py --config configs/twc_rr_smoke.yaml --require-legacy
python scripts/twc_verify_outputs.py --config configs/twc_rr_full.yaml --require-legacy
```

8. Push back:
```bash
git add   results/twc_rr_smoke_bundle   results/twc_rr_full_bundle   "fr3_twc_rr_smoke-${SMOKE}.out"   "fr3_twc_rr_full-${FULL}.out"

git commit -m "Add corrected TWC RR smoke/full outputs after feasible-mask fix"
git push
```

## What to send back for analysis
I need these:
- `results/twc_rr_smoke_bundle/twc_pipeline/`
- `results/twc_rr_full_bundle/twc_pipeline/`
- `fr3_twc_rr_smoke-<JOBID>.out`
- `fr3_twc_rr_full-<JOBID>.out`

Only after these reruns complete without `Traceback` will the v12 rate-recovery mechanism have valid metrics.
