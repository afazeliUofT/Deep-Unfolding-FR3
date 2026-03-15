This patch must be applied before any new smoke/full run.

Files replaced:
- src/fr3_twc/algorithms.py
- src/fr3_twc/unfolded.py
- src/fr3_twc/pipeline.py
- src/fr3_twc/wideband_channel.py

Why this patch is needed:
1. The live repo is not the same as the most recent patch: `src/fr3_twc/unfolded.py` still casts learnable real parameters (`step`, `damping`) to complex inside the unfolded loop.
2. That causes the repeated TensorFlow warning: `complex64 -> float32` during backprop through real->complex casts.
3. The live repo also still uses full-band BS power in `src/fr3_twc/wideband_channel.py` instead of the simulated-band power budget.
4. The live repo still uses the older FS-budget cloning path in `src/fr3_twc/pipeline.py`, so `risk_score`, `static_gate`, and `scenario_features` are not refreshed consistently for an FS-budget sweep.
5. The live repo still merges legacy watt-domain FS metrics into TWC normalized-excess columns without normalization.

What this patch changes:
- removes the warning-causing real->complex trainable scaling path in `unfolded.py`
- removes the same style of complex scaling inside `algorithms.py/project_per_bs_power`
- restores simulated-band BS power budget usage
- restores self-consistent FS-budget sweep cloning
- restores clean raw/aggregated summary export and legacy normalization

Run order:
1. Apply this patch in repo root.
2. Commit and push.
3. On Narval: `git pull`
4. Delete old TWC smoke pipeline outputs.
5. Re-run smoke.
6. Check the new smoke log only.
7. Only if smoke is clean, run full.
