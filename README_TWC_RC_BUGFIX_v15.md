# TWC RC bug-fix v15

This patch fixes the grouped feasible recovery crash in the RC track.

## Root cause
`src/fr3_twc/algorithms.py` used `tf.ceil(...)` inside `_equal_count_group_masks(...)`.
On the TensorFlow build used by the Narval jobs, `tf.ceil` is unavailable, which caused both:
- `fr3_twc_rc_smoke-57921837.out`
- `fr3_twc_rc_full-57921838.out`

to fail before `metrics_summary.csv` and the rest of the RC outputs were generated.

## Fix
Use the supported TensorFlow API:
- `tf.math.ceil(...)`
- `tf.math.floor(...)`

No config, Slurm, or pipeline changes are required.
