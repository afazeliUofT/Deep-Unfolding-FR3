# TWC BCZ override path fix v21

## Why this fix is needed
The v20 specialized workflow writes a merged temporary YAML under `results/_resolved_overrides/`.
The inherited BC configs contain dataset paths like `../data/ised_sms/...`.
Because `load_config()` records the YAML parent directory as `config_dir`, the dataset path is then resolved relative to `results/_resolved_overrides/`, which incorrectly points to `results/data/...` and makes all specialized jobs fail with `FileNotFoundError`.

## What this fix changes
- rewrites known ISED CSV path fields to absolute paths based on the **base config directory**
- materializes the merged temporary YAML directly under `configs/` instead of under `results/`
- leaves all specialized BCZ configs/slurms untouched

## Files replaced
- `scripts/twc_run_override_workflow.py`

## Files added
- `README_TWC_BCZ_OVERRIDE_FIX_v21.md`
