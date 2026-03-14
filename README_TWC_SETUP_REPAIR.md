TWC setup repair
================

Use this only if `bash scripts/setup_twc_venv.sh fr3_twc_venv` reports that `src/fr3_sim` is missing.

Recommended commands:

```bash
cd /home/rsadve1/FR3_DeepUnfolding
bash scripts/twc_repair_repo.sh
rm -rf fr3_twc_venv
bash scripts/setup_twc_venv.sh fr3_twc_venv
```

If repair still fails, capture and share:

```bash
git status --short
git branch --show-current
git rev-parse --short HEAD
git ls-tree -d --name-only HEAD src
git ls-tree -d --name-only origin/main src
git sparse-checkout list || true
python scripts/twc_probe_env.py --strict || true
```
