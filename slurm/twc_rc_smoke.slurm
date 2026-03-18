#!/bin/bash
#SBATCH --job-name=fr3_twc_rc_smoke
#SBATCH --account=def-rsadve
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=/home/rsadve1/FR3_DeepUnfolding
VENV_PATH=${REPO_ROOT}/fr3_twc_venv
RESULT_ROOT=${REPO_ROOT}/results/twc_rc_smoke_bundle

cd ${REPO_ROOT}
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.12 2>/dev/null || true
fi

source ${VENV_PATH}/bin/activate
export PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}

rm -rf ${RESULT_ROOT}/twc_pipeline

python scripts/twc_probe_env.py --strict
python scripts/twc_run_legacy_baseline.py --config configs/twc_rc_smoke.yaml
python scripts/twc_run_pipeline.py --config configs/twc_rc_smoke.yaml
python scripts/twc_plot_figures.py --config configs/twc_rc_smoke.yaml
python scripts/twc_verify_outputs.py --config configs/twc_rc_smoke.yaml --require-legacy
