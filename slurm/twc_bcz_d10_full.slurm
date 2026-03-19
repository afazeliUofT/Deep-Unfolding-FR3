#!/bin/bash
#SBATCH --job-name=fr3_twc_bcz_d10_full
#SBATCH --account=def-rsadve
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT=/home/rsadve1/FR3_DeepUnfolding
VENV_PATH=${REPO_ROOT}/fr3_twc_venv
RESULT_ROOT=/home/rsadve1/FR3_DeepUnfolding/results/twc_bcz_d10_full_bundle

cd ${REPO_ROOT}

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.12 2>/dev/null || true
fi

source ${VENV_PATH}/bin/activate
export PYTHONPATH=${REPO_ROOT}/src:${PYTHONPATH:-}

rm -rf ${RESULT_ROOT}/twc_pipeline

python scripts/twc_run_override_workflow.py --override configs/twc_bcz_d10_full.yaml
