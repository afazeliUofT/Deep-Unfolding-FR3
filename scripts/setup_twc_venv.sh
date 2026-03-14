#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_NAME="${1:-fr3_twc_venv}"
VENV_PATH="${REPO_ROOT}/${VENV_NAME}"

load_python_module_if_needed() {
  if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    return 0
  fi
  if command -v module >/dev/null 2>&1; then
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python/3.12 2>/dev/null || true
  fi
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  for py in python3.11 python3.10 python3.12 python3; do
    if command -v "${py}" >/dev/null 2>&1; then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

cd "${REPO_ROOT}"

load_python_module_if_needed
PYBIN="$(pick_python)"
if [[ -z "${PYBIN}" ]]; then
  echo "No suitable python executable found after module load attempt." >&2
  exit 1
fi

"${PYBIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_twc.txt
python -m pip install -e .

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
python scripts/twc_probe_env.py --strict
echo "Environment ready at ${VENV_PATH}"
