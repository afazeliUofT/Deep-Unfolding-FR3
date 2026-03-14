#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_NAME="${1:-fr3_twc_venv}"
VENV_PATH="${REPO_ROOT}/${VENV_NAME}"

restore_from_git_if_missing() {
  local rel_path="$1"
  if [[ -e "${REPO_ROOT}/${rel_path}" ]]; then
    return 0
  fi
  if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Missing ${rel_path}; attempting restore from git ..."
    git -C "${REPO_ROOT}" restore --source=HEAD --worktree --staged "${rel_path}" 2>/dev/null || \
      git -C "${REPO_ROOT}" checkout HEAD -- "${rel_path}" 2>/dev/null || true
  fi
}

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
  for py in python3.11 python3.10 python3.12 python3 python; do
    if command -v "${py}" >/dev/null 2>&1; then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

write_repo_pth() {
  export FR3_REPO_ROOT="${REPO_ROOT}"
  python - <<'PY'
from pathlib import Path
import os
import site
import sysconfig

repo_root = Path(os.environ["FR3_REPO_ROOT"]).resolve()
paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass
try:
    user_site = site.getusersitepackages()
    if user_site:
        paths.append(user_site)
except Exception:
    pass
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    paths.append(purelib)

site_pkgs = None
for p in paths:
    if p and Path(p).exists():
        site_pkgs = Path(p)
        break

if site_pkgs is None:
    raise SystemExit("Could not locate site-packages to write a .pth file")

pth_path = site_pkgs / "fr3_repo_src.pth"
pth_path.write_text(f"{repo_root}\n{repo_root / 'src'}\n", encoding="utf-8")
print(f"Wrote path file  : {pth_path}")
PY
}

cd "${REPO_ROOT}"

restore_from_git_if_missing "src/fr3_sim"
restore_from_git_if_missing "src/fr3_twc"
restore_from_git_if_missing "configs/twc_full.yaml"
restore_from_git_if_missing "scripts/twc_probe_env.py"

if [[ ! -d "${REPO_ROOT}/src/fr3_sim" ]]; then
  echo "ERROR: ${REPO_ROOT}/src/fr3_sim is missing." >&2
  echo "The TWC package depends on the original legacy source tree." >&2
  echo "Run this once, then rerun setup:" >&2
  echo "  git restore src/fr3_sim || git checkout HEAD -- src/fr3_sim" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/src/fr3_twc" ]]; then
  echo "ERROR: ${REPO_ROOT}/src/fr3_twc is missing." >&2
  echo "Re-apply the v2 package files into the repo and rerun setup." >&2
  exit 1
fi

load_python_module_if_needed
PYBIN="$(pick_python)"
if [[ -z "${PYBIN}" ]]; then
  echo "No suitable python executable found after module load attempt." >&2
  exit 1
fi

echo "Repo root        : ${REPO_ROOT}"
echo "Python selected  : ${PYBIN}"

if [[ -d "${VENV_PATH}" ]]; then
  echo "Removing existing venv at ${VENV_PATH}"
  rm -rf "${VENV_PATH}"
fi

"${PYBIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_twc.txt
write_repo_pth

# Editable install is useful, but the .pth file above is the real import-path guarantee.
if [[ -f pyproject.toml ]]; then
  python -m pip install -e . || echo "WARNING: editable install failed; continuing because repo paths were pinned via .pth" >&2
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
python scripts/twc_probe_env.py --strict
echo "Environment ready at ${VENV_PATH}"
