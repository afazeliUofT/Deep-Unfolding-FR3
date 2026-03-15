#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_NAME="${1:-fr3_twc_venv}"
VENV_PATH="${REPO_ROOT}/${VENV_NAME}"

note() { echo "$*"; }
warn() { echo "$*" >&2; }

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

git_ok() {
  git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1
}

git_ref_exists() {
  local ref="$1"
  git -C "${REPO_ROOT}" rev-parse --verify --quiet "${ref}" >/dev/null 2>&1
}

git_tree_has_path() {
  local ref="$1"
  local rel_path="$2"
  git -C "${REPO_ROOT}" cat-file -e "${ref}:${rel_path}" >/dev/null 2>&1
}

disable_sparse_checkout_if_any() {
  if ! git_ok; then
    return 0
  fi
  local sparse="false"
  if git -C "${REPO_ROOT}" config --bool core.sparseCheckout 2>/dev/null | grep -qi '^true$'; then
    sparse="true"
  fi
  if git -C "${REPO_ROOT}" sparse-checkout list >/dev/null 2>&1; then
    sparse="true"
  fi
  if [[ "${sparse}" == "true" ]]; then
    note "Sparse checkout detected; disabling it so legacy source paths can be restored ..."
    git -C "${REPO_ROOT}" sparse-checkout disable >/dev/null 2>&1 || true
    git -C "${REPO_ROOT}" config --unset core.sparseCheckout >/dev/null 2>&1 || true
  fi
}

try_restore_from_ref() {
  local ref="$1"
  local rel_path="$2"
  git -C "${REPO_ROOT}" update-index --no-skip-worktree -- "${rel_path}" >/dev/null 2>&1 || true
  git -C "${REPO_ROOT}" restore --source="${ref}" --worktree --staged "${rel_path}" >/dev/null 2>&1 || \
    git -C "${REPO_ROOT}" checkout "${ref}" -- "${rel_path}" >/dev/null 2>&1 || \
    return 1
  [[ -e "${REPO_ROOT}/${rel_path}" ]]
}

ensure_remote_ref() {
  if ! git_ok; then
    return 1
  fi

  if git_ref_exists "refs/remotes/origin/main"; then
    echo "origin/main"
    return 0
  fi
  if git_ref_exists "refs/remotes/origin/master"; then
    echo "origin/master"
    return 0
  fi

  note "Fetching remote refs from origin ..."
  git -C "${REPO_ROOT}" fetch origin main >/dev/null 2>&1 || true
  git -C "${REPO_ROOT}" fetch origin master >/dev/null 2>&1 || true

  if git_ref_exists "refs/remotes/origin/main"; then
    echo "origin/main"
    return 0
  fi
  if git_ref_exists "refs/remotes/origin/master"; then
    echo "origin/master"
    return 0
  fi
  return 1
}

restore_from_git_if_missing() {
  local rel_path="$1"
  [[ -e "${REPO_ROOT}/${rel_path}" ]] && return 0
  git_ok || return 0

  disable_sparse_checkout_if_any

  if git_tree_has_path HEAD "${rel_path}"; then
    note "Missing ${rel_path}; restoring from local HEAD ..."
    try_restore_from_ref HEAD "${rel_path}" && return 0
  fi

  local remote_ref=""
  if remote_ref="$(ensure_remote_ref)"; then
    if git_tree_has_path "${remote_ref}" "${rel_path}"; then
      note "Missing ${rel_path}; restoring from ${remote_ref} ..."
      try_restore_from_ref "${remote_ref}" "${rel_path}" && return 0
    fi
  fi

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

print_git_diagnostics() {
  warn "---- git diagnostics ----"
  git -C "${REPO_ROOT}" status --short --branch 2>/dev/null >&2 || true
  git -C "${REPO_ROOT}" branch -vv 2>/dev/null >&2 || true
  git -C "${REPO_ROOT}" remote -v 2>/dev/null >&2 || true
  git -C "${REPO_ROOT}" ls-tree -d --name-only HEAD -- src 2>/dev/null >&2 || true
  git -C "${REPO_ROOT}" ls-tree -d --name-only HEAD -- src/fr3_sim 2>/dev/null >&2 || true
  git -C "${REPO_ROOT}" ls-tree -d --name-only origin/main -- src/fr3_sim 2>/dev/null >&2 || true
  warn "-------------------------"
}

cd "${REPO_ROOT}"

restore_from_git_if_missing "src/fr3_sim" || true
restore_from_git_if_missing "src/fr3_twc" || true
restore_from_git_if_missing "configs/twc_full.yaml" || true
restore_from_git_if_missing "scripts/twc_probe_env.py" || true

if [[ ! -d "${REPO_ROOT}/src/fr3_sim" ]]; then
  warn "ERROR: ${REPO_ROOT}/src/fr3_sim is still missing."
  warn "Run this manual repair once, then rerun setup:"
  warn "  bash scripts/twc_repair_repo.sh"
  print_git_diagnostics
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/src/fr3_twc" ]]; then
  warn "ERROR: ${REPO_ROOT}/src/fr3_twc is missing."
  warn "Re-apply the TWC package files into the repo and rerun setup."
  print_git_diagnostics
  exit 1
fi

load_python_module_if_needed
PYBIN="$(pick_python)"
if [[ -z "${PYBIN}" ]]; then
  warn "No suitable python executable found after module load attempt."
  exit 1
fi

note "Repo root        : ${REPO_ROOT}"
note "Python selected  : ${PYBIN}"

if [[ -d "${VENV_PATH}" ]]; then
  note "Removing existing venv at ${VENV_PATH}"
  rm -rf "${VENV_PATH}"
fi

"${PYBIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_twc.txt
write_repo_pth

if [[ -f pyproject.toml ]]; then
  python -m pip install -e . || warn "WARNING: editable install failed; continuing because repo paths were pinned via .pth"
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
python scripts/twc_probe_env.py --strict
note "Environment ready at ${VENV_PATH}"
