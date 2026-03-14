#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DEFAULT_COMMIT="${1:-ae46934}"

find_commit_with_path() {
  local rel_path="$1"
  local sha

  if git -C "${REPO_ROOT}" cat-file -e "${DEFAULT_COMMIT}:${rel_path}" 2>/dev/null; then
    echo "${DEFAULT_COMMIT}"
    return 0
  fi

  while read -r sha; do
    if git -C "${REPO_ROOT}" cat-file -e "${sha}:${rel_path}" 2>/dev/null; then
      echo "${sha}"
      return 0
    fi
  done < <(git -C "${REPO_ROOT}" rev-list --all -- "${rel_path}" 2>/dev/null || true)

  return 1
}

restore_path() {
  local rel_path="$1"
  local sha=""

  if [[ -e "${REPO_ROOT}/${rel_path}" ]]; then
    echo "Already present   : ${rel_path}"
    return 0
  fi

  sha="$(find_commit_with_path "${rel_path}" || true)"
  if [[ -z "${sha}" ]]; then
    echo "ERROR: could not find any commit containing ${rel_path}" >&2
    return 1
  fi

  echo "Restoring         : ${rel_path} from ${sha}"
  git -C "${REPO_ROOT}" restore --source="${sha}" --worktree --staged "${rel_path}" 2>/dev/null || \
    git -C "${REPO_ROOT}" checkout "${sha}" -- "${rel_path}"
}

cd "${REPO_ROOT}"

echo "Repo root         : ${REPO_ROOT}"
echo "Current branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "Current HEAD      : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "Fetching history ..."
git fetch --all --tags --prune || true
if [[ "$(git rev-parse --is-shallow-repository 2>/dev/null || echo false)" == "true" ]]; then
  git fetch --unshallow || git fetch --depth=50 origin main || true
fi

restore_path "src/fr3_sim"
restore_path "data/ised_sms"

echo
echo "Verification:"
[[ -d "${REPO_ROOT}/src/fr3_sim" ]] && echo "  OK src/fr3_sim"
[[ -d "${REPO_ROOT}/data/ised_sms" ]] && echo "  OK data/ised_sms"

echo
git status --short src/fr3_sim data/ised_sms || true
