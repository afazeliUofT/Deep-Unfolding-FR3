#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${REPO_ROOT}"

echo "Repo root: ${REPO_ROOT}"

git rev-parse --git-dir >/dev/null 2>&1 || { echo "Not a git repository: ${REPO_ROOT}" >&2; exit 1; }

if git config --bool core.sparseCheckout 2>/dev/null | grep -qi '^true$' || git sparse-checkout list >/dev/null 2>&1; then
  echo "Disabling sparse checkout ..."
  git sparse-checkout disable >/dev/null 2>&1 || true
  git config --unset core.sparseCheckout >/dev/null 2>&1 || true
fi

echo "Fetching latest main from origin ..."
git fetch origin main

echo "Restoring legacy source tree from origin/main ..."
git update-index --no-skip-worktree -- src/fr3_sim >/dev/null 2>&1 || true
git restore --source=origin/main --worktree --staged src/fr3_sim 2>/dev/null || git checkout origin/main -- src/fr3_sim

echo "Restoring TWC files from origin/main when available ..."
git restore --source=origin/main --worktree --staged src/fr3_twc scripts configs pyproject.toml requirements_twc.txt README_TWC_UPGRADE.md 2>/dev/null || true

if [[ ! -d src/fr3_sim ]]; then
  echo "ERROR: src/fr3_sim is still missing after repair." >&2
  echo "Diagnostics:" >&2
  git status --short --branch >&2 || true
  git branch -vv >&2 || true
  git remote -v >&2 || true
  git ls-tree -d --name-only HEAD -- src/fr3_sim >&2 || true
  git ls-tree -d --name-only origin/main -- src/fr3_sim >&2 || true
  exit 1
fi

echo "Repair successful. Found: src/fr3_sim"
find src -maxdepth 2 -type d | sort | sed -n '1,40p'
