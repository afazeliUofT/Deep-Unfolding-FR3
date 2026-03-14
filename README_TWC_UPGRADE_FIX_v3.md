This fix package updates the setup/repair flow for cases where the Narval clone is missing `src/fr3_sim` even though the GitHub `main` branch contains it.

Changed files:
- scripts/setup_twc_venv.sh
- scripts/twc_repair_repo.sh

What changed:
- setup now tries `HEAD` first, then `origin/main` / `origin/master`
- setup disables sparse checkout if it is active
- added manual repair helper script for one-shot repo repair
