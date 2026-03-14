This v2 patch fixes the setup/import path issue reported during `bash scripts/setup_twc_venv.sh fr3_twc_venv`.

Main changes:
- setup script now restores `src/fr3_sim` from git if it is missing locally
- setup script writes a persistent `.pth` file into the venv so `repo_root` and `repo_root/src` are always importable
- probe script now prints the detected repo/src layout and the actual module specs
- run scripts insert both `REPO_ROOT` and `REPO_ROOT/src` into `sys.path`
- pyproject package discovery is now explicit
