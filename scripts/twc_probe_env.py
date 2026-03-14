#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pkg_ver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def spec_origin(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not-found"
    origin = getattr(spec, "origin", None)
    subloc = getattr(spec, "submodule_search_locations", None)
    if origin:
        return str(origin)
    if subloc:
        return ", ".join(str(x) for x in subloc)
    return "found"


def src_children() -> str:
    if not SRC.exists():
        return "<missing src directory>"
    names = sorted(p.name for p in SRC.iterdir() if p.is_dir())
    return ", ".join(names) if names else "<no subdirectories>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    failures: list[str] = []

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Repo root         : {REPO_ROOT}")
    print(f"Repo root exists  : {REPO_ROOT.exists()}")
    print(f"Src dir           : {SRC}")
    print(f"Src dir exists    : {SRC.exists()}")
    print(f"Src entries       : {src_children()}")
    print(f"sys.path[0:4]     : {sys.path[:4]}")
    print(f"tensorflow        : {pkg_ver('tensorflow')}")
    print(f"sionna-no-rt      : {pkg_ver('sionna-no-rt')}")
    print(f"numpy             : {pkg_ver('numpy')}")
    print(f"pandas            : {pkg_ver('pandas')}")
    print(f"fr3_sim spec      : {spec_origin('fr3_sim')}")
    print(f"fr3_twc spec      : {spec_origin('fr3_twc')}")

    if not (SRC / "fr3_sim").exists():
        failures.append(
            f"Required directory missing: {SRC / 'fr3_sim'} (restore with: git restore src/fr3_sim || git checkout HEAD -- src/fr3_sim)"
        )
    if not (SRC / "fr3_twc").exists():
        failures.append(f"Required directory missing: {SRC / 'fr3_twc'}")

    try:
        import tensorflow as tf

        print(f"TF version        : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"Visible GPUs      : {len(gpus)}")
        for idx, gpu in enumerate(gpus):
            print(f"  GPU[{idx}]        : {gpu}")
    except Exception as exc:
        failures.append(f"TensorFlow import failed: {exc}")

    for mod_name in ("fr3_sim", "fr3_twc", "sionna"):
        try:
            importlib.import_module(mod_name)
            print(f"{mod_name:<16}: OK")
        except Exception as exc:
            failures.append(f"{mod_name} import failed: {exc}")

    if failures:
        print("\nProbe failures:")
        for item in failures:
            print(" - " + item)
        return 1 if args.strict else 0

    print("\nProbe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
