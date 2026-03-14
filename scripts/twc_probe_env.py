#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pkg_ver(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    failures = []

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Repo root         : {REPO_ROOT}")
    print(f"tensorflow        : {pkg_ver('tensorflow')}")
    print(f"sionna-no-rt      : {pkg_ver('sionna-no-rt')}")
    print(f"numpy             : {pkg_ver('numpy')}")
    print(f"pandas            : {pkg_ver('pandas')}")

    try:
        import tensorflow as tf
        print(f"TF version        : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"Visible GPUs      : {len(gpus)}")
        for idx, gpu in enumerate(gpus):
            print(f"  GPU[{idx}]        : {gpu}")
    except Exception as exc:
        failures.append(f"TensorFlow import failed: {exc}")

    try:
        import fr3_sim  # noqa: F401
        print("fr3_sim import    : OK")
    except Exception as exc:
        failures.append(f"fr3_sim import failed: {exc}")

    try:
        import fr3_twc  # noqa: F401
        print("fr3_twc import    : OK")
    except Exception as exc:
        failures.append(f"fr3_twc import failed: {exc}")

    try:
        import sionna  # type: ignore  # noqa: F401
        print("sionna import     : OK")
    except Exception as exc:
        failures.append(f"sionna import failed: {exc}")

    if failures:
        print("\nProbe failures:")
        for item in failures:
            print(" - " + item)
        return 1 if args.strict else 0

    print("\nProbe passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
