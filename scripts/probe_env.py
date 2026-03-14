"""Environment probe for Jupyter / local execution.

Run this once in your terminal (or copy/paste into a notebook cell) to confirm:
- Python version
- TensorFlow version
- Sionna version
- Available GPUs and whether TF can see them

Example:
  python scripts/probe_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print("# Environment probe")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())

    try:
        import tensorflow as tf

        print("TensorFlow:", tf.__version__)
        gpus = tf.config.list_physical_devices("GPU")
        print("TF GPUs:", gpus)
        if gpus:
            for g in gpus:
                details = tf.config.experimental.get_device_details(g)
                print("  -", g.name, details)
    except Exception as e:
        print("TensorFlow import failed:", repr(e))

    try:
        import sionna

        ver = getattr(sionna, "__version__", "(unknown)")
        print("Sionna:", ver)
    except Exception as e:
        print("Sionna import failed:", repr(e))


if __name__ == "__main__":
    main()
