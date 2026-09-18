"""Diagnostic: report runtime GPU info.

Usage::

    python scripts/check_gpu.py

Prints a JSON-serialisable snapshot of the CUDA environment discovered at
runtime. Exits non-zero if the reported structure looks malformed.
"""

from __future__ import annotations

import json
import sys

# Allow `python scripts/check_gpu.py` to find the ``src`` package without an
# install (runs from the repo root).
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu import gpu_info  # noqa: E402


def main() -> int:
    info = gpu_info()
    print(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())