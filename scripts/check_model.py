"""Diagnostic: load the Parrotlet model and report placement/memory.

Usage::

    python scripts/check_model.py

Phase 1 (skeleton) confirms the loader wiring and reports the runtime GPU
snapshot plus the configured model source. The real decoder load happens in
Phase 3 after model inspection.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.gpu import gpu_info  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.model import load_model, model_metadata  # noqa: E402


def main() -> int:
    settings = get_settings()
    report = {
        "settings": settings.as_dict(),
        "runtime_gpu": gpu_info(),
    }

    try:
        bundle = load_model()
        report["load_status"] = "ok"
        report["metadata"] = bundle.metadata if bundle else None
    except Exception as exc:  # pragma: no cover - exercised in smoke phase
        report["load_status"] = "error"
        report["error"] = str(exc)
        import traceback

        report["traceback"] = traceback.format_exc()

    if model_metadata() is not None:
        report["cached_metadata"] = model_metadata()

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())