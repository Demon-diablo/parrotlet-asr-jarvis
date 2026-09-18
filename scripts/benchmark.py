"""Benchmark runner using the generic harness in src/benchmark.py.

Phase 1 (skeleton) runs the dummy echo inference under the harness so the
timing/summary machinery is verified. Phase 3 will swap the real loader and
decoder call in.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from src.benchmark import (  # noqa: E402
    BenchmarkResult,
    export_json,
    render_results_table,
    run_benchmark,
)
from src.config import get_settings  # noqa: E402
from src.gpu import gpu_info  # noqa: E402
from src.model import load_model  # noqa: E402
from src.inference import run_inference  # noqa: E402


def main() -> int:
    settings = get_settings()
    runtime = gpu_info()

    configuration = {
        "label": "baseline-skeleton",
        "gpu": runtime.get("gpus", [{}])[0].get("name", "n/a") if runtime["gpus"] else "cpu",
        "settings": settings.as_dict(),
        "runtime": runtime,
    }

    fixed_input = {"message": "hello from benchmark"}

    def load_fn():
        return load_model()

    def run_fn(bundle):
        return run_inference(bundle, fixed_input)

    result: BenchmarkResult = run_benchmark(
        configuration=configuration,
        load_fn=load_fn,
        run_fn=run_fn,
        warm_up_count=2,
        measurement_count=3,
        input_payload=fixed_input,
    )

    out = {
        "result": result.to_dict(),
        "table": render_results_table([result]),
    }
    print(json.dumps(out, indent=2, default=str))

    output_path = os.environ.get("BENCHMARK_OUTPUT", "benchmark_result.json")
    export_json(result, output_path)
    print(f"benchmark result written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())