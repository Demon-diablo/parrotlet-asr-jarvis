"""Benchmark utilities.

Kept separate from the production inference path so the measurement harness
never pollutes inference behaviour. Phase 1 provides the scaffolding:
timing, peak-memory reporting, warm-up, and a reproducible results table
that is written to JSON.

Phase 3 will extend the harness to run the real decoder with controlled
inputs and to populate the metrics table from #40/#41 of the spec.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("parrotlet.benchmark")


@dataclass
class Timing:
    """Wall-clock duration of one measured phase."""

    phase: str
    seconds: float
    tokens: Optional[int] = None


@dataclass
class RunRecord:
    """A single warm-inference measurement."""

    index: int
    timings: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregate result returned by :func:`run_benchmark`."""

    configuration: Dict[str, Any]
    warm_up_count: int
    measurement_count: int
    load_seconds: Optional[float]
    cold_seconds: Optional[float]
    peak_memory_gb: Optional[float]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised via test stubs
        log.exception("Benchmark step raised: %s", exc)
        raise


def _record_timings(run_index: int, labels: List[str], durations: List[float]) -> RunRecord:
    record = RunRecord(index=run_index)
    for label, dur in zip(labels, durations):
        record.timings.append({"phase": label, "seconds": round(dur, 4)})
    return record


def run_benchmark(
    configuration: Dict[str, Any],
    load_fn: Callable[[], Any],
    prepare_fn: Optional[Callable[[], Any]] = None,
    run_fn: Callable[[Any], Any] = None,
    *,
    warm_up_count: int = 1,
    measurement_count: int = 3,
    input_payload: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """Generic benchmark harness.

    Parameters
    ----------
    configuration: dict recorded verbatim for reproducibility.
    load_fn: called once to materialise and cache the model. Returns a bundle.
    prepare_fn: optional hot-path setup (e.g. tokenizer pre-warming) per run.
    run_fn: called with the bundle for each measured inference. Returns output.
    """
    # ---- cold load ------------------------------------------------------- #
    cold_start = time.perf_counter()
    bundle = _safe_call(load_fn)
    load_seconds = time.perf_counter() - cold_start
    cold_seconds = load_seconds

    peak_memory_gb = None
    try:
        from .gpu import peak_memory_gb as _peak  # noqa: WPS433

        peak_memory_gb = _peak(0)
    except Exception:
        pass

    result = BenchmarkResult(
        configuration=configuration,
        warm_up_count=warm_up_count,
        measurement_count=measurement_count,
        load_seconds=round(load_seconds, 4),
        cold_seconds=round(cold_seconds, 4),
        peak_memory_gb=peak_memory_gb,
    )

    # ---- warm-up --------------------------------------------------------- #
    for _ in range(warm_up_count):
        if prepare_fn is not None:
            _safe_call(prepare_fn)
        _safe_call(run_fn, bundle)

    # ---- measured runs --------------------------------------------------- #
    durations: List[float] = []
    for i in range(measurement_count):
        if prepare_fn is not None:
            _safe_call(prepare_fn)
        t0 = time.perf_counter()
        output = _safe_call(run_fn, bundle)
        dur = time.perf_counter() - t0
        durations.append(dur)

        record = RunRecord(index=i)
        record.timings.append({"phase": "inference", "seconds": round(dur, 4)})
        if isinstance(output, dict):
            record.timings.append(
                {"phase": "output_length", "value": len(json.dumps(output))}
            )
        result.runs.append(asdict(record))

    # ---- summary --------------------------------------------------------- #
    if durations:
        result.summary = {
            "mean_seconds": round(statistics.mean(durations), 4),
            "median_seconds": round(statistics.median(durations), 4),
            "stdev_seconds": round(statistics.pstdev(durations), 4)
            if len(durations) > 1
            else 0.0,
            "min_seconds": round(min(durations), 4),
            "max_seconds": round(max(durations), 4),
        }

    return result


def render_results_table(results: List[BenchmarkResult]) -> str:
    """Render one row per result into a markdown table (per spec #43)."""
    header = (
        "| Configuration | GPU Configuration | Load Time | Peak VRAM | "
        "Warm Latency | Tokens/s | Correct |"
    )
    sep = "|---|---|---:|---:|---:|---:|:---:|"
    lines = [header, sep]
    for r in results:
        cfg = r.configuration.get("label", "baseline")
        gpu = r.configuration.get("gpu", "n/a")
        load_t = f"{r.load_seconds:.3f}s" if r.load_seconds else "-"
        peak = f"{r.peak_memory_gb:.2f}GB" if r.peak_memory_gb else "-"
        warm = r.summary.get("mean_seconds")
        warm_t = f"{warm:.4f}s" if warm else "-"
        tok_s = "-"  # populated in Phase 3 once token counts are known
        correct = r.configuration.get("output_correct", "?")
        lines.append(f"| {cfg} | {gpu} | {load_t} | {peak} | {warm_t} | {tok_s} | {correct} |")
    return "\n".join(lines)


def export_json(result: BenchmarkResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2, sort_keys=True)
    log.info("Exported benchmark result to %s", path)