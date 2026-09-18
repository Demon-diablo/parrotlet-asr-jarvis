#!/usr/bin/env python3
"""Diagnostic: Verify and benchmark TensorFloat-32 (TF32) on RTX 6000 Pro / Ampere+ GPUs.

Usage::

    python scripts/check_tf32.py

Tests:
1. Hardware architecture check: compute capability >= 8.0 (Ampere / Ada Lovelace / Hopper / Blackwell).
2. Configuration & environment: ALLOW_TF32, FLOAT32_MATMUL_PRECISION, and DTYPE.
3. PyTorch runtime status: torch.backends.cuda.matmul.allow_tf32, cudnn.allow_tf32, precision.
4. Numerical & performance benchmark: FP32 vs TF32 GEMM speedup and numerical accuracy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.gpu import configure_tf32, gpu_info

try:
    import torch
except ImportError:
    torch = None


def run_gemm_benchmark(size: int = 4096, iters: int = 10) -> Dict[str, Any]:
    """Benchmark GEMM with TF32 enabled vs disabled."""
    if torch is None or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda:0")
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)

    # 1. Benchmark without TF32 (full FP32 precision)
    configure_tf32(enabled=False, precision="highest")
    torch.cuda.synchronize()
    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c_fp32 = torch.matmul(a, b)
    torch.cuda.synchronize()
    fp32_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # 2. Benchmark with TF32 (TensorFloat-32 Tensor Cores)
    configure_tf32(enabled=True, precision="high")
    torch.cuda.synchronize()
    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c_tf32 = torch.matmul(a, b)
    torch.cuda.synchronize()
    tf32_time_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # 3. Compute numerical difference
    max_diff = torch.max(torch.abs(c_fp32 - c_tf32)).item()
    mean_diff = torch.mean(torch.abs(c_fp32 - c_tf32)).item()
    speedup = round(fp32_time_ms / max(tf32_time_ms, 1e-6), 2)

    # TFLOPS calculation: 2 * N^3 operations
    tflops_fp32 = round((2.0 * (size ** 3) * 1e-12) / (fp32_time_ms * 1e-3), 2)
    tflops_tf32 = round((2.0 * (size ** 3) * 1e-12) / (tf32_time_ms * 1e-3), 2)

    return {
        "matrix_dim": f"{size}x{size}",
        "iterations": iters,
        "fp32_latency_ms": round(fp32_time_ms, 3),
        "fp32_tflops": tflops_fp32,
        "tf32_latency_ms": round(tf32_time_ms, 3),
        "tf32_tflops": tflops_tf32,
        "speedup": f"{speedup}x",
        "max_absolute_diff": float(f"{max_diff:.6e}"),
        "mean_absolute_diff": float(f"{mean_diff:.6e}"),
    }


def main() -> int:
    print("=" * 80)
    print("SLEEKCARE TF32 ARCHITECTURE DIAGNOSTIC (RTX 6000 Pro)")
    print("=" * 80)

    settings = get_settings()
    info = gpu_info()

    report: Dict[str, Any] = {
        "settings": {
            "allow_tf32": settings.allow_tf32,
            "float32_matmul_precision": settings.float32_matmul_precision,
            "dtype": settings.dtype,
        },
        "gpu_info": info,
    }

    print("\n1. Configuration Settings:")
    print(f"   ALLOW_TF32: {settings.allow_tf32}")
    print(f"   FLOAT32_MATMUL_PRECISION: {settings.float32_matmul_precision}")
    print(f"   DTYPE: {settings.dtype}")

    print("\n2. GPU Architecture Snapshot:")
    print(f"   CUDA Available: {info.get('cuda_available')}")
    print(f"   CUDA Version: {info.get('cuda_version')}")
    print(f"   GPU Count: {info.get('gpu_count')}")
    print(f"   Hardware TF32 Supported: {info.get('tf32_supported')}")

    for g in info.get("gpus", []):
        name = g.get("name") or "Unknown"
        cap = g.get("compute_capability")
        tf32_cap = g.get("tf32_supported")
        print(f"   GPU #{g['index']}: {name} (Compute Capability: {cap}, TF32: {tf32_cap})")

    # Configure TF32 per settings
    configured = configure_tf32(
        enabled=settings.allow_tf32,
        precision=settings.float32_matmul_precision,
    )
    report["tf32_status_after_config"] = configured

    print("\n3. Active PyTorch Precision State:")
    print(f"   allow_tf32 (matmul): {configured.get('allow_tf32_matmul')}")
    print(f"   allow_tf32 (cudnn): {configured.get('allow_tf32_cudnn')}")
    print(f"   float32_matmul_precision: {configured.get('float32_matmul_precision')}")

    if torch is not None and torch.cuda.is_available() and info.get("tf32_supported"):
        print("\n4. Running FP32 vs TF32 GEMM Benchmark...")
        bench = run_gemm_benchmark(size=2048, iters=5)
        report["benchmark"] = bench
        print(f"   Matrix Shape: {bench['matrix_dim']}")
        print(f"   FP32 (no TF32) Latency: {bench['fp32_latency_ms']} ms ({bench['fp32_tflops']} TFLOPS)")
        print(f"   TF32 Tensor Cores Latency: {bench['tf32_latency_ms']} ms ({bench['tf32_tflops']} TFLOPS)")
        print(f"   TF32 Speedup: {bench['speedup']}")
        print(f"   Numerical Max Diff: {bench['max_absolute_diff']}")
    else:
        print("\n4. Benchmark: Skipped (CUDA/TF32 hardware not active in current environment).")
        report["benchmark"] = "skipped_no_cuda_or_unsupported_hardware"

    print("\nFull Diagnostic JSON:")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
