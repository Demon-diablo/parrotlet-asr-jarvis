"""Runtime GPU discovery and diagnostics.

This module is intentionally read-only and side-effect free. It must never be
the place where device assumptions are baked in. The application asks the
runtime what is available; it does not declare what should be available.

Never use values returned here as hard-coded production assumptions. They are
for diagnostics, logs, and post-hoc placement validation only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch import should normally succeed
    torch = None  # type: ignore


def _bytes_to_gb(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / (1024 ** 3), 2)


def gpu_info() -> Dict[str, Any]:
    """Return a snapshot of the GPU environment.

    Structure::

        {
            "torch_version": str | None,
            "cuda_available": bool,
            "cuda_version": str | None,
            "gpu_count": int,
            "gpus": [
                {
                    "index": int,
                    "name": str,
                    "total_memory_gb": float,
                    "allocated_memory_gb": float | None,
                    "reserved_memory_gb": float | None,
                }
            ],
        }

    All values are best-effort. Missing fields are reported as ``None`` so
    downstream diagnostics stay JSON-serialisable.
    """
    info: Dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None) if torch else None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "tf32_supported": False,
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "float32_matmul_precision": None,
        "gpus": [],
    }

    if torch is None:
        return info

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    info["cuda_available"] = cuda_available

    if not cuda_available:
        return info

    try:
        info["cuda_version"] = torch.version.cuda
    except Exception:
        info["cuda_version"] = None

    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            info["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        info["allow_tf32_matmul"] = None

    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            info["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        info["allow_tf32_cudnn"] = None

    try:
        if hasattr(torch, "get_float32_matmul_precision"):
            info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        info["float32_matmul_precision"] = None

    try:
        gpu_count = int(torch.cuda.device_count())
    except Exception:
        gpu_count = 0
    info["gpu_count"] = gpu_count

    gpus: List[Dict[str, Any]] = []
    for index in range(gpu_count):
        entry: Dict[str, Any] = {
            "index": index,
            "name": None,
            "compute_capability": None,
            "tf32_supported": False,
            "total_memory_gb": None,
            "allocated_memory_gb": None,
            "reserved_memory_gb": None,
        }
        try:
            props = torch.cuda.get_device_properties(index)
            entry["name"] = getattr(props, "name", None)
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            if major is not None and minor is not None:
                entry["compute_capability"] = (int(major), int(minor))
                entry["tf32_supported"] = (int(major) >= 8)
            entry["total_memory_gb"] = _bytes_to_gb(
                getattr(props, "total_memory", None)
            )
        except Exception:
            pass
        try:
            entry["allocated_memory_gb"] = _bytes_to_gb(
                torch.cuda.memory_allocated(index)
            )
        except Exception:
            pass
        try:
            entry["reserved_memory_gb"] = _bytes_to_gb(
                torch.cuda.memory_reserved(index)
            )
        except Exception:
            pass
        gpus.append(entry)

    info["gpus"] = gpus
    info["tf32_supported"] = any(g.get("tf32_supported") for g in gpus) if gpus else False
    return info


def configure_tf32(enabled: bool = True, precision: str = "high") -> Dict[str, Any]:
    """Configure TensorFloat-32 (TF32) execution precision on CUDA.

    Safe to call in any environment (including CPU-only or test mocks).
    When CUDA is available, configures:
      - torch.backends.cuda.matmul.allow_tf32
      - torch.backends.cudnn.allow_tf32
      - torch.set_float32_matmul_precision(precision)

    Returns a status dictionary reporting the configured state.
    """
    status: Dict[str, Any] = {
        "configured": False,
        "cuda_available": False,
        "tf32_supported": False,
        "allow_tf32_matmul": None,
        "allow_tf32_cudnn": None,
        "float32_matmul_precision": None,
    }
    if torch is None:
        return status

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    status["cuda_available"] = cuda_available

    if not cuda_available:
        return status

    try:
        device_count = int(torch.cuda.device_count())
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            if getattr(props, "major", 0) >= 8:
                status["tf32_supported"] = True
                break
    except Exception:
        pass

    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            eff_precision = precision if enabled else "highest"
            torch.set_float32_matmul_precision(eff_precision)
            status["float32_matmul_precision"] = eff_precision
        elif hasattr(torch, "get_float32_matmul_precision"):
            status["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        pass

    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
            status["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass

    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = bool(enabled)
            status["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass

    status["configured"] = True
    return status


def peak_memory_gb(device_index: int = 0) -> Optional[float]:
    """Return peak allocated memory in GB for a given GPU, if available."""
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        peak = torch.cuda.max_memory_allocated(device_index)
    except Exception:
        return None
    return _bytes_to_gb(peak)


def reset_peak_memory(device_index: int = 0) -> None:
    """Reset peak memory tracking. Safe to call when CUDA is unavailable."""
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception:
        # Diagnostic helper; never raise out of a diagnostic helper.
        pass