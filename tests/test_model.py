"""Tests for src/model.py loader plumbing.

These tests verify loader behavior without loading any real weights. They
cover:

- Settings / device resolution logic (DEVICE_MAP_MODE -> "cuda"/"cpu")
- Prerequisite detection (missing torch / transformers -> clean error)
- Cached-load fast path (force_reload, is_loaded, reset_cache)
- Placement validator helper (_submodule_device)
- HF kwarg construction
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.model import (  # noqa: E402
    _build_hf_kwargs,
    _resolve_device,
    _resolve_dtype,
    _submodule_device,
    is_loaded,
    load_model,
    reset_cache,
)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def _runtime(cuda: bool, count: int = 0) -> dict:
    return {
        "cuda_available": cuda,
        "gpu_count": count,
        "gpus": [],
        "torch_version": None,
        "cuda_version": None,
    }


class _Settings:
    def __init__(self, device_map_mode: str, dtype: str = "auto", quantization: str = "none"):
        self.device_map_mode = device_map_mode
        self.dtype = dtype
        self.quantization = quantization
        self.model_source = "ekacare/parrotlet-a-2.5-pro"
        self.model_revision = ""
        self.hf_token = ""
        self.model_cache_dir = ""


def test_resolve_device_explicit_cpu():
    s = _Settings(device_map_mode="cpu")
    assert _resolve_device(s, _runtime(cuda=True)) == "cpu"
    assert _resolve_device(s, _runtime(cuda=False)) == "cpu"


def test_resolve_device_auto_with_cuda():
    s = _Settings(device_map_mode="auto")
    assert _resolve_device(s, _runtime(cuda=True)) == "cuda"


def test_resolve_device_auto_without_cuda_falls_back_to_cpu():
    s = _Settings(device_map_mode="auto")
    assert _resolve_device(s, _runtime(cuda=False)) == "cpu"


def test_resolve_device_explicit_cuda_without_runtime_cuda_falls_back():
    s = _Settings(device_map_mode="cuda")
    assert _resolve_device(s, _runtime(cuda=False)) == "cpu"


def test_resolve_device_is_never_gpu_index_specific():
    """Spec #21: never hard-code cuda:0."""
    s = _Settings(device_map_mode="auto")
    device = _resolve_device(s, _runtime(cuda=True))
    assert device == "cuda"
    assert ":" not in device


# ---------------------------------------------------------------------------
# Dtype resolution
# ---------------------------------------------------------------------------
def test_resolve_dtype_auto_is_none():
    assert _resolve_dtype(_Settings(device_map_mode="auto", dtype="auto")) is None


def test_resolve_dtype_bf16_resolves_to_torch_bfloat16_when_torch_present():
    """When torch is importable, bf16 -> torch.bfloat16."""
    try:
        import torch as _t  # noqa: F401
    except Exception:
        pytest.skip("torch not installed in this venv")
    result = _resolve_dtype(_Settings(device_map_mode="auto", dtype="bf16"))
    import torch
    assert result == torch.bfloat16


def test_resolve_dtype_tf32_resolves_to_torch_float32():
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed in this venv")
    result = _resolve_dtype(_Settings(device_map_mode="auto", dtype="tf32"))
    assert result == torch.float32


def test_resolve_dtype_unknown_returns_none_when_torch_missing():
    """When torch is not importable, _resolve_dtype returns None for all values."""
    with patch("src.model.torch", None):
        # _resolve_dtype reads src.model.torch; the patched None makes it
        # return None for non-auto choices.
        result = _resolve_dtype(_Settings(device_map_mode="auto", dtype="bf16"))
        assert result is None


# ---------------------------------------------------------------------------
# HF kwargs
# ---------------------------------------------------------------------------
def test_build_hf_kwargs_always_includes_trust_remote_code():
    s = _Settings(device_map_mode="auto")
    s.hf_token = ""
    s.model_revision = ""
    s.model_cache_dir = ""
    kw = _build_hf_kwargs(s)
    assert kw["trust_remote_code"] is True
    assert "token" not in kw
    assert "revision" not in kw
    assert "cache_dir" not in kw


def test_build_hf_kwargs_includes_token_revision_cache_when_set():
    s = _Settings(device_map_mode="auto")
    s.hf_token = "hf_test"
    s.model_revision = "abc123"
    s.model_cache_dir = "/mnt/cache"
    kw = _build_hf_kwargs(s)
    assert kw["token"] == "hf_test"
    assert kw["revision"] == "abc123"
    assert kw["cache_dir"] == "/mnt/cache"
    assert kw["trust_remote_code"] is True


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------
def test_is_loaded_starts_false():
    reset_cache()
    assert is_loaded() is False


def test_reset_cache_idempotent():
    reset_cache()
    reset_cache()
    assert is_loaded() is False


# ---------------------------------------------------------------------------
# Prereq detection: load_model must raise a clean error when torch missing
# ---------------------------------------------------------------------------
def test_load_model_raises_clean_error_when_torch_missing(monkeypatch):
    """Spec #50: model init errors must not be hidden. We patch src.model.torch
    to None to simulate a missing torch install and assert a clean RuntimeError
    (not an opaque AttributeError from deep inside transformers)."""
    monkeypatch.setattr("src.model.torch", None)
    reset_cache()
    with pytest.raises(RuntimeError, match="torch is not importable"):
        load_model()
    assert is_loaded() is False


# ---------------------------------------------------------------------------
# Placement validator
# ---------------------------------------------------------------------------
def test_submodule_device_returns_none_for_none():
    assert _submodule_device(None) is None


def test_submodule_device_returns_none_when_torch_missing(monkeypatch):
    monkeypatch.setattr("src.model.torch", None)
    assert _submodule_device(object()) is None


# ---------------------------------------------------------------------------
# FP8 quantization helper tests
# ---------------------------------------------------------------------------
def test_apply_fp8_quantization_skips_when_no_cuda(monkeypatch):
    from src.model import _apply_fp8_quantization
    from src.config import load_settings

    # Simulate no CUDA
    monkeypatch.setattr("src.model.torch", None)
    s = load_settings({"QUANTIZATION": "fp8"})
    res = _apply_fp8_quantization(object(), s, "cpu")
    assert res == {"status": "skipped", "reason": "no_cuda"}


def test_apply_fp8_quantization_raises_when_torchao_missing(monkeypatch):
    from src.model import _apply_fp8_quantization
    from src.config import load_settings
    from unittest.mock import MagicMock

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    monkeypatch.setattr("src.model.torch", mock_torch)

    # Simulate torchao import error by hiding it in sys.modules
    with patch.dict("sys.modules", {"torchao": None, "torchao.quantization": None}):
        s = load_settings({"QUANTIZATION": "fp8_compiled"})
        with pytest.raises(RuntimeError, match="torchao is required"):
            _apply_fp8_quantization(object(), s, "cuda:0")