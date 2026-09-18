"""Tests for TensorFloat-32 (TF32) architecture support on RTX 6000 Pro."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import Settings, load_settings, _ALLOWED_DTYPE, _ALLOWED_FLOAT32_MATMUL_PRECISION
from src.gpu import configure_tf32, gpu_info
from src.model import _resolve_dtype, load_model, reset_cache


def test_tf32_in_allowed_dtypes():
    assert "tf32" in _ALLOWED_DTYPE


def test_allowed_float32_matmul_precision():
    assert "highest" in _ALLOWED_FLOAT32_MATMUL_PRECISION
    assert "high" in _ALLOWED_FLOAT32_MATMUL_PRECISION
    assert "medium" in _ALLOWED_FLOAT32_MATMUL_PRECISION


def test_tf32_config_defaults():
    settings = load_settings({})
    assert settings.allow_tf32 is True
    assert settings.float32_matmul_precision == "high"


def test_tf32_config_env_overrides():
    env = {
        "ALLOW_TF32": "0",
        "FLOAT32_MATMUL_PRECISION": "highest",
        "DTYPE": "tf32",
    }
    settings = load_settings(env)
    assert settings.allow_tf32 is False
    assert settings.float32_matmul_precision == "highest"
    assert settings.dtype == "tf32"


def test_tf32_config_invalid_precision():
    with pytest.raises(ValueError, match="Invalid FLOAT32_MATMUL_PRECISION"):
        load_settings({"FLOAT32_MATMUL_PRECISION": "invalid_mode"})


def test_resolve_dtype_tf32():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    settings = load_settings({"DTYPE": "tf32"})
    resolved = _resolve_dtype(settings)
    assert resolved == torch.float32


def test_gpu_info_has_tf32_keys():
    info = gpu_info()
    assert "tf32_supported" in info
    assert "allow_tf32_matmul" in info
    assert "allow_tf32_cudnn" in info
    assert "float32_matmul_precision" in info


def test_configure_tf32_safe_on_cpu():
    result = configure_tf32(enabled=True, precision="high")
    assert isinstance(result, dict)
    assert "cuda_available" in result
    assert "tf32_supported" in result


def test_configure_tf32_with_mock_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    
    mock_prop = MagicMock()
    mock_prop.major = 8
    mock_prop.minor = 9  # RTX 6000 Ada / sm_89
    mock_torch.cuda.get_device_properties.return_value = mock_prop

    with patch("src.gpu.torch", mock_torch):
        status = configure_tf32(enabled=True, precision="high")
        assert status["cuda_available"] is True
        assert status["tf32_supported"] is True
        mock_torch.set_float32_matmul_precision.assert_called_with("high")
        assert mock_torch.backends.cuda.matmul.allow_tf32 is True
        assert mock_torch.backends.cudnn.allow_tf32 is True

        # Disable TF32 (highest precision)
        status_off = configure_tf32(enabled=False, precision="high")
        mock_torch.set_float32_matmul_precision.assert_called_with("highest")
        assert mock_torch.backends.cuda.matmul.allow_tf32 is False
        assert mock_torch.backends.cudnn.allow_tf32 is False


def test_load_model_captures_tf32_metadata():
    reset_cache()
    mock_speech_llm = MagicMock()
    mock_speech_llm.sampling_rate = 16000
    mock_speech_llm.encoder.parameters.return_value = iter([MagicMock(device="cpu", dtype="float32")])
    mock_speech_llm.projector.parameters.return_value = iter([MagicMock(device="cpu", dtype="float32")])
    mock_speech_llm.decoder.parameters.return_value = iter([MagicMock(device="cpu", dtype="float32")])

    with patch("src.model._load_speech_llm", return_value=mock_speech_llm):
        with patch.dict(os.environ, {"ALLOW_TF32": "1", "FLOAT32_MATMUL_PRECISION": "high"}):
            bundle = load_model(force_reload=True)
            assert bundle is not None
            assert "tf32" in bundle.metadata
            assert bundle.metadata["tf32"]["enabled"] is True
            assert bundle.metadata["tf32"]["precision"] == "high"
    reset_cache()
