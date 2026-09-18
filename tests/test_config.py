"""Unit tests for src/config.py (spec #55 Configuration)."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure src is importable when running pytest from the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import (  # noqa: E402
    Settings,
    load_settings,
    reset_settings_cache,
)


def _base_env() -> dict:
    return {
        "MODEL_ID": "ekacare/parrotlet-a-2.5-pro",
        "MODEL_DIR": "",
        "MODEL_REVISION": "",
        "HF_TOKEN": "",
        "QUANTIZATION": "none",
        "DTYPE": "auto",
        "DEVICE_MAP_MODE": "auto",
        "MODEL_CACHE_DIR": "",
        "MAX_NEW_TOKENS": "256",
        "TEMPERATURE": "",
    }


def test_default_model_id():
    env = _base_env()
    env.pop("MODEL_ID", None)
    settings = load_settings(env)
    assert settings.model_id == "ekacare/parrotlet-a-2.5-pro"


def test_env_override_model_id():
    env = _base_env()
    env["MODEL_ID"] = "myorg/my-model"
    settings = load_settings(env)
    assert settings.model_id == "myorg/my-model"


def test_env_override_max_new_tokens():
    env = _base_env()
    env["MAX_NEW_TOKENS"] = "512"
    settings = load_settings(env)
    assert settings.max_new_tokens == 512


def test_temperature_optional():
    env = _base_env()
    settings = load_settings(env)
    assert settings.temperature is None

    env["TEMPERATURE"] = "0.2"
    settings = load_settings(env)
    assert settings.temperature == 0.2


def test_model_source_uses_model_dir_when_set():
    env = _base_env()
    env["MODEL_DIR"] = "/mnt/models/parrotlet"
    settings = load_settings(env)
    assert settings.model_source == "/mnt/models/parrotlet"


def test_model_source_falls_back_to_model_id():
    env = _base_env()
    env.pop("MODEL_DIR", None)
    settings = load_settings(env)
    assert settings.model_source == settings.model_id


def test_invalid_quantization_raises():
    env = _base_env()
    env["QUANTIZATION"] = "16bit"
    with pytest.raises(ValueError, match="Invalid QUANTIZATION"):
        load_settings(env)


def test_invalid_dtype_raises():
    env = _base_env()
    env["DTYPE"] = "fp64"
    with pytest.raises(ValueError, match="Invalid DTYPE"):
        load_settings(env)


def test_invalid_device_map_raises():
    env = _base_env()
    env["DEVICE_MAP_MODE"] = "gpu-farm"
    with pytest.raises(ValueError, match="Invalid DEVICE_MAP_MODE"):
        load_settings(env)


def test_get_settings_single_model_id_field():
    """as_dict must never leak hf_token into logs/diagnostics."""
    env = _base_env()
    env["HF_TOKEN"] = "super-secret"
    settings = load_settings(env)
    dumped = settings.as_dict()
    assert "hf_token" not in dumped
    assert settings.hf_token == "super-secret"


def test_fp8_settings_valid():
    env = _base_env()
    env["QUANTIZATION"] = "fp8"
    env["FP8_COMPILE_MODE"] = "default"
    env["FP8_CONFIG"] = "dynamic"
    settings = load_settings(env)
    assert settings.quantization == "fp8"
    assert settings.is_fp8 is True
    assert settings.quantization_enabled is True
    assert settings.fp8_compile_mode == "default"
    assert settings.fp8_config == "dynamic"

    env["QUANTIZATION"] = "fp8_compiled"
    env["FP8_COMPILE_MODE"] = "reduce-overhead"
    env["FP8_CONFIG"] = "weight_only"
    settings = load_settings(env)
    assert settings.quantization == "fp8_compiled"
    assert settings.is_fp8 is True
    assert settings.quantization_enabled is True
    assert settings.fp8_compile_mode == "reduce-overhead"
    assert settings.fp8_config == "weight_only"


def test_invalid_fp8_compile_mode_raises():
    env = _base_env()
    env["QUANTIZATION"] = "fp8_compiled"
    env["FP8_COMPILE_MODE"] = "hyper-speed"
    with pytest.raises(ValueError, match="Invalid FP8_COMPILE_MODE"):
        load_settings(env)


def test_invalid_fp8_config_raises():
    env = _base_env()
    env["QUANTIZATION"] = "fp8"
    env["FP8_CONFIG"] = "ultra_int4"
    with pytest.raises(ValueError, match="Invalid FP8_CONFIG"):
        load_settings(env)


def test_tf32_settings_default():
    env = _base_env()
    settings = load_settings(env)
    assert settings.allow_tf32 is True
    assert settings.float32_matmul_precision == "high"


def test_tf32_env_override():
    env = _base_env()
    env["ALLOW_TF32"] = "0"
    env["FLOAT32_MATMUL_PRECISION"] = "medium"
    settings = load_settings(env)
    assert settings.allow_tf32 is False
    assert settings.float32_matmul_precision == "medium"


def test_dtype_tf32_valid():
    env = _base_env()
    env["DTYPE"] = "tf32"
    settings = load_settings(env)
    assert settings.dtype == "tf32"


def test_invalid_float32_matmul_precision_raises():
    env = _base_env()
    env["FLOAT32_MATMUL_PRECISION"] = "ultra_fast"
    with pytest.raises(ValueError, match="Invalid FLOAT32_MATMUL_PRECISION"):
        load_settings(env)


def test_as_dict_contains_tf32():
    env = _base_env()
    settings = load_settings(env)
    dumped = settings.as_dict()
    assert dumped["allow_tf32"] is True
    assert dumped["float32_matmul_precision"] == "high"