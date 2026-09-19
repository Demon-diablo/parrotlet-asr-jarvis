"""Centralised runtime configuration for the Jarvis SGLang ASR & Extraction service.

All values are read from environment variables.
Supports NVIDIA RTX PRO 6000 (96GB / 48GB Ada Lovelace / Blackwell).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env_str(name: str, default: str = "") -> str:
    """Return stripped env value, falling back to default if unset."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={value!r} is not a valid integer"
        ) from exc


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name}={value!r} is not a valid float"
        ) from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ----- allowed enums (kept as constants for validation) ---------------------- #

_ALLOWED_QUANTIZATION = {"none", "8bit", "4bit", "fp8", "fp8_compiled"}
_ALLOWED_DTYPE = {"auto", "fp16", "bf16", "bfloat16", "fp32", "tf32"}
_ALLOWED_DEVICE_MAP = {"auto", "cuda", "cpu", "balanced", "sequential"}
_ALLOWED_FP8_COMPILE_MODE = {"default", "reduce-overhead", "max-autotune"}
_ALLOWED_FP8_CONFIG = {"dynamic", "weight_only"}
_ALLOWED_FLOAT32_MATMUL_PRECISION = {"highest", "high", "medium"}
_ALLOWED_EXTRACTOR_BACKEND = {"sglang", "vllm", "transformers", "none"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration loaded at worker start.

    Loading happens once per process via :func:`get_settings`. Tests and
    scripts can pass a custom environment mapping through :func:`load_settings`.
    """

    model_id: str
    model_dir: str
    model_revision: str
    hf_token: str
    quantization: str
    dtype: str
    device_map_mode: str
    model_cache_dir: str
    max_new_tokens: int
    temperature: Optional[float]
    # Ban non-Latin Indic script tokens at the decoder (Devanagari, Telugu, etc.)
    ban_script_tokens: bool = True
    # Deterministic transcript cleaner: strip <noise tags>, unwrap [gloss] brackets
    clean_transcript: bool = True
    # FP8 compilation configuration
    fp8_compile_mode: str = "default"
    fp8_config: str = "dynamic"
    # Latency optimizations:
    use_static_cache: bool = True
    trim_silence: bool = False
    # TensorFloat-32 (TF32) architecture support on RTX 6000 Pro / Ampere+
    allow_tf32: bool = True
    float32_matmul_precision: str = "high"
    # Zero-hop MedGemma clinical extraction engine configuration
    extractor_backend: str = "sglang"
    medgemma_model_id: str = "google/medgemma-4b-it"
    extractor_max_tokens: int = 4096
    extractor_temperature: float = 0.0
    # SGLang specific configuration (Optimized for 96GB / 48GB GPUs)
    sglang_mem_fraction: float = 0.50
    sglang_context_len: int = 8192
    sglang_attention_backend: str = "flashinfer"
    speculative_draft_path: str = ""
    speculative_steps: int = 3
    # vLLM fallback settings (if ever toggled)
    extractor_gpu_memory_utilization: float = 0.45
    extractor_enforce_eager: bool = False

    @property
    def model_source(self) -> str:
        """Resolved model source identifier: local directory or HF repo id."""
        return self.model_dir if self.model_dir else self.model_id

    @property
    def quantization_enabled(self) -> bool:
        return self.quantization in {"8bit", "4bit", "fp8", "fp8_compiled"}

    @property
    def is_fp8(self) -> bool:
        return self.quantization in {"fp8", "fp8_compiled"}

    def as_dict(self) -> dict:
        """Serialisable view used for startup logging and diagnostics."""
        return {
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "model_revision": self.model_revision,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "device_map_mode": self.device_map_mode,
            "model_cache_dir": self.model_cache_dir,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "ban_script_tokens": self.ban_script_tokens,
            "clean_transcript": self.clean_transcript,
            "fp8_compile_mode": self.fp8_compile_mode,
            "fp8_config": self.fp8_config,
            "use_static_cache": self.use_static_cache,
            "trim_silence": self.trim_silence,
            "allow_tf32": self.allow_tf32,
            "float32_matmul_precision": self.float32_matmul_precision,
            "extractor_backend": self.extractor_backend,
            "medgemma_model_id": self.medgemma_model_id,
            "extractor_max_tokens": self.extractor_max_tokens,
            "extractor_temperature": self.extractor_temperature,
            "sglang_mem_fraction": self.sglang_mem_fraction,
            "sglang_context_len": self.sglang_context_len,
            "sglang_attention_backend": self.sglang_attention_backend,
            "speculative_draft_path": self.speculative_draft_path,
            "speculative_steps": self.speculative_steps,
        }


def load_settings(env: Optional[dict] = None) -> Settings:
    """Build a :class:`Settings` snapshot."""
    get = (lambda k, d="": env.get(k, d)) if env is not None else _env_str
    get_int = (lambda k, d: int(env.get(k, d))) if env is not None else _env_int
    get_float = (
        (lambda k, d: (float(env.get(k, "")) if env.get(k, "") not in ("", None) else d))
        if env is not None
        else _env_float
    )
    if env is not None:
        def _get_bool(k: str, d: bool) -> bool:
            v = env.get(k, "")
            if v == "" or v is None:
                return d
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}
    else:
        _get_bool = _env_bool

    quantization = (get("QUANTIZATION", "none") or "none").lower()
    dtype = (get("DTYPE", "auto") or "auto").lower()
    device_map_mode = (get("DEVICE_MAP_MODE", "auto") or "auto").lower()
    fp8_compile_mode = (get("FP8_COMPILE_MODE", "default") or "default").lower()
    fp8_config = (get("FP8_CONFIG", "weight_only") or "weight_only").lower()
    allow_tf32 = _get_bool("ALLOW_TF32", True)
    float32_matmul_precision = (get("FLOAT32_MATMUL_PRECISION", "high") or "high").lower()
    extractor_backend = (get("EXTRACTOR_BACKEND", "") or get("EXTRACTOR_ENGINE", "sglang") or "sglang").lower()

    if quantization not in _ALLOWED_QUANTIZATION:
        raise ValueError(
            f"Invalid QUANTIZATION={quantization!r}; expected one of {sorted(_ALLOWED_QUANTIZATION)}"
        )
    if dtype not in _ALLOWED_DTYPE:
        raise ValueError(
            f"Invalid DTYPE={dtype!r}; expected one of {sorted(_ALLOWED_DTYPE)}"
        )
    if device_map_mode not in _ALLOWED_DEVICE_MAP:
        raise ValueError(
            f"Invalid DEVICE_MAP_MODE={device_map_mode!r}; expected one of {sorted(_ALLOWED_DEVICE_MAP)}"
        )
    if fp8_compile_mode not in _ALLOWED_FP8_COMPILE_MODE:
        raise ValueError(
            f"Invalid FP8_COMPILE_MODE={fp8_compile_mode!r}; expected one of {sorted(_ALLOWED_FP8_COMPILE_MODE)}"
        )
    if fp8_config not in _ALLOWED_FP8_CONFIG:
        raise ValueError(
            f"Invalid FP8_CONFIG={fp8_config!r}; expected one of {sorted(_ALLOWED_FP8_CONFIG)}"
        )
    if float32_matmul_precision not in _ALLOWED_FLOAT32_MATMUL_PRECISION:
        raise ValueError(
            f"Invalid FLOAT32_MATMUL_PRECISION={float32_matmul_precision!r}; expected one of {sorted(_ALLOWED_FLOAT32_MATMUL_PRECISION)}"
        )
    if extractor_backend not in _ALLOWED_EXTRACTOR_BACKEND:
        raise ValueError(
            f"Invalid EXTRACTOR_BACKEND={extractor_backend!r}; expected one of {sorted(_ALLOWED_EXTRACTOR_BACKEND)}"
        )

    return Settings(
        model_id=get("MODEL_ID", "ekacare/parrotlet-a-2.5-pro"),
        model_dir=get("MODEL_DIR", ""),
        model_revision=get("MODEL_REVISION", ""),
        hf_token=get("HF_TOKEN", ""),
        quantization=quantization,
        dtype=dtype,
        device_map_mode=device_map_mode,
        model_cache_dir=get("MODEL_CACHE_DIR", ""),
        max_new_tokens=get_int("MAX_NEW_TOKENS", 256),
        temperature=get_float("TEMPERATURE", None),
        ban_script_tokens=_get_bool("BAN_SCRIPT_TOKENS", True),
        clean_transcript=_get_bool("CLEAN_TRANSCRIPT", True),
        fp8_compile_mode=fp8_compile_mode,
        fp8_config=fp8_config,
        use_static_cache=_get_bool("USE_STATIC_CACHE", True),
        trim_silence=_get_bool("TRIM_SILENCE", False),
        allow_tf32=allow_tf32,
        float32_matmul_precision=float32_matmul_precision,
        extractor_backend=extractor_backend,
        medgemma_model_id=get("MEDGEMMA_MODEL_ID", "") or get("MEDGEMMA_ID", "google/medgemma-4b-it"),
        extractor_max_tokens=get_int("EXTRACTOR_MAX_TOKENS", 4096),
        extractor_temperature=get_float("EXTRACTOR_TEMPERATURE", 0.0) or 0.0,
        sglang_mem_fraction=get_float("SGLANG_MEM_FRACTION", 0.50) or 0.50,
        sglang_context_len=get_int("SGLANG_CONTEXT_LEN", 8192),
        sglang_attention_backend=get("SGLANG_ATTENTION_BACKEND", "flashinfer") or "flashinfer",
        speculative_draft_path=get("SPECULATIVE_DRAFT_PATH", ""),
        speculative_steps=get_int("SPECULATIVE_STEPS", 3),
        extractor_gpu_memory_utilization=get_float("EXTRACTOR_GPU_MEMORY_UTILIZATION", 0.45) or 0.45,
        extractor_enforce_eager=_get_bool("EXTRACTOR_ENFORCE_EAGER", False),
    )


_cached: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide settings, loading on first call."""
    global _cached
    if _cached is None:
        _cached = load_settings()
    return _cached


def reset_settings_cache() -> None:
    """Clear the cached settings (useful in tests when env changes)."""
    global _cached
    _cached = None
