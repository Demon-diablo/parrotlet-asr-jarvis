"""Parrotlet-a 2.5 Pro model loader.

Architecture (verified during Phase 3 inspection of
``ekacare/parrotlet-a-2.5-pro``):

  Audio input (16 kHz numpy)
        │
        ▼
  WhisperEncoder  (openai/whisper-large-v3 encoder, bf16)
        │
        ▼
  EncoderProjectorConcat (custom Linear 1280*2 → 4096 → 2560)
        │
        ▼
  Gemma3ForConditionalGeneration decoder, fed via inputs_embeds=
  with the <audio> placeholder replaced by projected embeddings.
        │
        ▼
  ASR transcript (string)

The upstream loader (``modelling_speech-llm.py``) defines ``SpeechLLM`` and
``SpeechLLMConfig`` and registers them with ``AutoConfig`` / ``AutoModel`` via
the ``auto_map`` in the root ``config.json``. Loading therefore REQUIRES
``trust_remote_code=True`` and the ``AutoConfig.register`` + ``AutoModel.register``
calls (executed by the upstream loader itself).

This module is the single authoritative loader used by:

- ``serve_jarvis.py`` / ``src/server.py`` (FastAPI GPU worker and HTTP service)
- ``scripts/check_model.py``
- ``scripts/benchmark.py``

(spec rule: do not duplicate model logic).

Key design rules respected:
- GPU-agnostic (spec #21/#23): the device is resolved at runtime by
  ``src.gpu.gpu_info()`` + ``DEVICE_MAP_MODE`` env. We never hard-code
  ``cuda:0`` or assume multi-GPU.
- Load once (spec #5.4/#37): the module-level ``_CACHE`` is the authoritative
  store. A ``threading.Lock`` makes concurrent Serverless job dispatch safe.
- Placement validation (spec #22): after load we report each submodule's
  resolved device so "from_pretrained didn't raise" is NOT the success bar.
- Configurable source (spec #16/#48): ``MODEL_ID`` (HF repo) or ``MODEL_DIR``
  (local/mounted) — never a hard-coded temporary path.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .config import Settings, get_settings
from .gpu import configure_tf32, gpu_info, peak_memory_gb

log = logging.getLogger("parrotlet.model")


# ---------------------------------------------------------------------------
# Lazy optional imports — keep module importable in unit-test environments
# that don't have the full ML stack installed.
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Module-level cache (spec #6: load once, reuse).
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Any] = {
    "speech_llm": None,           # the SpeechLLM PreTrainedModel wrapper
    "encoder": None,              # Whisper encoder module
    "projector": None,            # EncoderProjectorConcat module
    "decoder": None,              # Gemma3ForConditionalGeneration
    "tokenizer": None,            # Gemma3 tokenizer
    "processor": None,            # WhisperProcessor
    "metadata": None,             # dict captured at load time
    "banned_ids": None,           # bad_words_ids list for script ban (or [])
    "ban_enabled": True,          # set at load; ensure_banned_ids() respects it
    "lock": threading.Lock(),
    "loaded": False,
}


# ---------------------------------------------------------------------------
# Script-token ban (Sept-2025 benchmark WIN).
#
# bad_words_ids with ~15k Telugu+Devanagari IDs cut semWER 0.73->0.21 on
# meds-abx-gi; prompt-steering alone LOST. We resolve IDs dynamically by
# scanning the live tokenizer vocab so the list survives tokenizer upgrades.
# Ranges cover Devanagari (Hindi/Marathi), Telugu, Gujarati (leaked in
# rx-acute ban run as એક ગોળી), Bengali (ফির leak in rx-htn-dm), plus the
# other major Indic blocks so a new data source can't reintroduce them.
# ---------------------------------------------------------------------------
_BANNED_SCRIPT_RANGES = (
    (0x0900, 0x097F),  # Devanagari (Hindi)
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)


def _is_banned_char(ch: str) -> bool:
    """True if a single character falls in a banned Indic block."""
    cp = ord(ch)
    for lo, hi in _BANNED_SCRIPT_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _token_has_banned_script(token_str: str) -> bool:
    """True if the decoded token text contains any banned-script char.

    The SentencePiece ``▁`` word-boundary marker carries no script signal and
    is skipped. Byte-fallback pieces (e.g. ``<0xE0>``) are pure ASCII, so they
    never match; the caller decodes each ID first, which resolves those to
    their true surface form before this check runs.
    """
    if not token_str:
        return False
    for ch in token_str:
        if ch == "▁":
            continue
        if _is_banned_char(ch):
            return True
    return False


def _resolve_banned_token_ids(tokenizer: Any) -> list:
    """Scan the tokenizer vocab for banned-script tokens.

    Returns a sorted list of token IDs suitable for
    ``generate(bad_words_ids=[[i] for i in ids])``. Returns [] when the
    tokenizer is missing or exposes no vocab (ban silently disables).
    """
    if tokenizer is None:
        return []
    try:
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, "vocab"):
            vocab = tokenizer.vocab
        else:
            return []
    except Exception:
        return []
    banned: list = []
    decode = getattr(tokenizer, "decode", None)
    for tok, tid in vocab.items():
        try:
            text = tok
            # Most HF tokenizers store raw pieces; decode one ID to get the
            # true surface form when possible (handles byte-fallback).
            if decode is not None and isinstance(tid, int):
                try:
                    text = tokenizer.decode([tid])
                except Exception:
                    text = tok
            if _token_has_banned_script(str(text)):
                banned.append(int(tid))
        except Exception:
            continue
    return sorted(set(banned))


def get_banned_token_ids() -> list:
    """Return cached banned IDs (may be []); never forces a model load."""
    return list(ensure_banned_ids() or [])


def ensure_banned_ids() -> list:
    """Compute the script-ban ID list on first use (NOT at load).

    The 256k-vocab scan costs seconds inside the critical load path, so
    load_model() only marks it pending (None) and the first transcribe that
    needs it pays once, cached under lock. Never forces a model load: with no
    cached tokenizer (or ban disabled) returns [].
    """
    with _CACHE["lock"]:
        if _CACHE.get("banned_ids") is not None:
            return list(_CACHE["banned_ids"])
        if not _CACHE.get("ban_enabled", True):
            return []
        tokenizer = _CACHE.get("tokenizer")
        if tokenizer is None:
            return []
        try:
            ids = _resolve_banned_token_ids(tokenizer)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("script-token ban resolution failed, disabling: %s", exc)
            ids = []
        _CACHE["banned_ids"] = list(ids)
        md = _CACHE.get("metadata")
        if isinstance(md, dict):
            md["banned_token_count"] = len(ids)
        return list(ids)


# ---------------------------------------------------------------------------
# Public dataclass returned by load_model().
# ---------------------------------------------------------------------------
@dataclass
class LoadedModel:
    """Bundle of everything ``inference.py`` needs.

    The dataclass intentionally exposes every submodule as its own field so
    downstream code never reaches into ``speech_llm.encoder`` /
    ``speech_llm.decoder`` etc. directly (which would couple inference to the
    custom loader's internal naming).
    """

    speech_llm: Any
    encoder: Any
    projector: Any
    decoder: Any
    tokenizer: Any
    processor: Any
    metadata: Dict[str, Any]
    sampling_rate: int
    # bad_words_ids resolved at load time ([] when ban disabled/unavailable).
    # Optional with default so older call sites / tests keep working.
    banned_token_ids: Any = None

    def unwrap(self) -> Tuple[Any, Any]:
        """Return ``(model_bundle, tokenizer)`` for code that just wants
        to call ``model.transcribe(...)``.
        """
        return self.speech_llm, self.tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_device(settings: Settings, runtime: Dict[str, Any]) -> str:
    """Map ``DEVICE_MAP_MODE`` + runtime GPU discovery to a concrete device.

    The upstream ``SpeechLLM.from_pretrained`` does not support
    ``device_map="auto"`` (it calls ``.to(device)`` per submodule), so we
    collapse to a single ``"cuda"`` / ``"cpu"`` device string.

    For multi-GPU rented workers, accelerate / FSDP is left out of the
    skeleton phase per spec #31/#33 — single-GPU is the right starting point
    (#30) and only if the baseline OOMs do we revisit model parallelism.
    """
    cuda_available = bool(runtime.get("cuda_available"))
    mode = settings.device_map_mode

    if mode == "cpu":
        return "cpu"

    if not cuda_available:
        if mode in {"auto", "cuda", "balanced", "sequential"}:
            log.warning(
                "CUDA not available but DEVICE_MAP_MODE=%s requested; "
                "falling back to 'cpu' for this worker.",
                mode,
            )
        return "cpu"

    # CUDA is available. We collapse to a single device index because the
    # upstream loader takes one device string and assigns each submodule to
    # it. Multi-GPU support would require patching the loader — deferred
    # until baseline measurements demand it (spec #31).
    return "cuda"


def _build_hf_kwargs(settings: Settings) -> Dict[str, Any]:
    """kwarg dict forwarded to ``from_pretrained`` / ``snapshot_download``."""
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    if settings.model_revision:
        kwargs["revision"] = settings.model_revision
    if settings.model_cache_dir:
        kwargs["cache_dir"] = settings.model_cache_dir
    return kwargs


def _resolve_dtype(settings: Settings):
    """Map DTYPE -> torch dtype. ``auto`` means "trust the upstream loader"
    which expects bf16 weights for this model.

    Returns ``None`` when ``DTYPE=auto`` so we leave dtype selection to the
    upstream loader (it will pick bf16 because every component ships bf16).
    """
    if torch is None:
        return None
    if settings.dtype == "auto":
        return None
    table = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "tf32": torch.float32,
    }
    return table.get(settings.dtype)


def _submodule_device(module: Any) -> Optional[str]:
    """Return the device string of ``module``'s first parameter, if any.

    Used for placement validation (spec #22). Returns ``None`` if the module
    has no parameters or no ``device`` attribute is reachable.
    """
    if module is None or torch is None:
        return None
    try:
        first_param = next(module.parameters(), None)
        if first_param is None:
            return None
        return str(first_param.device)
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Real loaders (Phase 3).
# ---------------------------------------------------------------------------
def _import_custom_classes(speech_llm_cls: Any, speech_llm_config_cls: Any) -> None:
    """Register ``SpeechLLM`` / ``SpeechLLMConfig`` with AutoModel and
    AutoConfig so ``from_pretrained`` can dispatch to the custom class.

    The upstream loader runs these register calls inside
    ``__main__``-only branch; we run them unconditionally because the same
    module is imported via ``trust_remote_code=True``.
    """
    try:
        from transformers import AutoConfig, AutoModel  # type: ignore

        AutoConfig.register("speech-llm", speech_llm_config_cls)
        AutoModel.register(speech_llm_config_cls, speech_llm_cls)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("AutoConfig/AutoModel registration skipped: %s", exc)


def _load_speech_llm(settings: Settings, hf_kwargs: Dict[str, Any], device: str, dtype) -> Any:
    """Load the custom SpeechLLM wrapper.

    Strategy: import the upstream module from the HF cache (downloaded by the
    custom ``from_pretrained`` inside the wrapper) and call it. We do not
    call ``AutoModel.from_pretrained`` here because the wrapper's own
    ``from_pretrained`` does the snapshot_download + per-submodule loading
    and is the contract documented in the README.

    Requires ``torch`` + ``transformers`` to be importable. When either is
    missing (e.g. inside a unit-test venv that didn't install them), raise a
    clear ``RuntimeError`` so the caller surfaces a structured error
    instead of an opaque traceback.
    """
    if torch is None:
        raise RuntimeError(
            "torch is not importable. The Parrotlet loader requires torch; "
            "install requirements.txt before invoking load_model()."
        )

    try:
        from transformers import AutoModel  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"transformers is not importable ({exc}). The Parrotlet loader "
            "requires transformers; install requirements.txt before invoking "
            "load_model()."
        ) from exc

    # Step 1: resolve the repo to a local directory. A local MODEL_DIR
    # is used directly; otherwise snapshot the HF repo into the cache.
    from huggingface_hub import snapshot_download  # type: ignore

    repo_id = settings.model_source
    if os.path.isdir(repo_id):
        local_dir = repo_id
    else:
        download_kwargs: Dict[str, Any] = {}
        if settings.hf_token:
            download_kwargs["token"] = settings.hf_token
        if settings.model_revision:
            download_kwargs["revision"] = settings.model_revision
        if settings.model_cache_dir:
            download_kwargs["cache_dir"] = settings.model_cache_dir

        local_dir = snapshot_download(repo_id=repo_id, **download_kwargs)

    # Step 2: import the custom module by file path.
    import importlib.util
    import sys as _sys

    module_path = os.path.join(local_dir, "modelling_speech-llm.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"Custom modelling file not found in snapshot: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "parrotlet_modelling_speech_llm", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom modelling module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["parrotlet_modelling_speech_llm"] = mod
    spec.loader.exec_module(mod)

    SpeechLLM = getattr(mod, "SpeechLLM")
    SpeechLLMConfig = getattr(mod, "SpeechLLMConfig")

    # Step 3: register so AutoModel dispatch works (also called inside
    # the upstream ``__main__`` branch which won't run for us).
    _import_custom_classes(SpeechLLM, SpeechLLMConfig)

    # Step 4: load. The upstream from_pretrained() only accepts HF repo ids
    # (it calls snapshot_download(repo_id=...) + cached_file(...) internally
    # and rejects local paths). When MODEL_DIR points at a local snapshot
    # that came from MODEL_ID, pass the HF id: with HF_HOME pointing at the
    # same cache dir (see MODEL_CACHE_DIR), the
    # upstream download is a cache hit, not a re-download.
    load_kwargs = dict(hf_kwargs)
    load_kwargs["device"] = device
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    pretrained_ref = repo_id
    if os.path.isdir(repo_id) and settings.model_id:
        pretrained_ref = settings.model_id

    # Attention backend selection. flash_attention_2 was previously hard-set
    # unconditionally, but the `flash-attn` package is optional
    # (requirements.txt) — if the upstream loader's own error handling swallows
    # the resulting ImportError, this silently lands on eager attention,
    # which has no fused kernel and no efficient StaticCache reuse, making
    # every decode step meaningfully slower. Only request flash_attention_2
    # when the package is actually importable in this container; otherwise
    # prefer sdpa (PyTorch's fused kernel, ships in the box, no extra
    # install) over eager. Whichever backend loads is logged so a slow
    # deployment can be diagnosed from the worker logs instead of guessed at.
    attn_candidates = []
    try:
        import flash_attn  # type: ignore  # noqa: F401  (availability probe only)

        attn_candidates.append("flash_attention_2")
    except Exception:
        log.info("flash-attn not importable in this image; skipping flash_attention_2")
    attn_candidates += ["sdpa", "eager"]

    last_exc: Optional[Exception] = None
    for attn_impl in attn_candidates:
        try_kwargs = dict(load_kwargs)
        try_kwargs["attn_implementation"] = attn_impl
        try:
            model_obj = SpeechLLM.from_pretrained(pretrained_ref, **try_kwargs)
            log.info("Parrotlet loaded with attn_implementation=%s", attn_impl)
            return model_obj
        except TypeError as exc:
            # Older/custom from_pretrained without attn support at all.
            if "attn_implementation" not in str(exc):
                raise
            log.warning("from_pretrained rejects attn_implementation kwarg; retrying without it: %s", exc)
            try_kwargs.pop("attn_implementation", None)
            return SpeechLLM.from_pretrained(pretrained_ref, **try_kwargs)
        except Exception as exc:
            last_exc = exc
            log.warning("attn_implementation=%s failed to load (%s); trying next backend", attn_impl, exc)
            continue

    raise RuntimeError(f"Could not load SpeechLLM with any attention backend: {last_exc}")


def _apply_fp8_quantization(speech_llm: Any, settings: Settings, device: str) -> Dict[str, Any]:
    """Applies FP8 quantization (and optional compilation) to the MedGemma decoder.

    Selectively targets the Gemma 3 language_model linear projections
    (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), preserving
    lm_head and any vision tower modules in full precision.

    If settings.quantization == 'fp8_compiled', wraps language_model in
    torch.compile(mode=settings.fp8_compile_mode, dynamic=True) and warms up
    the compiled graph to prevent first-request JIT latency spikes.
    """
    if torch is None or not torch.cuda.is_available():
        log.warning("CUDA not available; skipping FP8 quantization.")
        return {"status": "skipped", "reason": "no_cuda"}

    try:
        import torchao  # type: ignore  # noqa: F401
        from torchao.quantization import quantize_  # type: ignore
        if settings.fp8_config == "weight_only":
            from torchao.quantization import float8_weight_only  # type: ignore
            quant_cfg = float8_weight_only()
        else:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig  # type: ignore
            quant_cfg = Float8DynamicActivationFloat8WeightConfig()
    except Exception as exc:
        raise RuntimeError(
            f"torchao is required for QUANTIZATION={settings.quantization}. "
            f"Please ensure torchao is installed. Error: {exc}"
        ) from exc

    decoder = getattr(speech_llm, "decoder", None)
    if decoder is None:
        raise ValueError("SpeechLLM has no decoder attribute to quantize.")

    log.info(
        "Applying FP8 quantization (config=%s) to decoder...",
        settings.fp8_config,
    )
    t_q0 = time.perf_counter()

    # Filter: quantize all text linear layers, but exclude lm_head & vision.
    # NOTE: With fp8_config="weight_only", activations stay BF16 — safe for all layers.
    # With fp8_config="dynamic", activations are also quantized to FP8 — unstable on
    # this decoder (see quant report Ch 6); use weight_only for production.
    def _is_text_linear(mod: Any, fqn: str) -> bool:
        if not isinstance(mod, torch.nn.Linear):
            return False
        if "lm_head" in fqn or "vision" in fqn:
            return False
        return True

    target_module = getattr(decoder, "language_model", decoder)
    quantize_(target_module, quant_cfg, filter_fn=_is_text_linear)
    quant_time = time.perf_counter() - t_q0

    compile_time = 0.0
    is_compiled = settings.quantization == "fp8_compiled"

    if is_compiled:
        log.info(
            "Compiling FP8 decoder MLP blocks with mode=%s...",
            settings.fp8_compile_mode,
        )
        t_c0 = time.perf_counter()
        try:
            # Helper to extract decoder layers
            def _get_layers(target: Any) -> list[Any] | None:
                if hasattr(target, "model") and hasattr(target.model, "layers"):
                    return target.model.layers
                if hasattr(target, "language_model") and hasattr(target.language_model, "layers"):
                    return target.language_model.layers
                if hasattr(target, "layers"):
                    return target.layers
                return None

            layers = _get_layers(target_module) or _get_layers(decoder)
            if layers:
                log.info("Attaching torch.compile to %d FP8 MLP blocks...", len(layers))
                for layer in layers:
                    if hasattr(layer, "mlp"):
                        layer.mlp = torch.compile(layer.mlp, mode=settings.fp8_compile_mode)
            else:
                if hasattr(decoder, "language_model"):
                    decoder.language_model = torch.compile(
                        decoder.language_model,
                        mode=settings.fp8_compile_mode,
                        dynamic=True,
                    )
                else:
                    speech_llm.decoder = torch.compile(
                        decoder,
                        mode=settings.fp8_compile_mode,
                        dynamic=True,
                    )
            compile_time = time.perf_counter() - t_c0
            log.info("torch.compile wrapper attached in %.2fs", compile_time)

            # Pre-warm JIT compilation
            log.info("Warming up compiled FP8 decoder to trigger JIT compilation...")
            with torch.no_grad():
                if layers and hasattr(layers[0], "mlp"):
                    h_dim = getattr(
                        getattr(layers[0].mlp, "gate_proj", None), "in_features", 2560
                    )
                    dummy_in = torch.zeros((1, 1, h_dim), dtype=torch.bfloat16, device=device)
                    for layer in layers:
                        if hasattr(layer, "mlp"):
                            _ = layer.mlp(dummy_in)
                else:
                    dummy_ids = torch.zeros((1, 4), dtype=torch.long, device=device)
                    _ = decoder(input_ids=dummy_ids)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
            log.info("FP8 compilation warmup complete.")
        except Exception as exc:
            log.warning("torch.compile warmup encountered non-fatal error: %s", exc)

    return {
        "status": "applied",
        "fp8_config": settings.fp8_config,
        "is_compiled": is_compiled,
        "compile_mode": settings.fp8_compile_mode if is_compiled else None,
        "quant_seconds": round(quant_time, 3),
        "compile_seconds": round(compile_time, 3),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_model(force_reload: bool = False) -> LoadedModel:
    """Return a cached :class:`LoadedModel`, loading on first call.

    Thread-safe so concurrent Serverless jobs can't double-load. ``force_reload``
    bypasses the cache (tests + manual config changes).
    """
    lock = _CACHE["lock"]
    with lock:
        if _CACHE["loaded"] and not force_reload:
            return LoadedModel(
                speech_llm=_CACHE["speech_llm"],
                encoder=_CACHE["encoder"],
                projector=_CACHE["projector"],
                decoder=_CACHE["decoder"],
                tokenizer=_CACHE["tokenizer"],
                processor=_CACHE["processor"],
                metadata=_CACHE["metadata"],
                sampling_rate=int(_CACHE["metadata"].get("sampling_rate", 16000)),
                banned_token_ids=list(_CACHE.get("banned_ids") or []),
            )

        settings = get_settings()
        runtime = gpu_info()
        hf_kwargs = _build_hf_kwargs(settings)
        device = _resolve_device(settings, runtime)
        dtype = _resolve_dtype(settings)

        # TF32 execution configuration for RTX 6000 Pro / Ampere+ Tensor Cores
        tf32_status = configure_tf32(
            enabled=settings.allow_tf32,
            precision=settings.float32_matmul_precision,
        )
        if str(device).startswith("cuda") and settings.allow_tf32:
            log.info(
                "TF32 math enabled on Tensor Cores (precision=%s, matmul=%s, cudnn=%s)",
                settings.float32_matmul_precision,
                tf32_status.get("allow_tf32_matmul"),
                tf32_status.get("allow_tf32_cudnn"),
            )

        # Inductor dynamic shape cudagraph storm mitigation:
        # Keep inductor's fused kernels but never re-record CUDAGraphs
        # for dynamic decode positions.
        try:
            import torch as _t2

            _cfg = getattr(getattr(_t2, "_inductor", None), "config", None)
            if _cfg is not None:
                _cfg.triton.cudagraph_skip_dynamic_graphs = True
        except Exception as exc:
            log.debug("inductor cudagraph config skipped: %s", exc)

        model_source = settings.model_source

        log.info(
            "Loading Parrotlet source=%s revision=%s dtype=%s device=%s "
            "quantization=%s runtime_cuda=%s gpu_count=%d tf32=%s",
            model_source,
            settings.model_revision or "<latest>",
            settings.dtype,
            device,
            settings.quantization,
            runtime.get("cuda_available"),
            int(runtime.get("gpu_count", 0)),
            settings.allow_tf32,
        )

        start = time.perf_counter()
        speech_llm = _load_speech_llm(settings, hf_kwargs, device=device, dtype=dtype)
        elapsed = time.perf_counter() - start

        # If FP8 quantization requested, apply to decoder
        fp8_meta = {}
        if getattr(settings, "is_fp8", False):
            fp8_meta = _apply_fp8_quantization(speech_llm, settings, device)

        # Module references for placement reporting.
        encoder = getattr(speech_llm, "encoder", None)
        projector = getattr(speech_llm, "projector", None)
        decoder = getattr(speech_llm, "decoder", None)
        tokenizer = getattr(speech_llm, "tokenizer", None)
        processor = getattr(speech_llm, "processor", None)
        sampling_rate = int(getattr(speech_llm, "sampling_rate", 16000))

        # Placement validation: don't trust "from_pretrained() returned".
        placement = {
            "encoder": _submodule_device(encoder),
            "projector": _submodule_device(projector),
            "decoder": _submodule_device(decoder),
        }

        # Refresh runtime after TF32/CUDA setup
        runtime_current = gpu_info()

        metadata: Dict[str, Any] = {
            "model_source": model_source,
            "model_revision": settings.model_revision or "<latest>",
            "dtype": settings.dtype,
            "device_map_mode": settings.device_map_mode,
            "resolved_device": device,
            "quantization": settings.quantization,
            "load_seconds": round(elapsed, 3),
            "placement": placement,
            "runtime": runtime_current,
            "sampling_rate": sampling_rate,
            "fp8": fp8_meta,
            "tf32": {
                "enabled": settings.allow_tf32,
                "precision": settings.float32_matmul_precision,
                "supported": runtime_current.get("tf32_supported", False),
                "matmul_allow_tf32": runtime_current.get("allow_tf32_matmul"),
                "cudnn_allow_tf32": runtime_current.get("allow_tf32_cudnn"),
            },
        }

        # Script-token ban resolves LAZY on first transcribe (see
        # ensure_banned_ids): the 256k-vocab scan costs seconds and does not
        # belong in the critical load path. Mark pending (None) here.
        banned_ids: list = []
        ban_enabled = bool(getattr(settings, "ban_script_tokens", True))
        metadata["ban_script_tokens"] = ban_enabled
        metadata["banned_token_count"] = None  # filled by ensure_banned_ids()

        if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                metadata["peak_memory_gb"] = peak_memory_gb(0)
            except Exception:
                pass

        _CACHE["speech_llm"] = speech_llm
        _CACHE["encoder"] = encoder
        _CACHE["projector"] = projector
        _CACHE["decoder"] = decoder
        _CACHE["tokenizer"] = tokenizer
        _CACHE["processor"] = processor
        _CACHE["metadata"] = metadata
        _CACHE["banned_ids"] = None  # pending: see ensure_banned_ids()
        _CACHE["ban_enabled"] = ban_enabled
        _CACHE["loaded"] = True

        log.info(
            "Parrotlet loaded in %.3fs placement=%s sampling_rate=%d",
            elapsed,
            placement,
            sampling_rate,
        )
        log.info(
            "Script-token ban enabled=%s (ids resolve lazy on first transcribe)",
            ban_enabled,
        )

        return LoadedModel(
            speech_llm=speech_llm,
            encoder=encoder,
            projector=projector,
            decoder=decoder,
            tokenizer=tokenizer,
            processor=processor,
            metadata=metadata,
            sampling_rate=sampling_rate,
            banned_token_ids=list(banned_ids),
        )


def is_loaded() -> bool:
    return bool(_CACHE.get("loaded"))


def reset_cache() -> None:
    """Clear the module-level model cache (tests only)."""
    with _CACHE["lock"]:
        _CACHE["speech_llm"] = None
        _CACHE["encoder"] = None
        _CACHE["projector"] = None
        _CACHE["decoder"] = None
        _CACHE["tokenizer"] = None
        _CACHE["processor"] = None
        _CACHE["metadata"] = None
        _CACHE["banned_ids"] = None
        _CACHE["ban_enabled"] = True
        _CACHE["loaded"] = False


def model_metadata() -> Optional[Dict[str, Any]]:
    """Return the cached metadata dict without forcing a load."""
    return _CACHE.get("metadata")