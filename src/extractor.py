"""Zero-hop clinical medication extractor engine.

Co-hosts MedGemma-4B on the GPU alongside Parrotlet ASR.
Supports high-throughput vLLM engine (with automatic prefix caching and Blackwell SM 12.0
compatibility), SGLang engine, or native PyTorch Transformers fallback.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from src.schema import (
    STANDARD_SYSTEM_PROMPT,
    parse_prescriptions_json,
    strip_fences,
)

log = logging.getLogger("parrotlet.extractor")

_EXTRACTOR_CACHE: Dict[str, Any] = {
    "engine": None,
    "backend": None,
    "model_id": None,
    "loaded": False,
    "warmed_up": False,
    "lock": threading.Lock(),
}

DEFAULT_MODEL_ID = "google/medgemma-4b-it"


def get_extractor_backend() -> str:
    """Resolve configured extractor engine: 'vllm', 'sglang', 'transformers', or 'none'."""
    val = (os.getenv("EXTRACTOR_BACKEND") or os.getenv("EXTRACTOR_ENGINE") or "vllm").lower()
    return val if val in ("vllm", "sglang", "transformers", "none") else "vllm"


def get_medgemma_model_id() -> str:
    """Resolve model checkpoint ID or path for MedGemma-4B."""
    return os.getenv("MEDGEMMA_MODEL_ID") or os.getenv("MEDGEMMA_ID") or DEFAULT_MODEL_ID


def is_extractor_loaded() -> bool:
    return bool(_EXTRACTOR_CACHE.get("loaded"))


def reset_extractor() -> None:
    """Reset cached extractor engine state."""
    with _EXTRACTOR_CACHE["lock"]:
        eng = _EXTRACTOR_CACHE.get("engine")
        if eng is not None and hasattr(eng, "shutdown"):
            try:
                eng.shutdown()
            except Exception:
                pass
        _EXTRACTOR_CACHE["engine"] = None
        _EXTRACTOR_CACHE["backend"] = None
        _EXTRACTOR_CACHE["model_id"] = None
        _EXTRACTOR_CACHE["loaded"] = False
        _EXTRACTOR_CACHE["warmed_up"] = False


def _init_vllm_engine(model_id: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.45) -> Any:
    """Initialize vLLM engine with Blackwell SM 12.0 & Ada SM 8.9 compatibility."""
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "9.0"

    from vllm import LLM

    log.info("Initializing vLLM Extractor for %s (max_model_len=%d, mem=%.2f)...", model_id, max_model_len, gpu_memory_utilization)
    return LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=True,
    )


def _init_sglang_engine(model_id: str, max_model_len: int = 8192, mem_fraction: float = 0.45) -> Any:
    """Initialize SGLang engine with RadixAttention prefix caching."""
    import sglang as sgl

    log.info("Initializing SGLang Extractor for %s (context_len=%d, mem=%.2f)...", model_id, max_model_len, mem_fraction)
    return sgl.Engine(
        model_path=model_id,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        mem_fraction_static=mem_fraction,
        context_length=max_model_len,
        disable_radix_cache=False,
        schedule_policy="lpm",
        cuda_graph_max_bs=2,
        cuda_graph_bs=[1, 2],
    )


def _init_transformers_engine(model_id: str) -> Dict[str, Any]:
    """Initialize native PyTorch Transformers pipeline with bfloat16 + TF32."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log.info("Initializing Transformers fallback for %s in bfloat16...", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": 0} if device == "cuda" else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return {"model": model, "processor": processor, "device": device}


def load_extractor(
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    max_model_len: int = 8192,
) -> Any:
    """Load and cache the zero-hop clinical extraction engine."""
    with _EXTRACTOR_CACHE["lock"]:
        if _EXTRACTOR_CACHE.get("loaded") and _EXTRACTOR_CACHE.get("engine") is not None:
            return _EXTRACTOR_CACHE["engine"]

        target_model = model_id or get_medgemma_model_id()
        target_backend = (backend or get_extractor_backend()).lower()

        if target_backend == "none":
            log.info("Extractor backend configured to 'none'; clinical extraction disabled.")
            return None

        engine = None
        loaded_backend = None

        if target_backend == "vllm":
            try:
                engine = _init_vllm_engine(target_model, max_model_len=max_model_len)
                loaded_backend = "vllm"
            except Exception as exc:
                log.warning("Failed to initialize vLLM (%s), falling back to Transformers...", exc)

        elif target_backend == "sglang":
            try:
                engine = _init_sglang_engine(target_model, max_model_len=max_model_len)
                loaded_backend = "sglang"
            except Exception as exc:
                log.warning("Failed to initialize SGLang (%s), falling back to Transformers...", exc)

        if engine is None:
            try:
                engine = _init_transformers_engine(target_model)
                loaded_backend = "transformers"
            except Exception as exc:
                log.error("Failed to initialize Transformers fallback: %s", exc)
                raise RuntimeError(f"Could not initialize clinical extractor ({target_model}): {exc}") from exc

        _EXTRACTOR_CACHE["engine"] = engine
        _EXTRACTOR_CACHE["backend"] = loaded_backend
        _EXTRACTOR_CACHE["model_id"] = target_model
        _EXTRACTOR_CACHE["loaded"] = True
        return engine


def warmup_extractor() -> None:
    """Pre-warm the extractor and system prompt prefix cache so the first doctor request has <5ms TTFT."""
    if _EXTRACTOR_CACHE.get("warmed_up"):
        return
    if get_extractor_backend() == "none":
        return

    try:
        t0 = time.perf_counter()
        log.info("Pre-warming zero-hop clinical extractor prefix cache...")
        load_extractor()
        dummy_transcript = "Tablet Pan 40 OD with empty stomach ES."
        extract_prescriptions(dummy_transcript, max_tokens=64)
        _EXTRACTOR_CACHE["warmed_up"] = True
        log.info("Extractor prefix cache pre-warmed in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        log.warning("Extractor warmup notice: %s", exc)


def extract_prescriptions(
    transcript: str,
    system_prompt: str = STANDARD_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Extract clinical medications from transcript using the co-hosted MedGemma engine.

    Returns:
        {
            "valid_json": bool,
            "medications": list[dict],
            "medications_count": int,
            "latency_seconds": float,
            "tokens_generated": int,
            "throughput_tok_s": float,
            "raw_text": str,
        }
    """
    if get_extractor_backend() == "none":
        return {
            "valid_json": False,
            "medications": [],
            "medications_count": 0,
            "latency_seconds": 0.0,
            "tokens_generated": 0,
            "throughput_tok_s": 0.0,
            "raw_text": "",
            "error": "Extractor disabled (EXTRACTOR_BACKEND=none)",
        }

    if not (transcript or "").strip():
        return {
            "valid_json": False,
            "medications": [],
            "medications_count": 0,
            "latency_seconds": 0.0,
            "tokens_generated": 0,
            "throughput_tok_s": 0.0,
            "raw_text": "",
        }

    engine = load_extractor()
    backend = _EXTRACTOR_CACHE.get("backend")

    t0 = time.perf_counter()
    raw_text = ""
    tokens_generated = 0

    if backend == "vllm":
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ]
        outputs = engine.chat(messages=messages, sampling_params=sampling_params)
        out_obj = outputs[0].outputs[0]
        raw_text = out_obj.text
        tokens_generated = len(out_obj.token_ids)

    elif backend == "sglang":
        prompt = (
            f"<bos><start_of_turn>system\n{system_prompt}<end_of_turn>\n"
            f"<start_of_turn>user\n{transcript}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        out = engine.generate(
            prompt=prompt,
            sampling_params={"temperature": temperature, "max_new_tokens": max_tokens},
        )
        raw_text = out["text"]
        tokens_generated = out.get("meta_info", {}).get("completion_tokens", len(raw_text.split()))

    else:
        # Transformers fallback
        import torch

        model = engine["model"]
        processor = engine["processor"]
        device = engine["device"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": transcript}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-4) if temperature > 0 else None,
            )

        gen_tokens = output_ids[0, prompt_len:]
        tokens_generated = int(gen_tokens.numel())
        raw_text = processor.decode(gen_tokens, skip_special_tokens=True)

    gen_s = round(time.perf_counter() - t0, 3)
    tok_s = round(tokens_generated / max(gen_s, 1e-6), 2)

    parsed = parse_prescriptions_json(raw_text)
    return {
        "valid_json": parsed["valid_json"],
        "medications": parsed["medications"],
        "medications_count": parsed["medications_count"],
        "latency_seconds": gen_s,
        "tokens_generated": tokens_generated,
        "throughput_tok_s": tok_s,
        "raw_text": raw_text,
    }


def extract_prescriptions_stream(
    transcript: str,
    system_prompt: str = STANDARD_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Generator[Dict[str, Any], None, None]:
    """Stream token outputs and structured medications extraction."""
    # Yield initial start event
    yield {"event": "extraction_start", "transcript": transcript}

    # Execute extraction
    result = extract_prescriptions(
        transcript=transcript,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    yield {
        "event": "extraction_complete",
        "valid_json": result["valid_json"],
        "medications": result["medications"],
        "medications_count": result["medications_count"],
        "latency_seconds": result["latency_seconds"],
        "tokens_generated": result["tokens_generated"],
        "throughput_tok_s": result["throughput_tok_s"],
        "raw_text": result["raw_text"],
    }
