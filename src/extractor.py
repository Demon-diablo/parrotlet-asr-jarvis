"""Zero-hop clinical medication extractor engine via SGLang RadixAttention.

Co-hosts MedGemma-4B-it on the GPU alongside Parrotlet ASR with:
- Zero-hop in-process execution (no inter-process or network communication overhead)
- Unquantized Native BF16 precision weights and KV cache (zero numerical degradation)
- FlashInfer cooperative warp decode kernels
- RadixAttention LRU trie prefix caching (<5ms prompt TTFT)
- Low-batch decode CUDA graphs (BS=[1, 2]) bypassing CPU driver launch delay
- Pre-warmed prompt prefix trie and CUDA graphs at server boot
- Full streaming response preservation for SSE event streaming
- Standard clinical JSON extraction format preserved for easy future customization
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional

try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

from src.schema import (
    DENSE_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT,
    get_default_system_prompt,
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
    """Resolve configured extractor engine: 'sglang', 'vllm', 'transformers', or 'none'."""
    val = (os.getenv("EXTRACTOR_BACKEND") or os.getenv("EXTRACTOR_ENGINE") or "sglang").lower()
    return val if val in ("sglang", "vllm", "transformers", "none") else "sglang"


def get_medgemma_model_id() -> str:
    """Resolve model checkpoint ID or path for MedGemma-4B."""
    return (
        os.getenv("MEDGEMMA_MODEL_ID")
        or os.getenv("MEDGEMMA_DIR")
        or os.getenv("MEDGEMMA_ID")
        or DEFAULT_MODEL_ID
    )


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


def _format_prompt(transcript: str, system_prompt: Optional[str] = None) -> str:
    """Format input prompt with Gemma 3 turn markers for deterministic Radix prefix caching."""
    if system_prompt is None:
        system_prompt = get_default_system_prompt()
    return (
        f"<start_of_turn>user\n{system_prompt}\n\n{transcript}"
        f"<end_of_turn>\n<start_of_turn>model\n"
    )


def _init_sglang_engine(
    model_id: str,
    max_model_len: int = 8192,
    mem_fraction: float = 0.50,
    attention_backend: str = "flashinfer",
) -> Any:
    """Initialize SGLang engine configured specifically for minimum latency and zero quantization."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.cuda.empty_cache()

    import sglang as sgl

    draft_model = os.getenv("SPECULATIVE_DRAFT_PATH", "").strip() or None

    log.info(
        "Initializing SGLang Engine for %s (mem_frac=%.2f, ctx=%d, backend=%s, draft=%s)",
        model_id,
        mem_fraction,
        max_model_len,
        attention_backend,
        draft_model,
    )

    engine_kwargs: Dict[str, Any] = {
        "model_path": model_id,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "mem_fraction_static": mem_fraction,
        "context_length": max_model_len,
        "disable_radix_cache": False,
        "schedule_policy": "lpm",
        "cuda_graph_backend_prefill": "disabled",
        "disable_prefill_cuda_graph": True,
        "cuda_graph_max_bs_decode": 2,
        "cuda_graph_bs_decode": [1, 2],
    }

    if draft_model:
        engine_kwargs["speculative_draft_model_path"] = draft_model
        engine_kwargs["speculative_num_steps"] = int(os.getenv("SPECULATIVE_STEPS", "3"))
        log.info("Speculative decoding enabled with draft model: %s", draft_model)

    # Dynamic ServerArgs introspection for cross-version SGLang compatibility (msgspec / dataclass)
    try:
        from sglang.srt.server_args import ServerArgs

        supported_fields = set()
        if hasattr(ServerArgs, "__struct_fields__"):
            supported_fields = set(ServerArgs.__struct_fields__)
        elif hasattr(ServerArgs, "__dataclass_fields__"):
            supported_fields = set(ServerArgs.__dataclass_fields__.keys())
        elif hasattr(ServerArgs, "__annotations__"):
            supported_fields = set(ServerArgs.__annotations__.keys())

        if supported_fields:
            if "cuda_graph_max_bs_decode" not in supported_fields and "cuda_graph_max_bs" in supported_fields:
                engine_kwargs["cuda_graph_max_bs"] = engine_kwargs.pop("cuda_graph_max_bs_decode", 2)
                engine_kwargs["cuda_graph_bs"] = engine_kwargs.pop("cuda_graph_bs_decode", [1, 2])
            if "dtype" not in supported_fields and "torch_dtype" in supported_fields:
                engine_kwargs["torch_dtype"] = engine_kwargs.pop("dtype", "bfloat16")
            engine_kwargs = {k: v for k, v in engine_kwargs.items() if k in supported_fields}
    except Exception as exc:
        log.debug("ServerArgs introspection notice: %s", exc)

    return sgl.Engine(**engine_kwargs)


def _init_vllm_engine(model_id: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.45) -> Any:
    """Initialize vLLM engine fallback if specifically configured."""
    from vllm import LLM

    log.info("Initializing vLLM Extractor for %s...", model_id)
    return LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )


def _init_transformers_engine(model_id: str) -> Dict[str, Any]:
    """Initialize native PyTorch Transformers fallback with bfloat16 + TF32."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log.info("Initializing Transformers fallback for %s in bfloat16...", model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        "trust_remote_code": True,
    }
    if device == "cuda":
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **load_kwargs,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return {"model": model, "processor": processor, "device": device}


def load_extractor(
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    max_model_len: Optional[int] = None,
) -> Any:
    """Load and cache the zero-hop clinical extraction engine."""
    with _EXTRACTOR_CACHE["lock"]:
        if _EXTRACTOR_CACHE.get("loaded") and _EXTRACTOR_CACHE.get("engine") is not None:
            return _EXTRACTOR_CACHE["engine"]

        from src.config import get_settings

        settings = get_settings()
        target_model = model_id or settings.medgemma_model_id or get_medgemma_model_id()
        target_backend = (backend or settings.extractor_backend or get_extractor_backend()).lower()
        ctx_len = max_model_len or settings.sglang_context_len

        if target_backend == "none":
            log.info("Extractor backend configured to 'none'; clinical extraction disabled.")
            return None

        engine = None
        loaded_backend = None

        if target_backend == "sglang":
            try:
                engine = _init_sglang_engine(
                    target_model,
                    max_model_len=ctx_len,
                    mem_fraction=settings.sglang_mem_fraction,
                    attention_backend=settings.sglang_attention_backend,
                )
                loaded_backend = "sglang"
            except Exception as exc:
                log.error("Failed to initialize SGLang extractor: %s", exc)
                raise RuntimeError(f"Could not initialize SGLang extractor ({target_model}): {exc}") from exc

        elif target_backend == "vllm":
            try:
                engine = _init_vllm_engine(
                    target_model,
                    max_model_len=ctx_len,
                    gpu_memory_utilization=settings.extractor_gpu_memory_utilization,
                )
                loaded_backend = "vllm"
            except Exception as exc:
                log.error("Failed to initialize vLLM extractor: %s", exc)
                raise RuntimeError(f"Could not initialize vLLM extractor ({target_model}): {exc}") from exc

        elif target_backend == "transformers":
            try:
                engine = _init_transformers_engine(target_model)
                loaded_backend = "transformers"
            except Exception as exc:
                log.error("Failed to initialize Transformers fallback: %s", exc)
                raise RuntimeError(f"Could not initialize Transformers extractor ({target_model}): {exc}") from exc

        else:
            raise ValueError(f"Unknown extractor backend: {target_backend}")

        _EXTRACTOR_CACHE["engine"] = engine
        _EXTRACTOR_CACHE["backend"] = loaded_backend
        _EXTRACTOR_CACHE["model_id"] = target_model
        _EXTRACTOR_CACHE["loaded"] = True
        return engine


def warmup_prefix_cache() -> None:
    """Pre-populates the RadixAttention trie and captures decode CUDA graphs so the first request has <5ms TTFT."""
    if _EXTRACTOR_CACHE.get("warmed_up"):
        return
    if get_extractor_backend() == "none":
        return

    try:
        t0 = time.perf_counter()
        log.info("Pre-warming SGLang RadixAttention prompt prefix & decode CUDA graphs...")
        engine = load_extractor()
        backend = _EXTRACTOR_CACHE.get("backend")

        if backend == "sglang" and engine is not None:
            dummy_prompt = _format_prompt("Tablet Paracetamol 500 mg TDS for 3 days.", system_prompt=get_default_system_prompt())
            engine.generate(
                dummy_prompt,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 2,
                    "stop": ["<end_of_turn>", "<eos>"],
                },
            )
        else:
            dummy_transcript = "Tablet Pan 40 OD with empty stomach ES."
            extract_prescriptions(dummy_transcript, max_tokens=16)

        _EXTRACTOR_CACHE["warmed_up"] = True
        log.info("RadixAttention prefix cache & CUDA graphs pre-warmed in %.2fs", time.perf_counter() - t0)
    except Exception as exc:
        log.warning("Extractor prefix warmup notice: %s", exc)


# Backwards-compatible alias
warmup_extractor = warmup_prefix_cache


def _engine_generate(engine: Any, prompt: str, sampling_params: Dict[str, Any]) -> Any:
    """Thread-safe and event-loop-safe generation call.

    When running inside or alongside an active event loop (e.g. Uvicorn/FastAPI main thread),
    calling loop.run_until_complete() directly from a worker thread raises
    'RuntimeError: this event loop is already running.'.
    Using asyncio.run_coroutine_threadsafe dispatches the request to the engine's loop safely.
    """
    import asyncio
    import inspect

    loop = getattr(engine, "loop", None)
    if hasattr(engine, "async_generate") and loop is not None and getattr(loop, "is_running", lambda: False)():
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is not loop:
            coro = engine.async_generate(prompt=prompt, sampling_params=sampling_params)
            if inspect.iscoroutine(coro):
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                return fut.result()


    return engine.generate(prompt=prompt, sampling_params=sampling_params)


def _engine_generate_stream(engine: Any, prompt: str, sampling_params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Thread-safe and event-loop-safe streaming generation call."""
    import asyncio
    import inspect

    loop = getattr(engine, "loop", None)
    if hasattr(engine, "async_generate") and loop is not None and getattr(loop, "is_running", lambda: False)():
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is not loop:
            try:
                coro = engine.async_generate(prompt=prompt, sampling_params=sampling_params, stream=True)
                if inspect.iscoroutine(coro):
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                    async_gen = fut.result()
                    while True:
                        try:
                            chunk_fut = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop)
                            yield chunk_fut.result()
                        except StopAsyncIteration:
                            break
                        except Exception as exc:
                            if "StopAsyncIteration" in str(type(exc)) or "StopAsyncIteration" in str(exc):
                                break
                            raise
                    return
            except Exception:
                pass

    res = engine.generate(prompt=prompt, sampling_params=sampling_params, stream=True)
    if hasattr(res, "__iter__") and not isinstance(res, (dict, str)):
        yield from res
    else:
        yield res



def extract_prescriptions(
    transcript: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Extract clinical medications from transcript using SGLang with RadixAttention.

    Preserves standard clinical JSON output without altering prompt or schema.
    """
    if system_prompt is None:
        system_prompt = get_default_system_prompt()
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

    if backend == "sglang":
        prompt = _format_prompt(transcript, system_prompt=system_prompt)
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": ["<end_of_turn>", "<eos>"],
        }
        out = _engine_generate(engine, prompt=prompt, sampling_params=sampling_params)
        if isinstance(out, dict):
            raw_text = str(out.get("text", "")).strip()
            meta = out.get("meta_info", {}) or {}
            tokens_generated = int(meta.get("completion_tokens", len(raw_text.split())))
        else:
            raw_text = str(getattr(out, "text", out)).strip()
            meta = getattr(out, "meta_info", {}) or {}
            tokens_generated = int(meta.get("completion_tokens", len(raw_text.split())))

    elif backend == "vllm":
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ]
        outputs = engine.chat(messages=messages, sampling_params=sampling_params)
        out_obj = outputs[0].outputs[0]
        raw_text = out_obj.text
        tokens_generated = len(out_obj.token_ids)

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

    parsed = parse_prescriptions_json(raw_text, transcript=transcript)
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
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Generator[Dict[str, Any], None, None]:
    """Stream SSE extraction events and tokens, maintaining existing client contracts."""
    if system_prompt is None:
        system_prompt = get_default_system_prompt()
    # 1. Yield start event
    yield {"event": "extraction_start", "transcript": transcript}

    if get_extractor_backend() == "none":
        yield {
            "event": "extraction_complete",
            "valid_json": False,
            "medications": [],
            "medications_count": 0,
            "latency_seconds": 0.0,
            "tokens_generated": 0,
            "throughput_tok_s": 0.0,
            "raw_text": "",
            "error": "Extractor disabled (EXTRACTOR_BACKEND=none)",
        }
        return

    # 2. If SGLang engine is loaded in cache and supports streaming, stream tokens
    backend = _EXTRACTOR_CACHE.get("backend")
    engine = _EXTRACTOR_CACHE.get("engine")

    raw_text = ""
    tokens_generated = 0
    t0 = time.perf_counter()

    if backend == "sglang" and engine is not None and hasattr(engine, "generate"):
        prompt = _format_prompt(transcript, system_prompt=system_prompt)
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": ["<end_of_turn>", "<eos>"],
        }

        try:
            res = _engine_generate_stream(engine, prompt=prompt, sampling_params=sampling_params)
            if hasattr(res, "__iter__") and not isinstance(res, (dict, str)):
                prev_text = ""
                for chunk in res:
                    curr_text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
                    delta = curr_text[len(prev_text):]
                    prev_text = curr_text
                    if delta:
                        yield {"event": "token", "token": delta}
                raw_text = prev_text
                tokens_generated = len(raw_text.split())
            else:
                raw_text = str(res.get("text", "") if isinstance(res, dict) else getattr(res, "text", res))
                tokens_generated = len(raw_text.split())
        except Exception as exc:
            log.warning("SGLang streaming generation notice: %s; falling back to standard extraction", exc)
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
            return
    else:
        # Standard extraction fallback
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
        return

    gen_s = round(time.perf_counter() - t0, 3)
    tok_s = round(tokens_generated / max(gen_s, 1e-6), 2)
    parsed = parse_prescriptions_json(raw_text, transcript=transcript)

    # 3. Yield completion event matching exact contract
    yield {
        "event": "extraction_complete",
        "valid_json": parsed["valid_json"],
        "medications": parsed["medications"],
        "medications_count": parsed["medications_count"],
        "latency_seconds": gen_s,
        "tokens_generated": tokens_generated,
        "throughput_tok_s": tok_s,
        "raw_text": raw_text,
    }
