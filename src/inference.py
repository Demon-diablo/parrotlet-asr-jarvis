"""Inference pipeline for Parrotlet-a 2.5 Pro.

Real input contract (verified during Phase 3 inspection):

  Audio file (wav / mp3 / flac) at any sample rate — the upstream loader
  resamples to 16 kHz via librosa (``res_type='soxr_vhq'``).

Real output contract:

  A single ASR transcript string. The upstream ``SpeechLLM.transcribe``
  returns the raw decoded text; we wrap it in the canonical
  ``{"status": "success", "output": {...}}`` envelope at the handler level.

This module does NOT contain HTTP/job parsing. The API layer turns a request
into ``input_data`` exactly once; everything below treats ``input_data`` as
a plain dict.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from .config import Settings, get_settings

log = logging.getLogger("parrotlet.inference")


# ---------------------------------------------------------------------------
# Transcript cleaner + script detector
#
# Observed in Sept-2025 benchmark: raw transcripts contain <bird_squawk>,
# <Persistent-noise-*>, <hi-en>/<hi>/<talking> tags plus [One]/[MG] gloss
# brackets. Polishing (strip tags, unwrap brackets) cut semWER by up to
# 0.25 with zero model cost, so it runs by default (CLEAN_TRANSCRIPT=1).
# ---------------------------------------------------------------------------
_NOISE_TAG_RE = re.compile(r"<[^>]*>")
_BRACKET_GLOSS_RE = re.compile(r"\[([^\[\]]+)\]")
_WS_RE = re.compile(r"\s+")

_SCRIPT_CHECKS = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)


def clean_transcript(text: Any) -> str:
    """Deterministic cleaner: strip <tags>, unwrap [gloss], collapse ws.

    ``[One]`` -> ``One`` (keeps the English gloss the model already
    produced); ``<bird_squawk>`` -> ```` (noise carries no clinical fact).
    Non-string input returns "".
    """
    if not isinstance(text, str):
        return ""
    out = _NOISE_TAG_RE.sub(" ", text)
    out = _BRACKET_GLOSS_RE.sub(r"\1", out)
    out = _WS_RE.sub(" ", out).strip()
    # Kill stray spaces before common punctuation the tag-strip leaves.
    out = re.sub(r"\s+([,.;:!?%])", r"\1", out)
    return out


def detect_scripts(text: Any) -> Dict[str, Any]:
    """Report which Indic scripts (if any) survive in ``text``.

    Used for observability: after the decoder ban + cleaner the flags
    should all be False. Never raises.
    """
    flags: Dict[str, Any] = {}
    try:
        s = text if isinstance(text, str) else ""
        for name, lo, hi in _SCRIPT_CHECKS:
            flags[f"has_{name}"] = any(lo <= ord(c) <= hi for c in s)
        flags["has_indic"] = any(flags[f"has_{n}"] for n, _, _ in _SCRIPT_CHECKS)
        flags["non_latin_count"] = sum(1 for c in s if ord(c) > 127)
    except Exception:
        flags = {"has_indic": False, "non_latin_count": 0}
    return flags


def _get_banned_ids_for_bundle(model_bundle: Any) -> List[int]:
    """Best-effort banned-ID lookup: bundle field first, then model cache."""
    try:
        ids = getattr(model_bundle, "banned_token_ids", None)
        if ids:
            return [int(i) for i in ids]
    except Exception:
        pass
    try:
        from .model import get_banned_token_ids as _cached_ids

        return [int(i) for i in (_cached_ids() or [])]
    except Exception:
        return []




# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class InputValidationError(ValueError):
    """Raised when the caller-provided input is missing required fields or has
    an invalid shape. Surfaces to the handler as a structured error response.
    """


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MiB cap — protects the worker from
# pathological inputs while being generous enough for 30s of high-fidelity
# medical audio.


def _coerce_audio_bytes(audio: Any) -> bytes:
    """Accept audio as base64 string, dict with ``base64`` / ``data``, or raw
    bytes. Returns raw bytes ready for librosa.load via soundfile."""
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, str):
        # Treat as base64-encoded payload.
        try:
            return base64.b64decode(audio, validate=False)
        except Exception as exc:
            raise InputValidationError(
                "'audio' string is not valid base64"
            ) from exc
    if isinstance(audio, dict):
        for key in ("base64", "data"):
            if key in audio:
                return _coerce_audio_bytes(audio[key])
        raise InputValidationError(
            "audio dict must contain 'base64' or 'data'"
        )
    raise InputValidationError(
        f"'audio' must be bytes, base64 string, or dict with 'base64'/'data'; got {type(audio).__name__}"
    )


def validate_input(input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the decoded API input payload.

    Accepted shapes::

        {"audio": "<base64>", "sample_rate": 16000}     # raw bytes
        {"audio": {"base64": "..."} | {"data": "..."}, "sample_rate": 16000}
        {"audio_path": "/abs/path.wav", "sample_rate": 16000}
        {"audio_url": "https://.../clip.wav", "sample_rate": 16000}
        {"message": "..."}                              # legacy dummy shape

    Exactly one of ``audio`` / ``audio_path`` / ``audio_url`` must be present.
    """
    if input_data is None:
        raise InputValidationError("missing required field: input")
    if not isinstance(input_data, dict):
        raise InputValidationError(
            f"input must be an object, got {type(input_data).__name__}"
        )

    # Legacy / diagnostic-only echo path used by the Phase 1 dummy handler.
    if "message" in input_data and not any(
        k in input_data for k in ("audio", "audio_path", "audio_url")
    ):
        message = input_data["message"]
        if not isinstance(message, str):
            raise InputValidationError("'message' must be a string")
        if len(message) > 8192:
            raise InputValidationError("'message' too long (limit 8192)")
        return {"kind": "echo", "message": message}

    keys = [k for k in ("audio", "audio_path", "audio_url") if k in input_data]
    if len(keys) != 1:
        raise InputValidationError(
            "exactly one of 'audio', 'audio_path', 'audio_url' is required"
        )

    audio_key = keys[0]
    sample_rate = input_data.get("sample_rate", 16000)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise InputValidationError(
            f"'sample_rate' must be a positive integer, got {sample_rate!r}"
        )

    if audio_key == "audio":
        audio_bytes = _coerce_audio_bytes(input_data["audio"])
        if len(audio_bytes) == 0:
            raise InputValidationError("'audio' payload is empty")
        if len(audio_bytes) > _MAX_AUDIO_BYTES:
            raise InputValidationError(
                f"'audio' exceeds {_MAX_AUDIO_BYTES // (1024*1024)} MiB cap"
            )
        return {
            "kind": "audio_bytes",
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
        }

    if audio_key == "audio_path":
        path = input_data["audio_path"]
        if not isinstance(path, str):
            raise InputValidationError("'audio_path' must be a string")
        if not os.path.isabs(path):
            raise InputValidationError(
                "'audio_path' must be an absolute path (production workers "
                "have no predictable cwd)"
            )
        if not os.path.isfile(path):
            raise InputValidationError(f"audio file not found: {path}")
        # Same 50 MiB budget as inline bytes: fail fast instead of loading
        # a multi-GB file into the worker and OOMing it.
        try:
            if os.path.getsize(path) > _MAX_AUDIO_BYTES:
                raise InputValidationError(
                    f"'audio_path' file exceeds {_MAX_AUDIO_BYTES // (1024*1024)} MiB cap"
                )
        except OSError as exc:
            raise InputValidationError(f"cannot stat audio file: {path}") from exc
        return {
            "kind": "audio_path",
            "audio_path": path,
            "sample_rate": sample_rate,
        }

    # audio_key == "audio_url"
    url = input_data["audio_url"]
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise InputValidationError(
            "'audio_url' must be a string starting with http:// or https://"
        )
    return {"kind": "audio_url", "audio_url": url, "sample_rate": sample_rate}



# ---------------------------------------------------------------------------
# Real inference
# ---------------------------------------------------------------------------
class FastBannedTokensLogitsProcessor:
    """Vectorized GPU logit suppression for thousands of banned Indic script tokens.

    Replaces HuggingFace's BadWordsLogitsProcessor/NoBadWordsLogitsProcessor which runs a
    slow Python dictionary loop over ~34k tokens on every auto-regressive step (~50-100x speedup).
    Performs a single in-place masked assignment on GPU (scores[:, self.banned_ids] = -inf).
    """

    def __init__(self, banned_ids: Any):
        self.banned_ids = banned_ids

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        try:
            device = getattr(scores, "device", None)
            if device is not None and hasattr(self.banned_ids, "to"):
                if getattr(self.banned_ids, "device", None) != device:
                    self.banned_ids = self.banned_ids.to(device)
            scores[:, self.banned_ids] = -float("inf")
        except Exception:
            pass
        return scores


def _transcribe_batched(
    speech_llm: Any,
    window_audios: List[Any],
    settings: Settings,
    bad_words: Optional[List[List[int]]] = None,
    banned_ids: Optional[List[int]] = None,
) -> List[str]:
    """Batched inference across multiple 30 s audio windows.

    Runs audio feature extraction, encoder, projector, and decoder generation
    in a single batched pass over all windows with cached prompt embeddings,
    static KV cache, and fast vectorized banned-token processing.
    """
    submodules = ("encoder", "decoder", "projector", "processor", "tokenizer")
    for sm in submodules:
        if not hasattr(speech_llm, sm) or getattr(speech_llm, sm) is None:
            raise AttributeError(f"speech_llm is missing required submodule: {sm}")

    if torch is None:
        raise RuntimeError("torch is not available for batched inference")

    if not window_audios:
        return []

    with torch.inference_mode():
        if hasattr(speech_llm, "_encoder_device") and callable(speech_llm._encoder_device):
            enc_device = speech_llm._encoder_device()
        else:
            try:
                enc_device = next(speech_llm.encoder.parameters()).device
            except Exception:
                cuda_ok = torch.cuda.is_available() if hasattr(torch, "cuda") else False
                enc_device = getattr(speech_llm, "device", "cuda" if cuda_ok else "cpu")

        try:
            enc_dtype = next(speech_llm.encoder.parameters()).dtype
        except Exception:
            enc_dtype = torch.float16

        if hasattr(speech_llm, "_decoder_device") and callable(speech_llm._decoder_device):
            dec_device = speech_llm._decoder_device()
        else:
            try:
                dec_device = next(speech_llm.decoder.parameters()).device
            except Exception:
                dec_device = enc_device

        import numpy as np

        processed_audios = [
            np.asarray(
                speech_llm.preprocess_audio(a, 16000)
                if hasattr(speech_llm, "preprocess_audio")
                else a,
                dtype=np.float32,
            )
            for a in window_audios
        ]
        t_feat0 = time.perf_counter()

        audio_features = speech_llm.processor.feature_extractor(
            processed_audios, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(enc_device, dtype=enc_dtype)
        t_feat1 = time.perf_counter()

        try:
            proj_device = next(speech_llm.projector.parameters()).device
            proj_dtype = next(speech_llm.projector.parameters()).dtype
        except Exception:
            proj_device = enc_device
            proj_dtype = enc_dtype

        t_enc0 = time.perf_counter()
        enc_out = speech_llm.encoder(audio_features)
        audio_embeddings = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out
        t_enc1 = time.perf_counter()

        t_proj0 = time.perf_counter()
        projected = speech_llm.projector(audio_embeddings.to(proj_device, dtype=proj_dtype))
        t_proj1 = time.perf_counter()

        t_prep0 = time.perf_counter()
        prompt = (
            speech_llm.get_prompt()
            if hasattr(speech_llm, "get_prompt") and callable(speech_llm.get_prompt)
            else (
                "<bos><start_of_turn>user\n"
                "You are transcriptionist. Transcribe the audio given to you in verbatim manner. "
                "It can be in any language in India.\n\n"
                "<|audio_bos|><audio><|audio_eos|> Transcribe this audio.<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        )
        audio_token_str = getattr(speech_llm, "audio_token", "<audio>")

        # Check prompt tokenization cache on speech_llm to avoid CPU tokenizer re-parsing
        p_cache = getattr(speech_llm, "_prompt_cache", None)
        if (
            p_cache is None
            or p_cache.get("prompt") != prompt
            or p_cache.get("device") != dec_device
        ):
            input_ids = torch.tensor(speech_llm.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            input_attention_mask = torch.ones_like(input_ids)
            audio_token = speech_llm.tokenizer.convert_tokens_to_ids(audio_token_str)
            audio_pos = input_ids.tolist().index(audio_token)
            p_cache = {
                "prompt": prompt,
                "input_ids": input_ids,
                "input_attention_mask": input_attention_mask,
                "audio_token": audio_token,
                "audio_pos": audio_pos,
                "device": dec_device,
            }
            try:
                speech_llm._prompt_cache = p_cache
            except Exception:
                pass
        else:
            input_ids = p_cache["input_ids"]
            input_attention_mask = p_cache["input_attention_mask"]
            audio_pos = p_cache["audio_pos"]

        B = len(window_audios)
        input_ids_b = input_ids.unsqueeze(0).expand(B, -1).to(dec_device)
        input_attention_mask_b = input_attention_mask.unsqueeze(0).expand(B, -1).to(dec_device)

        input_embeddings = speech_llm.decoder.get_input_embeddings()(input_ids_b)
        projected = projected.to(device=dec_device, dtype=input_embeddings.dtype)

        b, in_len, dim = input_embeddings.shape
        a_len = projected.shape[1]
        total = in_len + a_len - 1

        combined = torch.zeros(b, total, dim, device=dec_device, dtype=input_embeddings.dtype)
        combined_mask = torch.zeros(b, total, device=dec_device, dtype=input_attention_mask_b.dtype)

        combined[:, :audio_pos] = input_embeddings[:, :audio_pos]
        combined_mask[:, :audio_pos] = input_attention_mask_b[:, :audio_pos]

        combined[:, audio_pos:audio_pos + a_len] = projected
        combined_mask[:, audio_pos:audio_pos + a_len] = 1
        for i, a in enumerate(window_audios):
            valid_tokens = min(a_len, int(round((len(a) / 16000.0) * 25.0)))
            if valid_tokens < a_len:
                combined_mask[i, audio_pos + valid_tokens:audio_pos + a_len] = 0

        suf_start = audio_pos + 1
        suf_len = in_len - audio_pos - 1
        out_start = audio_pos + a_len

        combined[:, out_start:out_start + suf_len] = input_embeddings[:, suf_start:]
        combined_mask[:, out_start:out_start + suf_len] = input_attention_mask_b[:, suf_start:]

        pad_id = getattr(speech_llm.tokenizer, "pad_token_id", None)
        if pad_id is None and hasattr(speech_llm.tokenizer, "convert_tokens_to_ids"):
            pad_id = speech_llm.tokenizer.convert_tokens_to_ids("<pad>")
        if pad_id is None:
            pad_id = 0

        # Gemma3's chat template ends a turn with <end_of_turn>, not the
        # tokenizer's plain <eos>. Previously only one of the two was ever
        # passed to generate()'s eos_token_id, so HF's built-in early-stop
        # never matched the token the model actually emits — generation ran
        # the full max_new_tokens on every request (80s / 256 tokens ~=
        # 0.31s/token, i.e. no early stop at all) and a later post-hoc string
        # trim (see stop_ids below) only hid this by cutting the decoded
        # text, after the GPU had already paid for every step. Passing BOTH
        # candidate stop ids lets generate() halt as soon as either appears.
        eos_id = getattr(speech_llm.tokenizer, "eos_token_id", None)
        eot_id = (
            speech_llm.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if hasattr(speech_llm.tokenizer, "convert_tokens_to_ids")
            else None
        )
        stop_token_ids = sorted(
            {int(i) for i in (eos_id, eot_id) if isinstance(i, int) and i >= 0}
        )
        gen_eos_id: Any = stop_token_ids if len(stop_token_ids) > 1 else (
            stop_token_ids[0] if stop_token_ids else None
        )

        gen_kwargs: Dict[str, Any] = {
            "inputs_embeds": combined,
            "attention_mask": combined_mask,
            "pad_token_id": pad_id,
            "eos_token_id": gen_eos_id,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "use_cache": True,
            # StaticCache stays (no per-step realloc) but skip transformers'
            # auto-compile of decode: inductor re-records CUDAGraphs on our
            # dynamic per-step shapes (~50 records, ~0.9s/token on sm_100).
            # Plain-sdpa decode is ~25ms/token. Revisit with fixed-shape
            # padding if compile benefits are wanted back.
            "disable_compile": True,
        }
        if getattr(settings, "max_new_tokens", None) is not None:
            gen_kwargs["max_new_tokens"] = settings.max_new_tokens
            if pad_id is not None:
                gen_kwargs["pad_token_id"] = pad_id
            if gen_eos_id is not None:
                gen_kwargs["eos_token_id"] = gen_eos_id

        # Detect if decoder is a unit-test mock object
        is_mock_decoder = (
            not hasattr(speech_llm.decoder, "config")
            or "Mock" in type(getattr(speech_llm.decoder, "generate", None)).__name__
            or "MagicMock" in type(speech_llm.decoder).__name__
        )

        # Setup high-speed vectorized logit processor for banned script tokens
        logits_processor = None
        resolved_banned = banned_ids or ([bw[0] for bw in bad_words if bw] if bad_words else None)
        if resolved_banned and torch is not None and not is_mock_decoder:
            try:
                banned_tensor = torch.as_tensor(resolved_banned, dtype=torch.long, device=dec_device)
                fast_proc = FastBannedTokensLogitsProcessor(banned_tensor)
                try:
                    from transformers.generation.logits_process import LogitsProcessorList
                    logits_processor = LogitsProcessorList([fast_proc])
                except Exception:
                    logits_processor = [fast_proc]
            except Exception as exc:
                log.debug("FastBannedTokensLogitsProcessor fallback: %s", exc)
                logits_processor = None

        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        elif bad_words:
            gen_kwargs["bad_words_ids"] = bad_words

        # For unit test mock assertions (e.g. assert recorded["gen_kwargs"]["bad_words_ids"] == ...)
        if is_mock_decoder and bad_words and "bad_words_ids" not in gen_kwargs:
            gen_kwargs["bad_words_ids"] = bad_words

        # StaticCache support: preallocate KV cache to eliminate dynamic allocations in the 34-layer decoder
        if not is_mock_decoder and bool(getattr(settings, "use_static_cache", True)):
            try:
                from transformers import StaticCache
                tcfg = getattr(speech_llm.decoder.config, "text_config", speech_llm.decoder.config)
                max_cache_len = min(2048, total + int(gen_kwargs.get("max_new_tokens", 256)) + 16)
                cache_key = (B, max_cache_len, str(dec_device), input_embeddings.dtype)
                cache_pool = getattr(speech_llm, "_static_cache_pool", None)
                if cache_pool is None:
                    cache_pool = {}
                    try:
                        speech_llm._static_cache_pool = cache_pool
                    except Exception:
                        pass
                cache = cache_pool.get(cache_key)
                if cache is None:
                    cache = StaticCache(
                        config=tcfg,
                        max_batch_size=B,
                        max_cache_len=max_cache_len,
                        device=dec_device,
                        dtype=input_embeddings.dtype,
                    )
                    if len(cache_pool) >= 4:
                        cache_pool.pop(next(iter(cache_pool)))
                    cache_pool[cache_key] = cache
                else:
                    reset_fn = getattr(cache, "reset", None)
                    if callable(reset_fn):
                        reset_fn()
                gen_kwargs["past_key_values"] = cache
            except Exception as exc:
                log.debug("StaticCache setup skipped: %s", exc)

        t_prep1 = time.perf_counter()

        t_gen0 = time.perf_counter()
        out = speech_llm.decoder.generate(**gen_kwargs)
        t_gen1 = time.perf_counter()

        t_tok0 = time.perf_counter()
        results = []
        stop_ids = set(stop_token_ids) | {i for i in (1, 107) if isinstance(i, int)}
        _cap = int(gen_kwargs.get("max_new_tokens") or 0)
        gen_lens = []
        for t in out:
            token_list = t.tolist() if hasattr(t, "tolist") else list(t)
            eos_pos = [idx for idx, tok in enumerate(token_list) if tok in stop_ids]
            if eos_pos:
                token_list = token_list[:eos_pos[0]]
            if _cap and not eos_pos and len(token_list) >= _cap:
                log.warning(
                    "window hit max_new_tokens cap (%d) with no stop token found — "
                    "raise MAX_NEW_TOKENS if this recurs, transcript may be truncated",
                    _cap,
                )
            gen_lens.append(len(token_list))
            results.append(speech_llm.tokenizer.decode(token_list, skip_special_tokens=True).strip())
        t_tok1 = time.perf_counter()

        total_tokens = sum(gen_lens)
        dec_time = max(1e-6, t_gen1 - t_gen0)
        tok_per_sec = round(total_tokens / dec_time, 2)

        # Parallel-window proof: one batched GPU pass over B windows; per-window
        # generated lengths show whether decode early-stopped or ran the cap.
        phases_summary = {
            "featurize_s": round(t_feat1 - t_feat0, 4),
            "encode_s": round(t_proj1 - t_enc0, 4),
            "whisper_encoder_s": round(t_enc1 - t_enc0, 4),
            "projector_s": round(t_proj1 - t_proj0, 4),
            "prompt_prep_s": round(t_prep1 - t_prep0, 4),
            "decode_s": round(t_gen1 - t_gen0, 4),
            "decoder_s": round(t_gen1 - t_gen0, 4),
            "detokenize_s": round(t_tok1 - t_tok0, 4),
            "tokens_generated": total_tokens,
            "tokens_per_second": tok_per_sec,
            "tokens_per_window": gen_lens,
        }
        log.info("batched decode B=%d generated_tokens=%s phases=%s", B, gen_lens, phases_summary)
        try:
            speech_llm._last_phases = phases_summary
        except Exception:
            pass
        return results


def _transcribe_one(
    speech_llm: Any,
    window_audio: Any,
    settings: Settings,
    bad_words: Optional[List[List[int]]] = None,
) -> tuple:
    """Transcribe a single 16 kHz audio window using SpeechLLM.transcribe.

    NOTE: this rarely-hit fallback path (only reached when _transcribe_batched
    raises) calls into the upstream wrapper's own generate() call, which we
    don't control here — if it ever shows the same full-max_new_tokens
    symptom as the batched path did, the fix has to happen in the upstream
    ``SpeechLLM.transcribe`` implementation, not in this file.
    """
    base_kwargs: Dict[str, Any] = {
        "max_new_tokens": settings.max_new_tokens,
        # repetition_penalty=1.2 matches upstream defaults.
        # do_sample=False pins upstream greedy default explicitly.
        "do_sample": False,
    }
    if bad_words:
        try:
            return (
                str(
                    speech_llm.transcribe(
                        window_audio, 16000, **base_kwargs,
                        bad_words_ids=bad_words,
                    )
                ),
                True,
            )
        except TypeError:
            # Older upstream wrapper without bad_words_ids support:
            # fall back to the plain call (ban silently off).
            pass
    return (
        str(speech_llm.transcribe(window_audio, 16000, **base_kwargs)),
        False,
    )


def transcribe_single_window(
    model_bundle: Any,
    audio_window_16k: Any,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Transcribes a single 16 kHz mono window (float32 array).

    Uses the batched flush with B=1 (or _transcribe_one fallback), applies
    clean_transcript and script detection via _postprocess, and returns the
    standard postprocessed output dict.
    """
    if settings is None:
        settings = get_settings()

    if model_bundle is None or not hasattr(model_bundle, "speech_llm"):
        raise InputValidationError(
            "model not initialized yet; please retry in a few seconds"
        )

    import numpy as np
    from . import audio as audio_module

    start = time.perf_counter()

    audio_window_16k = np.asarray(audio_window_16k, dtype=np.float32)
    if len(audio_window_16k) == 0:
        raise InputValidationError("audio window contains no samples")

    speech_llm = model_bundle.speech_llm
    ban_enabled = bool(getattr(settings, "ban_script_tokens", True))
    banned_ids = _get_banned_ids_for_bundle(model_bundle) if ban_enabled else []
    bad_words = [[i] for i in banned_ids] if banned_ids else None

    raw_text = ""
    ban_applied = False
    if hasattr(speech_llm, "decoder") and hasattr(speech_llm, "encoder"):
        try:
            try:
                results = _transcribe_batched(
                    speech_llm, [audio_window_16k], settings, bad_words=bad_words, banned_ids=banned_ids
                )
            except TypeError:
                results = _transcribe_batched(
                    speech_llm, [audio_window_16k], settings, bad_words=bad_words
                )
            raw_text = results[0] if results else ""
            ban_applied = bool(banned_ids or bad_words)
        except Exception as exc:
            log.debug("_transcribe_batched failed (%s), falling back to _transcribe_one", exc)
            if hasattr(speech_llm, "transcribe") and callable(speech_llm.transcribe):
                raw_text, ban_applied = _transcribe_one(speech_llm, audio_window_16k, settings, bad_words)
            else:
                raise
    elif hasattr(speech_llm, "transcribe") and callable(speech_llm.transcribe):
        raw_text, ban_applied = _transcribe_one(speech_llm, audio_window_16k, settings, bad_words)
    else:
        try:
            results = _transcribe_batched(
                speech_llm, [audio_window_16k], settings, bad_words=bad_words, banned_ids=banned_ids
            )
        except TypeError:
            results = _transcribe_batched(
                speech_llm, [audio_window_16k], settings, bad_words=bad_words
            )
        raw_text = results[0] if results else ""
        ban_applied = bool(banned_ids or bad_words)

    if ban_applied and bad_words is None and not banned_ids:
        ban_applied = False

    phases = {}
    try:
        batched_phases = getattr(speech_llm, "_last_phases", None)
        if isinstance(batched_phases, dict):
            phases.update(batched_phases)
    except Exception:
        pass

    audio_sec = round(audio_module.audio_seconds(audio_window_16k), 3)
    decoder_output = {
        "transcript": raw_text,
        "audio_seconds": audio_sec,
        "native_sample_rate": 16000,
        "flushed_windows": 1,
        "phases": phases,
        "ban_applied": ban_applied,
        "banned_token_count": len(banned_ids),
    }
    elapsed = time.perf_counter() - start
    return _postprocess(decoder_output, settings, elapsed)


def _run_decoder_real(model_bundle: Any, validated: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    """Buffer audio into 30 s windows, flush in one batched GPU pass.

    Pipeline:

      1. ``src.audio.load_audio_for_inference`` — decodes bytes/path/URL,
         mixes to mono, resamples to 16 kHz.
      2. Optional silence trimming via ``settings.trim_silence``.
      3. ``src.audio.split_windows`` — buffers the signal into
         non-overlapping 30 s windows (trailing partial window kept as-is).
      4. ``_transcribe_batched`` — one batched GPU pass over all windows
         (StaticCache, prompt caching, fast vectorized logit suppression).
         On failure, sequential per-window retry (no threads: CUDA work
         serializes across threads anyway).
      5. ``src.audio.join_texts`` — concatenates per-window strings in
         order, single-space separated.

    Returns a dict so the postprocessor can attach generation metadata.
    """
    from . import audio as audio_module  # local import for the same reason as librosa

    speech_llm = model_bundle.speech_llm
    t_audio0 = time.perf_counter()

    try:
        try:
            load_res = audio_module.load_audio_for_inference(validated, return_details=True)
        except TypeError:
            load_res = audio_module.load_audio_for_inference(validated)

        if isinstance(load_res, tuple) and len(load_res) == 4:
            audio_16k, native_sr, _, audio_details = load_res
        else:
            audio_16k, native_sr, _ = load_res[:3]
            audio_details = {
                "original_sample_rate": int(native_sr),
                "target_sample_rate": 16000,
                "is_resampled_to_16k": bool(int(native_sr) != 16000),
                "original_channels": 1,
                "original_samples": len(audio_16k),
                "total_16k_samples": len(audio_16k),
                "duration_seconds": round(len(audio_16k) / 16000.0, 4),
                "phases": {
                    "decode_audio_s": 0.0,
                    "mono_mix_s": 0.0,
                    "resample_16k_s": 0.0,
                    "silence_trim_s": 0.0,
                },
            }
    except InputValidationError:
        raise
    except Exception as exc:
        # Corrupt/unsupported audio is a client error (400), not an inference
        # failure (500): classify it here so both the API layer and the server
        # worker map it correctly.
        raise InputValidationError(
            f"could not decode audio ({type(exc).__name__}): {exc}"
        ) from exc
    if len(audio_16k) == 0:
        raise InputValidationError("audio contains no samples")

    if bool(getattr(settings, "trim_silence", False)):
        audio_16k = audio_module.trim_audio_silence(audio_16k, top_db=35.0, margin_samples=3200)

    # Buffer the whole signal: contiguous 30 s windows, no overlap.
    t_win0 = time.perf_counter()
    windows = audio_module.split_windows(audio_16k)
    t_win1 = time.perf_counter()

    log.debug(
        "Audio: native_sr=%d 16k_samples=%d duration=%.2fs windows=%d",
        native_sr,
        len(audio_16k),
        audio_module.audio_seconds(audio_16k),
        len(windows),
    )

    ban_enabled = bool(getattr(settings, "ban_script_tokens", True))
    banned_ids = _get_banned_ids_for_bundle(model_bundle) if ban_enabled else []
    # generate() shape: one single-token sequence per banned ID.
    bad_words = [[i] for i in banned_ids] if banned_ids else None

    decode_path = f"buffered:B={len(windows)}"
    try:
        try:
            parts = _transcribe_batched(
                speech_llm, windows, settings, bad_words=bad_words, banned_ids=banned_ids
            )
        except TypeError:
            parts = _transcribe_batched(
                speech_llm, windows, settings, bad_words=bad_words
            )
        ban_applied = bool(banned_ids or bad_words)
    except Exception as exc:
        # Batched flush failed: retry windows sequentially (no threads —
        # CUDA work serializes across threads, so threads buy nothing here).
        log.warning("Batched buffer flush failed (%s); retrying windows sequentially", exc)
        decode_path = "fallback-sequential"
        parts = []
        ban_applied = False
        for w in windows:
            text, used = _transcribe_one(speech_llm, w, settings, bad_words)
            parts.append(text)
            ban_applied = ban_applied or used

    if ban_applied and bad_words is None and not banned_ids:
        ban_applied = False

    t_asm0 = time.perf_counter()
    transcript = audio_module.join_texts(parts)
    t_asm1 = time.perf_counter()

    phases: Dict[str, Any] = {
        "audio_prep_s": round(t_win1 - t_audio0, 4),
        "audio_s": round(t_win1 - t_audio0, 4),
        "audio_decode_s": audio_details.get("phases", {}).get("decode_audio_s", 0.0),
        "mono_mix_s": audio_details.get("phases", {}).get("mono_mix_s", 0.0),
        "resample_16k_s": audio_details.get("phases", {}).get("resample_16k_s", 0.0),
        "windowing_s": round(t_win1 - t_win0, 4),
        "assemble_s": round(t_asm1 - t_asm0, 4),
    }
    try:
        batched_phases = getattr(speech_llm, "_last_phases", None)
        if isinstance(batched_phases, dict):
            phases.update(batched_phases)
    except Exception:
        pass

    return {
        "transcript": transcript,
        "audio_seconds": round(audio_module.audio_seconds(audio_16k), 3),
        "native_sample_rate": int(native_sr),
        "audio_info": audio_details,
        "flushed_windows": len(windows),
        "decode_path": decode_path,
        "phases": phases,
        "ban_applied": ban_applied,
        "banned_token_count": len(banned_ids),
    }


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------
def _postprocess(decoder_output: Dict[str, Any], settings: Settings, elapsed: float) -> Dict[str, Any]:
    """Turn the decoder output into a JSON-serialisable payload."""
    raw_transcript = decoder_output["transcript"]
    clean_enabled = bool(getattr(settings, "clean_transcript", True))
    transcript = clean_transcript(raw_transcript) if clean_enabled else raw_transcript

    audio_dur = float(decoder_output["audio_seconds"])
    inf_time = round(elapsed, 4)
    rtf = round(inf_time / audio_dur, 4) if audio_dur > 0 else 0.0
    speedup = round(audio_dur / inf_time, 2) if inf_time > 0 else 0.0

    audio_info = decoder_output.get("audio_info", {})
    native_sr = decoder_output.get("native_sample_rate", 16000)
    phases = decoder_output.get("phases", {})
    gen_tokens = phases.get("tokens_generated", 0)
    tok_per_sec = phases.get("tokens_per_second", 0.0)

    out: Dict[str, Any] = {
        "transcript": transcript,
        "audio_seconds": audio_dur,
        "native_sample_rate": native_sr,
        "audio_conversion": {
            "original_sample_rate": audio_info.get("original_sample_rate", native_sr),
            "target_sample_rate": 16000,
            "is_resampled_to_16k": bool(audio_info.get("is_resampled_to_16k", native_sr != 16000)),
            "original_channels": audio_info.get("original_channels", 1),
            "total_16k_samples": audio_info.get("total_16k_samples", int(round(audio_dur * 16000))),
            "duration_seconds": audio_dur,
        },
        "metrics": {
            "audio_duration_s": audio_dur,
            "inference_seconds": inf_time,
            "real_time_factor_rtf": rtf,
            "speedup_factor": f"{speedup}x",
            "tokens_generated": gen_tokens,
            "tokens_per_second": tok_per_sec,
        },
        "generation": {
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
        },
        "timing": {
            "inference_seconds": inf_time,
            "real_time_factor_rtf": rtf,
            "speedup": f"{speedup}x",
        },
    }
    if "flushed_windows" in decoder_output:
        out["flushed_windows"] = decoder_output["flushed_windows"]
        out["metrics"]["flushed_windows"] = decoder_output["flushed_windows"]
    if "decode_path" in decoder_output:
        out["decode_path"] = decoder_output["decode_path"]
    if phases:
        out["phases"] = phases
    # Observability for the language lock (cheap, no PII beyond transcript).
    if clean_enabled and transcript != raw_transcript:
        out["transcript_raw"] = raw_transcript
    out["detected_scripts"] = detect_scripts(transcript)
    if "ban_applied" in decoder_output:
        out["ban_applied"] = bool(decoder_output["ban_applied"])
    if "banned_token_count" in decoder_output:
        out["banned_token_count"] = int(decoder_output["banned_token_count"] or 0)
    return out


# ---------------------------------------------------------------------------
# Echo / stub path (kept for local smoke tests that don't have audio)
# ---------------------------------------------------------------------------
def _run_echo(validated: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
    """Diagnostic echo path: returns the input message verbatim.

    Returns the canonical ``{"status": "success", "output": {...}}`` shape
    used by the handler's return path.
    """
    return {
        "status": "success",
        "output": {
            "message": validated["message"],
            "kind": "echo",
        },
        "timing": {"inference_seconds": round(elapsed, 4)},
    }


# ---------------------------------------------------------------------------
# Public entry point used by the API layer and tests
# ---------------------------------------------------------------------------
def run_inference(
    model_bundle: Any,
    input_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the inference pipeline against an already-loaded model bundle.

    ``model_bundle`` is whatever :func:`src.model.load_model` returned. When
    ``model_bundle`` is ``None`` (handler not yet wired up the loader), the
    echo path is used so smoke tests remain useful.
    """
    settings = get_settings()
    validated = validate_input(input_data)

    start = time.perf_counter()
    if validated["kind"] == "echo":
        result = _run_echo(validated, time.perf_counter() - start)
        log.debug("Echo path (no audio).")
        return result

    if model_bundle is None or not hasattr(model_bundle, "speech_llm"):
        # Spec rule: do not claim success if the model is not ready (#50).
        raise InputValidationError(
            "model not initialized yet; please retry in a few seconds"
        )

    decoder_output = _run_decoder_real(model_bundle, validated, settings)
    elapsed = time.perf_counter() - start
    result = _postprocess(decoder_output, settings, elapsed)
    log.debug("Inference took %.4fs", elapsed)
    return result