"""FastAPI server and Parrotlet worker for JarvisLabs GPU VM serving."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, security
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

log = logging.getLogger("parrotlet.server")


def _warm_bundle():
    """Rebuild the run_inference() bundle view from the module cache."""
    from src.model import _CACHE

    md = _CACHE.get("metadata") or {}
    return type(
        "_Bundle",
        (),
        {
            "speech_llm": _CACHE.get("speech_llm"),
            "encoder": _CACHE.get("encoder"),
            "projector": _CACHE.get("projector"),
            "decoder": _CACHE.get("decoder"),
            "tokenizer": _CACHE.get("tokenizer"),
            "processor": _CACHE.get("processor"),
            "metadata": md,
            "sampling_rate": int(md.get("sampling_rate", 16000)),
            "banned_token_ids": list(_CACHE.get("banned_ids") or []),
            "unwrap": lambda self: (self.speech_llm, self.tokenizer),
        },
    )()


class _CallableMethod:
    """Wraps a worker method to support direct invocation as well as
    .remote(), .local(), and .remote_gen() for seamless interface compatibility."""

    def __init__(self, fn, is_generator: bool = False):
        self._fn = fn
        self._is_generator = is_generator

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    @property
    def local(self):
        return self

    @property
    def remote(self):
        return self

    def remote_gen(self, *args, **kwargs):
        res = self._fn(*args, **kwargs)
        if hasattr(res, "__iter__") and not isinstance(res, (dict, str, bytes)):
            yield from res
        else:
            yield res


class ParrotletWorker:
    """In-process GPU worker for speech transcription, session buffering,
    and streaming inference on JarvisLabs VM."""

    _buffers: Dict[str, Dict[str, Any]] = {}
    _buffers_lock: threading.Lock = threading.Lock()
    _BUFFER_TTL_S = 600.0  # drop sessions idle >10 min
    _BUFFER_MAX_S = 600.0  # max 10 min audio per session (~38 MB float32)

    def __init__(self, *args, **kwargs):
        pass

    def load(self) -> None:
        """Pre-warm model and TF32 on the GPU."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
        from src.config import get_settings
        from src.gpu import configure_tf32, gpu_info
        from src.model import ensure_banned_ids, load_model

        try:
            settings = get_settings()
            tf32_status = configure_tf32(
                enabled=settings.allow_tf32,
                precision=settings.float32_matmul_precision,
            )
            print(f"[parrotlet] TF32 architecture configured: {tf32_status}", flush=True)
        except Exception as exc:
            print(f"[parrotlet] TF32 init warning: {exc}", flush=True)

        t0 = time.perf_counter()
        bundle = load_model()
        print(
            f"[parrotlet] model ready in {time.perf_counter() - t0:.1f}s "
            f"placement={(getattr(bundle, 'metadata', None) or {}).get('placement')} "
            f"gpus={[g.get('name') for g in gpu_info().get('gpus', [])]}",
            flush=True,
        )
        try:
            ensure_banned_ids()
        except Exception as exc:
            print(f"[parrotlet] banned-ids prewarm skipped: {exc}", flush=True)

        try:
            from src.extractor import warmup_extractor
            warmup_extractor()
        except Exception as exc:
            print(f"[parrotlet] extractor prewarm skipped: {exc}", flush=True)

    def _transcribe_dict_impl(self, input_data: dict) -> dict:
        from src.inference import InputValidationError, run_inference
        from src.model import is_loaded, load_model

        bundle = load_model() if not is_loaded() else _warm_bundle()
        t0 = time.perf_counter()
        try:
            output = run_inference(bundle, input_data)
        except InputValidationError as exc:
            msg = str(exc)
            kind = "not_ready" if "not initialized" in msg else "validation"
            return {"status": "error", "error": msg, "error_kind": kind}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            return {"status": "error", "error": msg[:500], "error_kind": "inference"}
        print(
            f"[parrotlet] inference took {time.perf_counter() - t0:.2f}s "
            f"path={output.get('decode_path', '?')} windows={output.get('flushed_windows', '?')}",
            flush=True,
        )
        return {"status": "success", "output": output}

    @property
    def transcribe_dict(self):
        return _CallableMethod(self._transcribe_dict_impl)

    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """Convenience wrapper for multipart uploads."""
        return self._transcribe_dict_impl({"audio": audio_bytes})

    def _transcribe_chunk_bytes_impl(
        self, chunk_bytes: bytes, window_index: int = 0, is_last: bool = False, session_id: str = ""
    ) -> dict:
        from src import audio as audio_module
        from src.config import get_settings
        from src.inference import InputValidationError, transcribe_single_window
        from src.model import is_loaded, load_model

        t0 = time.perf_counter()
        bundle = load_model() if not is_loaded() else _warm_bundle()

        try:
            if not chunk_bytes:
                raise InputValidationError("audio piece payload is empty")
            audio_16k, native_sr, _ = audio_module.load_audio_for_inference(
                {"kind": "audio_bytes", "audio_bytes": chunk_bytes, "sample_rate": 16000}
            )
            if len(audio_16k) == 0:
                raise InputValidationError("audio piece contains no samples")
            settings = get_settings()
        except InputValidationError as exc:
            msg = str(exc)
            kind = "not_ready" if "not initialized" in msg else "validation"
            return {"status": "error", "error": msg, "error_kind": kind}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            return {"status": "error", "error": msg[:500], "error_kind": "inference"}

        import numpy as np

        new_pcm = np.asarray(audio_16k, dtype=np.float32).reshape(-1)

        if not session_id:
            try:
                output = transcribe_single_window(bundle, new_pcm, settings)
            except InputValidationError as exc:
                msg = str(exc)
                kind = "not_ready" if "not initialized" in msg else "validation"
                return {"status": "error", "error": msg, "error_kind": kind}
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                return {"status": "error", "error": msg[:500], "error_kind": "inference"}
            print(
                f"[parrotlet] piece {window_index} (no session) in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
            return {
                "status": "success",
                "output": output,
                "window_index": window_index,
                "is_last": is_last,
            }

        max_samples = int(self._BUFFER_MAX_S * 16000)

        now = time.time()
        with self._buffers_lock:
            for sid in list(self._buffers.keys()):
                try:
                    if now - float(self._buffers[sid].get("updated_at", now)) > self._BUFFER_TTL_S:
                        del self._buffers[sid]
                except Exception:
                    pass
            entry = self._buffers.get(session_id)
            if entry is None:
                entry = {
                    "pcm": np.zeros(0, dtype=np.float32),
                    "texts": [],
                    "updated_at": now,
                }
            base = np.asarray(entry.get("pcm", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if len(base) + len(new_pcm) > max_samples:
                return {
                    "status": "error",
                    "error": "session buffer exceeds 600 s cap; send is_last or start a new session",
                    "error_kind": "validation",
                }
            pcm_all = np.concatenate([base, new_pcm]) if len(base) else new_pcm
            if is_last:
                window_audios = audio_module.split_windows(pcm_all)
                consumed = len(pcm_all)
            else:
                window_audios, consumed = audio_module.take_full_windows(pcm_all)
            consumed_pcm = pcm_all[:consumed].copy() if consumed else np.zeros(0, dtype=np.float32)
            remainder = pcm_all[consumed:].astype(np.float32, copy=True) if consumed else pcm_all
            entry["pcm"] = remainder
            entry["updated_at"] = now
            self._buffers[session_id] = entry

        new_texts: list = []
        if window_audios:
            try:
                for wa in window_audios:
                    out = transcribe_single_window(bundle, wa, settings)
                    new_texts.append(out.get("transcript", ""))
            except InputValidationError as exc:
                with self._buffers_lock:
                    ent = self._buffers.get(session_id)
                    if ent is None:
                        ent = {"pcm": np.zeros(0, dtype=np.float32), "texts": [], "updated_at": time.time()}
                    cur = np.asarray(ent.get("pcm", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
                    ent["pcm"] = np.concatenate([consumed_pcm, cur]) if len(cur) else consumed_pcm
                    ent["updated_at"] = time.time()
                    self._buffers[session_id] = ent
                msg = str(exc)
                kind = "not_ready" if "not initialized" in msg else "validation"
                return {"status": "error", "error": msg, "error_kind": kind}
            except Exception as exc:
                with self._buffers_lock:
                    ent = self._buffers.get(session_id)
                    if ent is None:
                        ent = {"pcm": np.zeros(0, dtype=np.float32), "texts": [], "updated_at": time.time()}
                    cur = np.asarray(ent.get("pcm", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1)
                    ent["pcm"] = np.concatenate([consumed_pcm, cur]) if len(cur) else consumed_pcm
                    ent["updated_at"] = time.time()
                    self._buffers[session_id] = ent
                msg = f"{type(exc).__name__}: {exc}"
                return {"status": "error", "error": msg[:500], "error_kind": "inference"}

        with self._buffers_lock:
            ent = self._buffers.get(session_id)
            if ent is None:
                ent = {"pcm": np.zeros(0, dtype=np.float32), "texts": [], "updated_at": time.time()}
            texts = list(ent.get("texts", []))
            texts.extend([t for t in new_texts if str(t).strip()])
            ent["texts"] = texts
            ent["updated_at"] = time.time()
            accumulated = audio_module.join_texts(texts)
            buffered_seconds = round(float(np.asarray(ent.get("pcm")).shape[0]) / 16000, 3)
            output: Dict[str, Any] = {
                "transcript": audio_module.join_texts([t for t in new_texts if str(t).strip()]),
                "audio_seconds": round(consumed / 16000, 3),
                "native_sample_rate": 16000,
                "flushed_windows": len(window_audios),
                "decode_path": f"buffered:B={len(window_audios)}",
                "buffered_seconds": buffered_seconds,
                "accumulated_transcript": accumulated,
            }
            res: Dict[str, Any] = {
                "status": "success",
                "output": output,
                "window_index": window_index,
                "is_last": is_last,
                "session_id": session_id,
                "accumulated_transcript": accumulated,
                "buffered_seconds": buffered_seconds,
                "flushed_windows": len(window_audios),
                "flushed": bool(window_audios),
            }
            if is_last:
                res["full_transcript"] = accumulated
                output["full_transcript"] = accumulated
                try:
                    del self._buffers[session_id]
                except KeyError:
                    pass

        print(
            f"[parrotlet] buffer session={session_id} idx={window_index} "
            f"flushed={len(window_audios)} buffered={buffered_seconds}s "
            f"is_last={is_last} in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
        return res

    @property
    def transcribe_chunk_bytes(self):
        return _CallableMethod(self._transcribe_chunk_bytes_impl)

    def _transcribe_stream_bytes_impl(self, audio_bytes: bytes):
        from src import audio as audio_module
        from src.config import get_settings
        from src.inference import InputValidationError, transcribe_single_window
        from src.model import is_loaded, load_model

        bundle = load_model() if not is_loaded() else _warm_bundle()

        try:
            if not audio_bytes:
                raise InputValidationError("audio payload is empty")
            audio_16k, native_sr, _ = audio_module.load_audio_for_inference(
                {"kind": "audio_bytes", "audio_bytes": audio_bytes, "sample_rate": 16000}
            )
            if len(audio_16k) == 0:
                raise InputValidationError("audio contains no samples")
        except Exception as exc:
            yield {
                "event": "error",
                "error": str(exc)[:500],
                "error_kind": "validation" if isinstance(exc, (InputValidationError, ValueError)) else "inference",
                "status": "error",
            }
            return

        settings = get_settings()
        windows = audio_module.split_windows(audio_16k)
        parts = []
        for i, window in enumerate(windows):
            is_last = (i == len(windows) - 1)
            try:
                window_out = transcribe_single_window(bundle, window, settings)
                text = window_out.get("transcript", "")
            except Exception as exc:
                yield {
                    "event": "error",
                    "window_index": i,
                    "error": str(exc)[:500],
                    "status": "error",
                }
                return

            parts.append(text)
            yield {
                "event": "window",
                "window_index": i,
                "flushed_windows": len(windows),
                "is_last": is_last,
                "transcript": text,
            }

        full_transcript = audio_module.join_texts(parts)

        yield {
            "event": "final",
            "full_transcript": full_transcript,
            "flushed_windows": len(windows),
        }

    @property
    def transcribe_stream_bytes(self):
        return _CallableMethod(self._transcribe_stream_bytes_impl, is_generator=True)

    def _extract_text_impl(
        self,
        transcript: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        from src.config import get_settings
        from src.extractor import extract_prescriptions
        from src.schema import STANDARD_SYSTEM_PROMPT

        settings = get_settings()
        sys_prompt = system_prompt if system_prompt is not None else STANDARD_SYSTEM_PROMPT
        temp = temperature if temperature is not None else settings.extractor_temperature
        max_tok = max_tokens if max_tokens is not None else settings.extractor_max_tokens

        try:
            res = extract_prescriptions(
                transcript=transcript,
                system_prompt=sys_prompt,
                temperature=temp,
                max_tokens=max_tok,
            )
            return {"status": "success", "output": res}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            return {"status": "error", "error": msg[:500], "error_kind": "extractor"}

    @property
    def extract_text(self):
        return _CallableMethod(self._extract_text_impl)

    def _extract_stream_text_impl(
        self,
        transcript: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        from src.config import get_settings
        from src.extractor import extract_prescriptions_stream
        from src.schema import STANDARD_SYSTEM_PROMPT

        settings = get_settings()
        sys_prompt = system_prompt if system_prompt is not None else STANDARD_SYSTEM_PROMPT
        temp = temperature if temperature is not None else settings.extractor_temperature
        max_tok = max_tokens if max_tokens is not None else settings.extractor_max_tokens

        try:
            yield from extract_prescriptions_stream(
                transcript=transcript,
                system_prompt=sys_prompt,
                temperature=temp,
                max_tokens=max_tok,
            )
        except Exception as exc:
            yield {
                "event": "error",
                "error": str(exc)[:500],
                "status": "error",
            }

    @property
    def extract_stream_text(self):
        return _CallableMethod(self._extract_stream_text_impl, is_generator=True)

    def _pipeline_impl(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        t0 = time.perf_counter()
        asr_res = self._transcribe_dict_impl({"audio": audio_bytes})
        if asr_res.get("status") == "error":
            return asr_res

        transcript = asr_res.get("output", {}).get("transcript", "")
        asr_latency = round(time.perf_counter() - t0, 3)

        ext_res = self._extract_text_impl(
            transcript=transcript,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if ext_res.get("status") == "error":
            return ext_res

        total_latency = round(time.perf_counter() - t0, 3)
        return {
            "status": "success",
            "output": {
                "transcript": transcript,
                "asr_output": asr_res.get("output", {}),
                "extraction": ext_res.get("output", {}),
                "asr_latency_seconds": asr_latency,
                "total_latency_seconds": total_latency,
            },
        }

    @property
    def pipeline(self):
        return _CallableMethod(self._pipeline_impl)

    def _pipeline_stream_impl(
        self,
        audio_bytes: bytes,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        asr_parts = []
        full_transcript = ""
        for item in self._transcribe_stream_bytes_impl(audio_bytes):
            if item.get("event") == "error":
                yield item
                return
            yield item
            if item.get("event") == "window":
                asr_parts.append(item.get("transcript", ""))
            elif item.get("event") == "final":
                full_transcript = item.get("full_transcript", "")

        transcript = full_transcript if full_transcript else " ".join(asr_parts)

        yield {"event": "asr_complete", "transcript": transcript}

        for item in self._extract_stream_text_impl(
            transcript=transcript,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield item

    @property
    def pipeline_stream(self):
        return _CallableMethod(self._pipeline_stream_impl, is_generator=True)

    def _health_impl(self) -> dict:
        import traceback

        try:
            from src.gpu import gpu_info
            from src.model import is_loaded, model_metadata

            result = {
                "loaded": is_loaded(),
                "metadata": model_metadata(),
                "gpu": gpu_info(),
            }
            return json.loads(json.dumps(result, default=str))
        except Exception:
            return {"loaded": False, "error": traceback.format_exc()[-2000:]}

    @property
    def health(self):
        return _CallableMethod(self._health_impl)


def build_fastapi_app(worker_cls=None):
    """Build the FastAPI app. ``worker_cls`` override exists for tests."""
    api = FastAPI(title="parrotlet-asr")
    bearer = security.HTTPBearer(auto_error=False)

    def check_auth(creds=Depends(bearer)) -> None:
        expected = os.getenv("AUTH_TOKEN") or os.getenv("MODAL_AUTH_TOKEN", "")
        if not expected:
            return  # auth not configured; endpoint open (dev only)
        token = creds.credentials if creds else ""
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid bearer token")

    def raise_for_worker_result(result: dict) -> dict:
        """Map the worker's error-dict contract to HTTP status codes."""
        if isinstance(result, dict) and result.get("status") == "error":
            kind = result.get("error_kind", "inference")
            detail = str(result.get("error", "inference failed"))[:500]
            if kind == "validation":
                raise HTTPException(status_code=400, detail=detail)
            if kind == "not_ready":
                raise HTTPException(status_code=503, detail=detail)
            raise HTTPException(status_code=500, detail=detail)
        return result

    async def _call_worker(fn, *args, **kwargs):
        return await run_in_threadpool(fn, *args, **kwargs)

    def _get_worker():
        cls = worker_cls if worker_cls is not None else ParrotletWorker
        return cls()

    @api.get("/")
    async def index():
        return {
            "ok": True,
            "endpoints": [
                "GET /health",
                "POST /transcribe",
                "POST /transcribe_b64",
                "POST /transcribe_chunk",
                "POST /transcribe_stream",
                "POST /extract",
                "POST /extract_stream",
                "POST /pipeline",
                "POST /pipeline_stream",
            ],
        }

    @api.get("/health")
    async def health(_: None = Depends(check_auth)):
        w = _get_worker()
        fn = getattr(w.health, "remote", w.health)
        return JSONResponse(await _call_worker(fn))

    @api.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...), _: None = Depends(check_auth)
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio exceeds 50 MiB cap")
        try:
            w = _get_worker()
            fn = getattr(w.transcribe_bytes, "remote", w.transcribe_bytes)
            result = await _call_worker(fn, raw, file.filename or "audio.wav")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)[:500])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500])
        return JSONResponse(raise_for_worker_result(result))

    @api.post("/transcribe_b64")
    async def transcribe_b64(payload: dict, _: None = Depends(check_auth)):
        try:
            w = _get_worker()
            fn = getattr(w.transcribe_dict, "remote", w.transcribe_dict)
            result = await _call_worker(fn, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)[:500])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500])
        return JSONResponse(raise_for_worker_result(result))

    @api.post("/transcribe_chunk")
    async def transcribe_chunk(
        file: UploadFile = File(...),
        window_index: int = Form(0),
        is_last: bool = Form(False),
        session_id: str = Form(""),
        _: None = Depends(check_auth),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio exceeds 50 MiB cap")
        try:
            w = _get_worker()
            fn = getattr(w.transcribe_chunk_bytes, "remote", w.transcribe_chunk_bytes)
            result = await _call_worker(
                fn,
                raw, window_index=window_index, is_last=is_last, session_id=session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)[:500])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500])
        return JSONResponse(raise_for_worker_result(result))

    @api.post("/transcribe_stream")
    async def transcribe_stream(
        file: UploadFile = File(...), _: None = Depends(check_auth)
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio exceeds 50 MiB cap")

        def event_generator():
            w = _get_worker()
            gen_fn = getattr(w.transcribe_stream_bytes, "remote_gen", w.transcribe_stream_bytes)
            for item in gen_fn(raw):
                if isinstance(item, dict):
                    event_type = item.get("event", "message")
                    yield f"event: {event_type}\ndata: {json.dumps(item)}\n\n"
                elif isinstance(item, str):
                    yield item if item.endswith("\n\n") else f"data: {item}\n\n"
                else:
                    yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api.post("/extract")
    async def extract(payload: dict, _: None = Depends(check_auth)):
        transcript = payload.get("transcript", "")
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="transcript cannot be empty")
        system_prompt = payload.get("system_prompt")
        temperature = payload.get("temperature")
        max_tokens = payload.get("max_tokens")
        try:
            w = _get_worker()
            fn = getattr(w.extract_text, "remote", w.extract_text)
            result = await _call_worker(
                fn,
                transcript=transcript,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)[:500])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500])
        return JSONResponse(raise_for_worker_result(result))

    @api.post("/extract_stream")
    async def extract_stream(payload: dict, _: None = Depends(check_auth)):
        transcript = payload.get("transcript", "")
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=400, detail="transcript cannot be empty")
        system_prompt = payload.get("system_prompt")
        temperature = payload.get("temperature")
        max_tokens = payload.get("max_tokens")

        def event_generator():
            w = _get_worker()
            gen_fn = getattr(w.extract_stream_text, "remote_gen", w.extract_stream_text)
            for item in gen_fn(
                transcript=transcript,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if isinstance(item, dict):
                    event_type = item.get("event", "message")
                    yield f"event: {event_type}\ndata: {json.dumps(item)}\n\n"
                elif isinstance(item, str):
                    yield item if item.endswith("\n\n") else f"data: {item}\n\n"
                else:
                    yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @api.post("/pipeline")
    async def pipeline(
        file: UploadFile = File(...),
        system_prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(None),
        max_tokens: Optional[int] = Form(None),
        _: None = Depends(check_auth),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio exceeds 50 MiB cap")
        try:
            w = _get_worker()
            fn = getattr(w.pipeline, "remote", w.pipeline)
            result = await _call_worker(
                fn,
                raw,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)[:500])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:500])
        return JSONResponse(raise_for_worker_result(result))

    @api.post("/pipeline_stream")
    async def pipeline_stream(
        file: UploadFile = File(...),
        system_prompt: Optional[str] = Form(None),
        temperature: Optional[float] = Form(None),
        max_tokens: Optional[int] = Form(None),
        _: None = Depends(check_auth),
    ):
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio exceeds 50 MiB cap")

        def event_generator():
            w = _get_worker()
            gen_fn = getattr(w.pipeline_stream, "remote_gen", w.pipeline_stream)
            for item in gen_fn(
                raw,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if isinstance(item, dict):
                    event_type = item.get("event", "message")
                    yield f"event: {event_type}\ndata: {json.dumps(item)}\n\n"
                elif isinstance(item, str):
                    yield item if item.endswith("\n\n") else f"data: {item}\n\n"
                else:
                    yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return api
