"""Tests for Pipelined / Streaming Buffering (transcribe_single_window,

transcribe_chunk_bytes, session joining, and streaming endpoints).
"""

from __future__ import annotations

import io
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_settings
from src.inference import InputValidationError, transcribe_single_window
from src.server import ParrotletWorker, build_fastapi_app


def _get_test_settings(**kwargs):
    env = {
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
        "BAN_SCRIPT_TOKENS": "1",
        "CLEAN_TRANSCRIPT": "1",
    }
    env.update(kwargs)
    return load_settings(env)


def _create_wav_bytes(duration_sec=0.2, sr=16000):
    buf = io.BytesIO()
    samples = int(duration_sec * sr)
    sf.write(buf, np.zeros(samples, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def mock_model_cache(monkeypatch):
    import src.model as model_mod
    monkeypatch.setitem(model_mod._CACHE, "loaded", True)
    monkeypatch.setitem(model_mod._CACHE, "speech_llm", MagicMock())
    monkeypatch.setitem(model_mod._CACHE, "metadata", {"placement": "cpu", "sampling_rate": 16000})
    monkeypatch.setitem(model_mod._CACHE, "banned_ids", [])
    monkeypatch.setattr(model_mod, "ensure_banned_ids", lambda: None)


@pytest.fixture(autouse=True)
def _pin_real_worker(monkeypatch):
    import src.server as server_mod
    monkeypatch.setattr(server_mod, "ParrotletWorker", ParrotletWorker)


# ---------------------------------------------------------------------------
# Unit tests for transcribe_single_window
# ---------------------------------------------------------------------------
def test_transcribe_single_window_basic():
    settings = _get_test_settings()
    mock_speech_llm = MagicMock()
    mock_speech_llm.transcribe.return_value = "patient has mild cough"
    bundle = SimpleNamespace(speech_llm=mock_speech_llm, banned_token_ids=[101, 102])

    audio_window = np.zeros(16000, dtype=np.float32)  # 1 second
    out = transcribe_single_window(bundle, audio_window, settings)

    assert out["transcript"] == "patient has mild cough"
    assert out["audio_seconds"] == 1.0
    assert out["native_sample_rate"] == 16000
    assert out["flushed_windows"] == 1
    assert "generation" in out
    assert "timing" in out
    assert "detected_scripts" in out
    mock_speech_llm.transcribe.assert_called_once()
    call_args, call_kwargs = mock_speech_llm.transcribe.call_args
    assert call_args[1] == 16000
    assert call_kwargs.get("bad_words_ids") == [[101], [102]]


def test_transcribe_single_window_clean_and_script_detection():
    settings = _get_test_settings(CLEAN_TRANSCRIPT="1")
    mock_speech_llm = MagicMock()
    # Noise tags and bracketed glosses should be cleaned, and Hindi script detected
    mock_speech_llm.transcribe.return_value = "<talking> [Chief] complaint is fever नमस्ते </talking>"
    bundle = SimpleNamespace(speech_llm=mock_speech_llm)

    audio_window = np.zeros(8000, dtype=np.float32)
    out = transcribe_single_window(bundle, audio_window, settings)

    assert out["transcript"] == "Chief complaint is fever नमस्ते"
    assert out["transcript_raw"] == "<talking> [Chief] complaint is fever नमस्ते </talking>"
    assert out["detected_scripts"]["has_devanagari"] is True
    assert out["detected_scripts"]["has_indic"] is True


def test_transcribe_single_window_empty_audio_raises():
    settings = _get_test_settings()
    bundle = SimpleNamespace(speech_llm=MagicMock())
    with pytest.raises(InputValidationError, match="no samples"):
        transcribe_single_window(bundle, np.zeros(0, dtype=np.float32), settings)


def test_transcribe_single_window_uninitialized_model_raises():
    settings = _get_test_settings()
    with pytest.raises(InputValidationError, match="model not initialized"):
        transcribe_single_window(None, np.zeros(16000, dtype=np.float32), settings)

    with pytest.raises(InputValidationError, match="model not initialized"):
        transcribe_single_window(SimpleNamespace(), np.zeros(16000, dtype=np.float32), settings)


def test_transcribe_single_window_fallback_to_batched(monkeypatch):
    settings = _get_test_settings()
    # When speech_llm lacks transcribe, it should fall back to _transcribe_batched
    mock_llm = SimpleNamespace(decoder=MagicMock())
    bundle = SimpleNamespace(speech_llm=mock_llm)

    import src.inference as inf_mod
    monkeypatch.setattr(inf_mod, "_transcribe_batched", lambda llm, windows, s, bad_words=None: ["batched output"])

    out = transcribe_single_window(bundle, np.zeros(16000, dtype=np.float32), settings)
    assert out["transcript"] == "batched output"


# ---------------------------------------------------------------------------
# Unit tests for ParrotletWorker.transcribe_chunk_bytes and session joining
# ---------------------------------------------------------------------------
def test_worker_transcribe_chunk_bytes_session_joining(monkeypatch):
    """Buffering: small appends emit nothing until is_last flushes one window."""
    worker = ParrotletWorker()
    worker._buffers.clear()

    calls = []

    def fake_transcribe_single(bundle, audio_16k, settings):
        calls.append(len(audio_16k))
        return {
            "transcript": "The patient presents with bronchial asthma.",
            "audio_seconds": 1.0,
            "native_sample_rate": 16000,
            "generation": {},
            "timing": {"inference_seconds": 0.05},
        }

    monkeypatch.setattr("src.inference.transcribe_single_window", fake_transcribe_single)
    monkeypatch.setattr("src.model.is_loaded", lambda: True)
    monkeypatch.setattr("src.model.ensure_banned_ids", lambda: None)
    monkeypatch.setattr("src.model.load_model", lambda: SimpleNamespace(speech_llm=MagicMock(), metadata={}, banned_token_ids=[]))

    wav_bytes = _create_wav_bytes()
    session_id = "test_session_123"

    res0 = worker.transcribe_chunk_bytes.local(wav_bytes, window_index=0, is_last=False, session_id=session_id)
    assert res0["status"] == "success"
    assert res0["flushed"] is False
    assert res0["flushed_windows"] == 0
    assert calls == []
    assert session_id in worker._buffers

    res1 = worker.transcribe_chunk_bytes.local(wav_bytes, window_index=1, is_last=False, session_id=session_id)
    assert res1["status"] == "success"
    assert res1["flushed"] is False
    assert calls == []

    res2 = worker.transcribe_chunk_bytes.local(wav_bytes, window_index=2, is_last=True, session_id=session_id)
    assert res2["status"] == "success"
    assert res2["is_last"] is True
    assert res2["flushed"] is True
    assert res2["flushed_windows"] == 1
    assert len(calls) == 1
    expected_full = "The patient presents with bronchial asthma."
    assert res2["full_transcript"] == expected_full
    assert res2["accumulated_transcript"] == expected_full
    assert res2["output"]["full_transcript"] == expected_full

    # Session must be cleaned up after is_last
    assert session_id not in worker._buffers


def test_worker_transcribe_chunk_bytes_threshold_flush(monkeypatch):
    """20s + 20s: second append flushes one full 30s window, keeps 10s buffered."""
    worker = ParrotletWorker()
    worker._buffers.clear()

    window_texts = ["first thirty seconds of speech", "remaining tail speech"]
    calls = []

    def fake_transcribe_single(bundle, audio_16k, settings):
        calls.append(len(audio_16k))
        idx = len(calls) - 1
        return {
            "transcript": window_texts[idx] if idx < len(window_texts) else f"extra {idx}",
            "audio_seconds": round(len(audio_16k) / 16000, 3),
            "native_sample_rate": 16000,
            "generation": {},
            "timing": {"inference_seconds": 0.05},
        }

    monkeypatch.setattr("src.inference.transcribe_single_window", fake_transcribe_single)
    monkeypatch.setattr("src.model.is_loaded", lambda: True)
    monkeypatch.setattr("src.model.ensure_banned_ids", lambda: None)
    monkeypatch.setattr("src.model.load_model", lambda: SimpleNamespace(speech_llm=MagicMock(), metadata={}, banned_token_ids=[]))

    session_id = "test_threshold_sess"
    wav20a = _create_wav_bytes(duration_sec=20.0)
    wav20b = _create_wav_bytes(duration_sec=20.0)

    res0 = worker.transcribe_chunk_bytes.local(wav20a, window_index=0, is_last=False, session_id=session_id)
    assert res0["status"] == "success"
    assert res0["flushed"] is False
    assert calls == []

    res1 = worker.transcribe_chunk_bytes.local(wav20b, window_index=1, is_last=False, session_id=session_id)
    assert res1["status"] == "success"
    assert res1["flushed"] is True
    assert res1["flushed_windows"] == 1
    assert len(calls) == 1
    assert calls[0] == 480000  # full 30 s window
    assert res1["accumulated_transcript"] == "first thirty seconds of speech"
    # 40 s in, 30 s flushed -> 10 s left buffered.
    assert res1["buffered_seconds"] == pytest.approx(10.0, abs=0.5)

    res2 = worker.transcribe_chunk_bytes.local(
        _create_wav_bytes(duration_sec=1.0), window_index=2, is_last=True, session_id=session_id
    )
    assert res2["status"] == "success"
    assert res2["is_last"] is True
    assert res2["full_transcript"] == "first thirty seconds of speech remaining tail speech"
    assert session_id not in worker._buffers


def test_worker_transcribe_chunk_bytes_no_overlap_no_dedup(monkeypatch):
    """35 s single upload with is_last -> 2 windows, plain-joined as-is."""
    worker = ParrotletWorker()
    worker._buffers.clear()

    window_texts = [
        "patient reported high fever and cough",
        "fever and cough since yesterday morning",
    ]
    calls = []

    def fake_transcribe_single(bundle, audio_16k, settings):
        calls.append(len(audio_16k))
        idx = len(calls) - 1
        return {
            "transcript": window_texts[idx],
            "audio_seconds": round(len(audio_16k) / 16000, 3),
            "native_sample_rate": 16000,
            "generation": {},
            "timing": {"inference_seconds": 0.05},
        }

    monkeypatch.setattr("src.inference.transcribe_single_window", fake_transcribe_single)
    monkeypatch.setattr("src.model.is_loaded", lambda: True)
    monkeypatch.setattr("src.model.ensure_banned_ids", lambda: None)
    monkeypatch.setattr("src.model.load_model", lambda: SimpleNamespace(speech_llm=MagicMock(), metadata={}, banned_token_ids=[]))

    wav_bytes = _create_wav_bytes(duration_sec=35.0)
    session_id = "test_no_overlap_sess"

    res1 = worker.transcribe_chunk_bytes.local(wav_bytes, window_index=0, is_last=True, session_id=session_id)

    assert res1["status"] == "success"
    assert res1["flushed_windows"] == 2
    # No overlap, no dedup: repeated "fever and cough" is kept verbatim.
    assert res1["full_transcript"] == (
        "patient reported high fever and cough "
        "fever and cough since yesterday morning"
    )
    assert session_id not in worker._buffers


def test_worker_transcribe_chunk_bytes_without_session():
    worker = ParrotletWorker()
    worker._buffers.clear()

    wav_bytes = _create_wav_bytes()
    res = worker.transcribe_chunk_bytes.local(wav_bytes, window_index=0, is_last=False, session_id="")
    assert res["status"] == "success"
    assert "session_id" not in res
    assert len(worker._buffers) == 0


def test_worker_transcribe_chunk_bytes_empty_payload():
    worker = ParrotletWorker()
    res = worker.transcribe_chunk_bytes.local(b"", window_index=0, is_last=False, session_id="")
    assert res["status"] == "error"
    assert res["error_kind"] == "validation"


# ---------------------------------------------------------------------------
# HTTP surface tests: /transcribe_chunk and /transcribe_stream
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(monkeypatch):
    class _TestFakeWorker:
        transcribe_chunk_bytes_impl = staticmethod(
            lambda raw, window_index=0, is_last=False, session_id="": {
                "status": "success",
                "output": {"transcript": f"window {window_index} text"},
                "window_index": window_index,
                "is_last": is_last,
                "session_id": session_id,
                "accumulated_transcript": f"accumulated through {window_index}",
                "full_transcript": "full transcript assembled" if is_last else "",
            }
        )
        transcribe_stream_bytes_impl = staticmethod(
            lambda raw: [
                {"event": "window", "window_index": 0, "flushed_windows": 2, "is_last": False, "transcript": "window 0"},
                {"event": "window", "window_index": 1, "flushed_windows": 2, "is_last": True, "transcript": "window 1"},
                {"event": "final", "full_transcript": "window 0 window 1", "flushed_windows": 2},
            ]
        )

        @property
        def transcribe_chunk_bytes(self):
            return SimpleNamespace(
                local=self.transcribe_chunk_bytes_impl,
                remote=self.transcribe_chunk_bytes_impl,
            )

        @property
        def transcribe_stream_bytes(self):
            impl = self.transcribe_stream_bytes_impl

            def _remote_gen(*args, **kwargs):
                yield from impl(*args, **kwargs)

            return SimpleNamespace(local=impl, remote=impl, remote_gen=_remote_gen)

    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MODAL_AUTH_TOKEN", raising=False)
    return TestClient(build_fastapi_app(worker_cls=_TestFakeWorker))


def test_http_transcribe_chunk_endpoint(client):
    wav = _create_wav_bytes()
    resp = client.post(
        "/transcribe_chunk",
        files={"file": ("part0.wav", io.BytesIO(wav), "audio/wav")},
        data={"window_index": "0", "is_last": "false", "session_id": "sess_http"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["window_index"] == 0
    assert data["is_last"] is False
    assert data["session_id"] == "sess_http"

    resp_last = client.post(
        "/transcribe_chunk",
        files={"file": ("part1.wav", io.BytesIO(wav), "audio/wav")},
        data={"window_index": "1", "is_last": "true", "session_id": "sess_http"},
    )
    assert resp_last.status_code == 200
    data_last = resp_last.json()
    assert data_last["is_last"] is True
    assert data_last["full_transcript"] == "full transcript assembled"


def test_http_transcribe_chunk_empty_upload(client):
    resp = client.post(
        "/transcribe_chunk",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
        data={"window_index": "0", "is_last": "false"},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_http_transcribe_stream_sse_endpoint(client):
    wav = _create_wav_bytes()
    resp = client.post(
        "/transcribe_stream",
        files={"file": ("full.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    content = resp.text
    assert "event: window" in content
    assert "event: final" in content
    assert "window 0" in content
    assert "full_transcript" in content


def test_http_transcribe_stream_empty_upload(client):
    resp = client.post(
        "/transcribe_stream",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert resp.status_code == 400
