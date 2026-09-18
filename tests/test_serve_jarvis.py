"""Tests for serve_jarvis.py — FastAPI app on JarvisLabs VM.

Uses a fake worker so no GPU/model is needed.
Verifies the server preserves the API contract: routes, buffering
fields (window_index / flushed_windows / window events), and auth.
"""

from __future__ import annotations

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import serve_jarvis
from src.server import ParrotletWorker


class _FakeMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def remote_gen(self, *args, **kwargs):
        yield from self._fn(*args, **kwargs)


class _FakeWorker:
    transcribe_dict_impl = staticmethod(lambda payload: {"status": "success", "output": {"transcript": "ok"}})
    transcribe_bytes_impl = staticmethod(lambda raw, name="audio.wav": {"status": "success", "output": {"transcript": "ok"}})
    transcribe_chunk_bytes_impl = staticmethod(
        lambda raw, window_index=0, is_last=False, session_id="": {
            "status": "success",
            "output": {"transcript": "ok"},
            "window_index": window_index,
            "is_last": is_last,
            "session_id": session_id,
            "accumulated_transcript": "ok",
            "full_transcript": "ok" if is_last else "",
            "flushed_windows": 1,
            "flushed": True,
        }
    )
    transcribe_stream_bytes_impl = staticmethod(
        lambda raw: [
            {"event": "window", "window_index": 0, "flushed_windows": 1, "is_last": True, "transcript": "ok"},
            {"event": "final", "full_transcript": "ok", "flushed_windows": 1},
        ]
    )
    health_impl = staticmethod(lambda: {"loaded": True})

    def __init__(self, *args, **kwargs):
        pass

    @property
    def transcribe_dict(self):
        return _FakeMethod(self.transcribe_dict_impl)

    @property
    def transcribe_bytes(self):
        return _FakeMethod(self.transcribe_bytes_impl)

    @property
    def transcribe_chunk_bytes(self):
        return _FakeMethod(self.transcribe_chunk_bytes_impl)

    @property
    def transcribe_stream_bytes(self):
        return _FakeMethod(self.transcribe_stream_bytes_impl)

    @property
    def health(self):
        return _FakeMethod(self.health_impl)


@pytest.fixture()
def client():
    return TestClient(serve_jarvis.build_app(_FakeWorker))


def _wav_upload(name="sample.wav", payload=b"RIFF" + b"\x00" * 100):
    return {"file": (name, io.BytesIO(payload), "audio/wav")}


def test_shim_health_and_index(client):
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_shim_transcribe_passthrough(client):
    resp = client.post("/transcribe", files=_wav_upload())
    assert resp.status_code == 200
    assert resp.json()["output"]["transcript"] == "ok"


def test_shim_chunk_uses_window_index(client):
    resp = client.post(
        "/transcribe_chunk",
        files=_wav_upload(),
        data={"window_index": "3", "is_last": "true", "session_id": "s1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["window_index"] == 3
    assert "chunk_index" not in data
    assert data["full_transcript"] == "ok"


def test_shim_stream_emits_window_events(client):
    resp = client.post("/transcribe_stream", files=_wav_upload())
    assert resp.status_code == 200
    assert "event: window" in resp.text
    assert "event: final" in resp.text
    assert "event: chunk" not in resp.text


def test_shim_auth_enforced(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "shim-secret")
    authed = TestClient(serve_jarvis.build_app(_FakeWorker))
    try:
        assert authed.post("/transcribe", files=_wav_upload()).status_code == 401
        ok = authed.post(
            "/transcribe",
            files=_wav_upload(),
            headers={"Authorization": "Bearer shim-secret"},
        )
        assert ok.status_code == 200
    finally:
        monkeypatch.delenv("AUTH_TOKEN", raising=False)


def test_worker_remote_methods():
    """Worker methods support .remote(), .local(), and direct call."""
    w = ParrotletWorker()
    assert callable(w.transcribe_dict)
    assert callable(w.transcribe_dict.remote)
    assert callable(w.transcribe_dict.local)
    assert callable(w.transcribe_stream_bytes.remote_gen)
