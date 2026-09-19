"""Server HTTP API contract tests (no GPU, no model download, no network).

Covers src/server.py's HTTP surface:
  - route contract (/transcribe, /transcribe_b64, /transcribe_chunk, /transcribe_stream, /health, /)
  - bearer auth accept/reject (401)
  - validation mapping (400 empty/invalid, 413 oversize, 500 inference failure)
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from src.server import build_fastapi_app


class _FakeMethod:
    def __init__(self, fn):
        self._fn = fn

    def local(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def remote_gen(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        if isinstance(result, list):
            yield from result
        else:
            yield result


class _FakeWorker:
    """Stands in for ParrotletWorker. Configure per-test via class attrs."""

    transcribe_bytes_impl = staticmethod(
        lambda raw, name="audio.wav": {"status": "success", "output": {"transcript": "ok"}}
    )
    transcribe_dict_impl = staticmethod(
        lambda payload: {"status": "success", "output": {"transcript": "ok"}}
    )
    transcribe_chunk_bytes_impl = staticmethod(
        lambda raw, window_index=0, is_last=False, session_id="": {
            "status": "success",
            "output": {"transcript": "ok"},
            "window_index": window_index,
            "is_last": is_last,
            "session_id": session_id,
            "accumulated_transcript": "ok",
            "full_transcript": "ok" if is_last else "",
        }
    )
    transcribe_stream_bytes_impl = staticmethod(
        lambda raw: [
            {"event": "window", "window_index": 0, "flushed_windows": 1, "is_last": True, "transcript": "ok"},
            {"event": "final", "full_transcript": "ok", "flushed_windows": 1},
        ]
    )
    health_impl = staticmethod(lambda: {"loaded": True, "metadata": {}, "gpu": {}})
    extract_text_impl = staticmethod(
        lambda transcript="", system_prompt=None, temperature=None, max_tokens=None: {
            "status": "success",
            "output": {"valid_json": True, "medications": [{"spoken_name": "Pan 40"}], "medications_count": 1},
        }
    )
    extract_stream_text_impl = staticmethod(
        lambda transcript="", system_prompt=None, temperature=None, max_tokens=None: [
            {"event": "extraction_start", "transcript": transcript},
            {"event": "extraction_complete", "valid_json": True, "medications": [{"spoken_name": "Pan 40"}]},
        ]
    )
    pipeline_impl = staticmethod(
        lambda raw, system_prompt=None, temperature=None, max_tokens=None: {
            "status": "success",
            "output": {
                "transcript": "Pan 40 OD",
                "extraction": {"valid_json": True, "medications": [{"spoken_name": "Pan 40"}]},
            },
        }
    )
    pipeline_stream_impl = staticmethod(
        lambda raw, system_prompt=None, temperature=None, max_tokens=None: [
            {"event": "window", "transcript": "Pan 40 OD"},
            {"event": "final", "full_transcript": "Pan 40 OD"},
            {"event": "asr_complete", "transcript": "Pan 40 OD"},
            {"event": "extraction_complete", "medications": [{"spoken_name": "Pan 40"}]},
        ]
    )

    def __init__(self, *args, **kwargs):
        pass

    @property
    def transcribe_bytes(self):
        return _FakeMethod(self.transcribe_bytes_impl)

    @property
    def transcribe_dict(self):
        return _FakeMethod(self.transcribe_dict_impl)

    @property
    def transcribe_chunk_bytes(self):
        return _FakeMethod(self.transcribe_chunk_bytes_impl)

    @property
    def transcribe_stream_bytes(self):
        return _FakeMethod(self.transcribe_stream_bytes_impl)

    @property
    def extract_text(self):
        return _FakeMethod(self.extract_text_impl)

    @property
    def extract_stream_text(self):
        return _FakeMethod(self.extract_stream_text_impl)

    @property
    def pipeline(self):
        return _FakeMethod(self.pipeline_impl)

    @property
    def pipeline_stream(self):
        return _FakeMethod(self.pipeline_stream_impl)

    @property
    def health(self):
        return _FakeMethod(self.health_impl)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MODAL_AUTH_TOKEN", raising=False)
    return TestClient(build_fastapi_app(worker_cls=_FakeWorker))


@pytest.fixture()
def authed_client(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-secret-token")
    return TestClient(build_fastapi_app(worker_cls=_FakeWorker))


def _wav_upload(name="sample.wav", payload=b"RIFF" + b"\x00" * 100):
    return {"file": (name, io.BytesIO(payload), "audio/wav")}


# ---------------------------------------------------------------------------
# Health & Index
# ---------------------------------------------------------------------------
def test_index_route(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "POST /transcribe" in resp.json()["endpoints"]


def test_health_open_without_token(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["loaded"] is True


def test_health_requires_token_when_configured(authed_client):
    assert authed_client.get("/health").status_code == 401
    assert authed_client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert authed_client.get("/health", headers={"Authorization": "Bearer test-secret-token"}).status_code == 200


# ---------------------------------------------------------------------------
# POST /transcribe (multipart)
# ---------------------------------------------------------------------------
def test_transcribe_success_passthrough(client):
    resp = client.post("/transcribe", files=_wav_upload())
    assert resp.status_code == 200
    assert resp.json()["output"]["transcript"] == "ok"


def test_transcribe_empty_upload_is_400(client):
    resp = client.post("/transcribe", files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")})
    assert resp.status_code == 400


def test_transcribe_oversize_is_413(client):
    big = b"\x00" * (50 * 1024 * 1024 + 1)
    resp = client.post("/transcribe", files={"file": ("big.wav", io.BytesIO(big), "audio/wav")})
    assert resp.status_code == 413


def test_transcribe_validation_error_is_400(client, monkeypatch):
    def boom(raw, name="audio.wav"):
        raise ValueError("'audio' payload is empty")

    monkeypatch.setattr(_FakeWorker, "transcribe_bytes_impl", staticmethod(boom))
    resp = client.post("/transcribe", files=_wav_upload())
    assert resp.status_code == 400


def test_transcribe_inference_error_is_500_without_leak(client, monkeypatch):
    def boom(raw, name="audio.wav"):
        raise RuntimeError("CUDA OOM on device 0\ntraceback line 2")

    monkeypatch.setattr(_FakeWorker, "transcribe_bytes_impl", staticmethod(boom))
    resp = client.post("/transcribe", files=_wav_upload())
    assert resp.status_code == 500
    assert "traceback" not in resp.json()["detail"] or len(resp.json()["detail"]) <= 500


def test_transcribe_auth_enforced(authed_client):
    assert authed_client.post("/transcribe", files=_wav_upload()).status_code == 401
    ok = authed_client.post(
        "/transcribe",
        files=_wav_upload(),
        headers={"Authorization": "Bearer test-secret-token"},
    )
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# POST /transcribe_b64 (base64 JSON input)
# ---------------------------------------------------------------------------
def test_transcribe_b64_success(client):
    resp = client.post("/transcribe_b64", json={"audio": "aGVsbG8="})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_transcribe_b64_validation_error_is_400(client, monkeypatch):
    def boom(payload):
        raise ValueError("exactly one of 'audio', 'audio_path', 'audio_url' is required")

    monkeypatch.setattr(_FakeWorker, "transcribe_dict_impl", staticmethod(boom))
    resp = client.post("/transcribe_b64", json={"nope": 1})
    assert resp.status_code == 400
    assert "exactly one" in resp.json()["detail"]


@pytest.mark.parametrize(
    "error_kind,expected_status",
    [("validation", 400), ("not_ready", 503), ("inference", 500)],
)
def test_worker_error_dict_maps_to_status(client, monkeypatch, error_kind, expected_status):
    """Local worker errors retain the existing HTTP status-code contract."""

    def fail(payload):
        return {"status": "error", "error": "bad input shape", "error_kind": error_kind}

    monkeypatch.setattr(_FakeWorker, "transcribe_dict_impl", staticmethod(fail))
    resp = client.post("/transcribe_b64", json={"audio": "aGVsbG8="})
    assert resp.status_code == expected_status
    assert "bad input shape" in resp.json()["detail"]


@pytest.mark.parametrize("path", ["/transcribe_chunk", "/transcribe_stream"])
def test_existing_chunk_routes_require_auth(authed_client, path):
    assert authed_client.post(path, files=_wav_upload()).status_code == 401


def test_stream_uses_remote_generator(client):
    response = client.post("/transcribe_stream", files=_wav_upload())
    assert response.status_code == 200
    assert "event: window" in response.text
    assert "event: final" in response.text


def test_index_includes_new_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    endpoints = resp.json()["endpoints"]
    assert "POST /extract" in endpoints
    assert "POST /extract_stream" in endpoints
    assert "POST /pipeline" in endpoints
    assert "POST /pipeline_stream" in endpoints


def test_extract_endpoint(client):
    resp = client.post("/extract", json={"transcript": "Pan 40 OD"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["output"]["medications_count"] == 1


def test_extract_empty_400(client):
    resp = client.post("/extract", json={"transcript": ""})
    assert resp.status_code == 400


def test_extract_stream_endpoint(client):
    resp = client.post("/extract_stream", json={"transcript": "Pan 40 OD"})
    assert resp.status_code == 200
    assert "event: extraction_start" in resp.text
    assert "event: extraction_complete" in resp.text


def test_pipeline_endpoint(client):
    resp = client.post("/pipeline", files=_wav_upload())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["output"]["transcript"] == "Pan 40 OD"


def test_pipeline_stream_endpoint(client):
    resp = client.post("/pipeline_stream", files=_wav_upload())
    assert resp.status_code == 200
    assert "event: window" in resp.text
    assert "event: asr_complete" in resp.text
    assert "event: extraction_complete" in resp.text


@pytest.mark.parametrize("path", ["/extract", "/extract_stream"])
def test_extract_routes_require_auth(authed_client, path):
    assert authed_client.post(path, json={"transcript": "test"}).status_code == 401


@pytest.mark.parametrize("path", ["/pipeline", "/pipeline_stream"])
def test_pipeline_routes_require_auth(authed_client, path):
    assert authed_client.post(path, files=_wav_upload()).status_code == 401
