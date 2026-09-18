"""Tests for src/inference.py input validation + echo path.

These tests call validate_input()/run_inference() directly, pinning the
contract at the layer the server worker actually uses.
"""

from __future__ import annotations

import base64
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference import InputValidationError, run_inference, validate_input  # noqa: E402


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Echo path (diagnostic validation shape)
# ---------------------------------------------------------------------------
def test_echo_roundtrip():
    out = run_inference(None, {"message": "hello"})
    assert out["status"] == "success"
    assert out["output"]["message"] == "hello"
    assert out["output"]["kind"] == "echo"


@pytest.mark.parametrize("msg", ["hello", "Parrotlet smoke test"])
def test_echo_roundtrip_parametrized(msg):
    out = run_inference(None, {"message": msg})
    assert out["output"]["message"] == msg


def test_message_longer_than_8k_rejected():
    with pytest.raises(InputValidationError):
        validate_input({"message": "x" * 8193})


def test_audio_is_never_echoed():
    """Real audio must not fall through to the echo path: without a model it
    raises not-ready instead of echoing."""
    with pytest.raises(InputValidationError, match="not initialized"):
        run_inference(None, {"audio": _b64(b"\x00" * 32)})


# ---------------------------------------------------------------------------
# Audio shapes
# ---------------------------------------------------------------------------
def test_missing_input():
    with pytest.raises(InputValidationError):
        validate_input(None)
    with pytest.raises(InputValidationError):
        validate_input({})
    with pytest.raises(InputValidationError):
        validate_input("not-a-dict")


def test_audio_requires_one_of_three_keys():
    with pytest.raises(InputValidationError):
        validate_input({"sample_rate": 16000})
    with pytest.raises(InputValidationError):
        validate_input({"audio": "ZmFrZQ==", "audio_path": "/tmp/x.wav"})


def test_audio_path_must_be_absolute():
    with pytest.raises(InputValidationError, match="absolute"):
        validate_input({"audio_path": "relative.wav"})


def test_audio_path_must_exist():
    with pytest.raises(InputValidationError, match="not found"):
        validate_input({"audio_path": "/nonexistent/file.wav"})


def test_audio_url_must_be_http():
    with pytest.raises(InputValidationError):
        validate_input({"audio_url": "ftp://example.com/x.wav"})


def test_sample_rate_must_be_positive_integer():
    with pytest.raises(InputValidationError):
        validate_input({"audio": _b64(b"\x00" * 16), "sample_rate": -1})


def test_audio_without_model_is_not_ready_error():
    """Audio + no model bundle must raise (never fake success)."""
    from types import SimpleNamespace

    with pytest.raises(InputValidationError, match="not initialized"):
        run_inference(None, {"audio": _b64(b"\x00" * 32)})
    with pytest.raises(InputValidationError, match="not initialized"):
        run_inference(SimpleNamespace(), {"audio": _b64(b"\x00" * 32)})
