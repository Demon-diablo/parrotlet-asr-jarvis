"""Inference-layer hardening (audit 2026-09-04).

Covers error classification without loading the model: corrupt/empty audio
must raise InputValidationError (HTTP 400 downstream), never leak through as
a generic inference failure (HTTP 500). Uses a minimal fake bundle — decode
fails before the model is touched.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference import InputValidationError, run_inference, validate_input  # noqa: E402

_FAKE_BUNDLE = SimpleNamespace(speech_llm=object())


def test_corrupt_audio_is_validation_error():
    payload = {"audio": base64.b64encode(b"\x00" * 64).decode("ascii")}
    with pytest.raises(InputValidationError, match="could not decode"):
        run_inference(_FAKE_BUNDLE, payload)


def test_empty_wav_has_no_samples():
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(0, dtype=np.float32), 16000, format="WAV")
    payload = {"audio": base64.b64encode(buf.getvalue()).decode("ascii")}
    with pytest.raises(InputValidationError, match="no samples"):
        run_inference(_FAKE_BUNDLE, payload)


def test_oversize_audio_path_rejected_without_reading():
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        os.ftruncate(fd, 51 * 1024 * 1024)  # sparse: no disk used
        os.close(fd)
        with pytest.raises(InputValidationError, match="exceeds"):
            validate_input({"audio_path": path})
    finally:
        os.unlink(path)


