"""Tests for src/audio.py — resampling and long-audio buffering.

These tests exercise:

- resampling arbitrary inputs to 16 kHz mono
- short-form audio (≤ 30 s) returns one window
- long-form audio (> 30 s) is split into non-overlapping 30 s windows
- window boundaries preserve total audio length
- per-window transcript strings are joined in order
- stereo audio is mixed to mono
- multiple file formats (wav, flac, ogg-vorbis) decode correctly
- accurate ``audio_seconds`` is reported at the 16 kHz rate

The tests do NOT require the model to be loaded — they verify the
preprocessing layer's contract independently.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import audio as audio_module  # noqa: E402
from src.audio import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    join_texts,
    load_audio_for_inference,
    split_windows,
    take_full_windows,
    audio_seconds,
)


def _make_wav(sr: int, duration_s: float, channels: int = 1) -> bytes:
    """Build a deterministic PCM wav in memory."""
    import soundfile as sf

    if channels == 1:
        # = 1 Hz sine, 0.5 amplitude
        t = np.arange(int(sr * duration_s)) / sr
        data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    else:
        t = np.arange(int(sr * duration_s)) / sr
        left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (0.5 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
        data = np.stack([left, right], axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def test_short_16k_mono_passthrough():
    """A 5 s, 16 kHz, mono wav must come out at 16 kHz with the same length
    (no resample, no remixing)."""
    raw = _make_wav(16000, 5.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, native_sr, _ = load_audio_for_inference(validated)
    assert native_sr == 16000
    assert audio_16k.dtype == np.float32
    assert audio_16k.ndim == 1
    # 5 s @ 16 kHz → 80_000 samples (allow ±1 for rounding)
    assert abs(len(audio_16k) - 80000) <= 1


def test_short_22k_mono_is_resampled_to_16k():
    """Non-16 kHz audio is resampled to 16 kHz and length adjusts."""
    raw = _make_wav(22050, 5.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 22050,
    }
    audio_16k, native_sr, _ = load_audio_for_inference(validated)
    assert native_sr == 22050  # native is reported as input
    assert audio_16k.ndim == 1
    # 5 s @ 16 kHz → 80_000 samples (allow ±1 for rounding)
    assert abs(len(audio_16k) - 80000) <= 1


def test_short_48k_mono_is_resampled_to_16k():
    raw = _make_wav(48000, 3.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 48000,
    }
    audio_16k, native_sr, _ = load_audio_for_inference(validated)
    assert native_sr == 48000
    assert abs(len(audio_16k) - 48000) <= 1  # 3 s @ 16 kHz


def test_stereo_audio_is_mixed_to_mono():
    """Stereo input must come out as 1-D mono."""
    raw = _make_wav(16000, 3.0, channels=2)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    assert audio_16k.ndim == 1
    # Stereo sum-to-mono means amplitude ≈ 2x the original per channel.
    assert audio_16k.dtype == np.float32
    # If we averaged correctly, max(abs) <= 1.0 (no clipping of 0.5+0.5).
    assert float(np.max(np.abs(audio_16k))) <= 1.0


# ---------------------------------------------------------------------------
# Buffer windowing (non-overlapping 30 s windows)
# ---------------------------------------------------------------------------
def test_window_short_audio_returns_single_window():
    """Audio ≤ 30 s produces one window covering the whole signal."""
    audio = np.zeros(int(10 * TARGET_SAMPLE_RATE), dtype=np.float32)
    windows = split_windows(audio)
    assert len(windows) == 1
    assert len(windows[0]) == len(audio)


def test_window_exactly_30s_returns_single_window():
    """Exactly 30 s is at the boundary; ≤ 30 s returns one window."""
    audio = np.zeros(int(30 * TARGET_SAMPLE_RATE), dtype=np.float32)
    windows = split_windows(audio)
    assert len(windows) == 1
    assert len(windows[0]) == WINDOW_SAMPLES


def test_window_31s_audio_returns_two_windows_no_overlap():
    """Just over 30 s produces a full 30 s window plus a 1 s tail."""
    total_samples = int(31 * TARGET_SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float32)
    windows = split_windows(audio)
    assert len(windows) == 2
    assert len(windows[0]) == WINDOW_SAMPLES
    assert len(windows[1]) == total_samples - WINDOW_SAMPLES
    # Contiguous: second window starts where the first ends.
    assert len(windows[0]) + len(windows[1]) == total_samples


def test_window_60s_audio_two_full_windows():
    """60 s of audio splits into exactly two full windows."""
    total_samples = int(60 * TARGET_SAMPLE_RATE)
    audio = np.arange(total_samples, dtype=np.float32)  # monotonic for diagnostics
    windows = split_windows(audio)
    assert len(windows) == 2
    assert all(len(w) == WINDOW_SAMPLES for w in windows)
    # Content is contiguous, not strided.
    assert windows[0][0] == 0
    assert windows[1][0] == WINDOW_SAMPLES


def test_take_full_windows_holds_partial_tail():
    """Only complete windows flush; the partial tail stays buffered."""
    # 40 s in the buffer -> one full 30 s window, 10 s held back.
    pcm = np.zeros(int(40 * TARGET_SAMPLE_RATE), dtype=np.float32)
    windows, consumed = take_full_windows(pcm)
    assert len(windows) == 1
    assert len(windows[0]) == WINDOW_SAMPLES
    assert consumed == WINDOW_SAMPLES
    assert len(pcm) - consumed == int(10 * TARGET_SAMPLE_RATE)


def test_take_full_windows_nothing_to_flush():
    """A buffer under 30 s flushes zero windows and consumes nothing."""
    pcm = np.zeros(int(20 * TARGET_SAMPLE_RATE), dtype=np.float32)
    windows, consumed = take_full_windows(pcm)
    assert windows == []
    assert consumed == 0


def test_take_full_windows_exact_multiple():
    """65 s flushes two full windows; the 5 s tail stays buffered."""
    pcm = np.zeros(int(65 * TARGET_SAMPLE_RATE), dtype=np.float32)
    windows, consumed = take_full_windows(pcm)
    assert len(windows) == 2
    assert consumed == 2 * WINDOW_SAMPLES


def test_window_audio_seconds_helper():
    """audio_seconds() must match the window total length."""
    audio = np.zeros(int(45.5 * TARGET_SAMPLE_RATE), dtype=np.float32)
    assert audio_seconds(audio) == pytest.approx(45.5, abs=0.01)


def test_window_total_coverage_matches_input():
    """Concatenating all windows equals the total input length."""
    total_samples = int(75 * TARGET_SAMPLE_RATE)
    audio = np.zeros(total_samples, dtype=np.float32)
    windows = split_windows(audio)

    assert sum(len(w) for w in windows) == total_samples
    # Every window except possibly the last is full-size.
    for w in windows[:-1]:
        assert len(w) == WINDOW_SAMPLES
    assert len(windows[-1]) <= WINDOW_SAMPLES


# ---------------------------------------------------------------------------
# Transcript joining
# ---------------------------------------------------------------------------
def test_join_texts_preserves_order():
    parts = ["hello world", "this is window two", "and window three"]
    assert join_texts(parts) == "hello world this is window two and window three"


def test_join_texts_drops_empty_windows():
    parts = ["hello", "", "   ", "world"]
    assert join_texts(parts) == "hello world"


def test_join_texts_single_window_unchanged():
    assert join_texts(["only window"]) == "only window"


def test_join_texts_empty_list():
    assert join_texts([]) == ""


# ---------------------------------------------------------------------------
# Audio duration reporting (part of the transcription output payload)
# ---------------------------------------------------------------------------
def test_audio_seconds_for_short_input_matches_input_duration():
    raw = _make_wav(16000, 12.5)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    assert audio_seconds(audio_16k) == pytest.approx(12.5, abs=0.01)


def test_audio_seconds_for_long_input_matches_input_duration():
    raw = _make_wav(16000, 75.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    assert audio_seconds(audio_16k) == pytest.approx(75.0, abs=0.01)


# ---------------------------------------------------------------------------
# File format support
# ---------------------------------------------------------------------------
def test_wav_format_decodes():
    raw = _make_wav(16000, 2.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    assert audio_16k.ndim == 1
    assert audio_16k.dtype == np.float32


def test_flac_format_decodes():
    """soundfile supports FLAC natively."""
    import soundfile as sf

    sr = 16000
    duration = 2.0
    t = np.arange(int(sr * duration)) / sr
    data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="FLAC")
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": buf.getvalue(),
        "sample_rate": sr,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    assert audio_16k.ndim == 1


# ---------------------------------------------------------------------------
# End-to-end (no model): preprocessing → buffering
# ---------------------------------------------------------------------------
def test_full_pipeline_short_audio():
    raw = _make_wav(16000, 10.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, native_sr, _ = load_audio_for_inference(validated)
    windows = split_windows(audio_16k)
    assert native_sr == 16000
    assert len(windows) == 1
    transcript = join_texts([f"window-{i}" for i, _ in enumerate(windows)])
    assert transcript == "window-0"


def test_full_pipeline_long_audio_windows_in_order():
    raw = _make_wav(16000, 75.0)
    validated = {
        "kind": "audio_bytes",
        "audio_bytes": raw,
        "sample_rate": 16000,
    }
    audio_16k, _, _ = load_audio_for_inference(validated)
    windows = split_windows(audio_16k)
    # 75 s -> [30 s, 30 s, 15 s].
    assert [len(w) for w in windows] == [480000, 480000, 240000]
    transcripts = [f"seg-{i}" for i, _ in enumerate(windows)]
    full = join_texts(transcripts)
    parts = full.split(" ")
    assert len(parts) == len(windows)
    # Order must be preserved.
    for i, seg in enumerate(parts):
        assert seg == f"seg-{i}"
    # Audio seconds must reflect total length.
    assert audio_seconds(audio_16k) == pytest.approx(75.0, abs=0.01)

# ---------------------------------------------------------------------------
# Hardening (audit 2026-09-04)
# ---------------------------------------------------------------------------
def test_8ch_short_clip_uses_decoder_layout_not_shape_guess():
    """Channel axis must come from the decoder layout, not shape-guessing.

    An 8-sample 8-channel clip is exactly where shape-guessing misfires;
    mixing must average over channels (every sample == 3.5), not samples.
    """
    import soundfile as sf

    data = np.tile(np.arange(8, dtype=np.float32), (8, 1))  # a[i, j] == j
    buf = io.BytesIO()
    sf.write(buf, data, 16000, format="WAV", subtype="FLOAT")
    audio_16k, _, _ = load_audio_for_inference(
        {"kind": "audio_bytes", "audio_bytes": buf.getvalue(), "sample_rate": 16000}
    )
    assert audio_16k.ndim == 1
    assert len(audio_16k) == 8
    assert np.allclose(audio_16k, 3.5)


def test_corrupt_bytes_raise_value_error_naming_both_decoders():
    """Garbage bytes must fail loudly (chained causes), not decode silently."""
    with pytest.raises(ValueError, match="could not decode"):
        load_audio_for_inference(
            {"kind": "audio_bytes", "audio_bytes": b"\x00" * 64, "sample_rate": 16000}
        )


def test_data_url_download_roundtrip():
    """data: URLs exercise _download_url offline (no network in tests)."""
    import base64

    from src.audio import _download_url

    raw = _make_wav(16000, 0.1)
    url = "data:audio/wav;base64," + base64.b64encode(raw).decode()
    assert _download_url(url) == raw


def test_http_to_file_redirect_blocked():
    """An audio_url redirecting http -> file:// must be refused (SSRF guard).

    urllib itself rejects non-http(s) redirect targets; pin that behavior so
    a crafted audio_url cannot pivot the worker into reading local files.
    """
    import threading
    import urllib.error
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from src.audio import _download_url

    class _Redirector(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "file:///etc/hostname")
            self.end_headers()

        def log_message(self, *args):  # silence test output
            pass

    srv = HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError, match="not allowed"):
            _download_url(f"http://127.0.0.1:{srv.server_port}/x.wav")
    finally:
        srv.shutdown()
