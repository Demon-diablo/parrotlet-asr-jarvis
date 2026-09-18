"""Audio preprocessing for Parrotlet-a 2.5 Pro.

Responsibilities:

1. Load arbitrary audio (bytes / file / URL) into a float32 mono numpy array.
2. Resample to 16 kHz mono using the same ``soxr_vhq`` resampler the upstream
   Parrotlet loader uses (``librosa.resample(..., res_type='soxr_vhq')``).
3. Buffer audio longer than the Whisper model's 30 s window into
   non-overlapping 30 s windows (a trailing partial window is kept as-is
   and flushed on ``is_last``).
4. Reassemble per-window transcripts into a single ordered transcript.

Why we do this here (not inside the model loader):

- The model wrapper's ``preprocess_audio`` resamples, but it assumes mono and
  hands the entire array to ``WhisperFeatureExtractor``, which silently
  truncates inputs over ``n_samples=480000`` (30 s @ 16 kHz). Anything past
  30 s is lost.
- The wrapper has no concept of buffering or transcript reassembly.
- Buffering belongs at the **inference / input contract** layer, so the
  API layer and the model loader stay unchanged.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from typing import Any, Dict, List, Tuple

log = logging.getLogger("parrotlet.audio")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000  # Hz — what Parrotlet / Whisper expects
TARGET_CHANNELS = 1         # mono
WINDOW_SECONDS = 30.0       # Whisper feature extractor window
# Whisper's feature extractor hard-caps n_samples at 480000 (= 30 s @ 16 kHz).
# Audio beyond that is silently truncated by the feature extractor, so we
# buffer into non-overlapping windows BEFORE handing to the model.

WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_SAMPLE_RATE)               # 480_000


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _decode_bytes_to_numpy(raw: bytes) -> Tuple["numpy.ndarray", int, bool]:
    """Decode raw audio into ``(array, sample_rate, channels_first)``.

    wav / flac / ogg / mp3 via libsndfile first (returns ``(samples,
    channels)``), falling back to librosa/ffmpeg via a temp file (returns
    ``(channels, samples)``). The layout flag travels with the array so
    mono-mixing never has to guess the channel axis from the shape (which
    misfires on e.g. short >8-channel clips).
    """
    import numpy as np
    import soundfile as sf

    try:
        audio_np, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return audio_np, int(sr), False
    except Exception as sf_exc:
        # soundfile couldn't decode — try librosa via temp file.
        import tempfile

        import librosa

        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            audio_np, sr = librosa.load(tmp_path, sr=None, mono=False)
        except Exception as lo_exc:
            raise ValueError(
                f"could not decode audio bytes (soundfile: {sf_exc}; "
                f"librosa fallback: {lo_exc})"
            ) from lo_exc
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return audio_np, int(sr), True


def _to_mono(audio_np: "numpy.ndarray", channels_first: bool) -> "numpy.ndarray":
    """Mix down to mono. Returns float32 1-D.

    ``channels_first`` selects the channel axis explicitly: ``True`` for
    ``(channels, samples)`` (librosa layout), ``False`` for ``(samples,
    channels)`` (soundfile layout). 1-D input is already mono.
    """
    import numpy as np

    a = np.asarray(audio_np)
    if a.ndim == 1:
        return a.astype(np.float32, copy=False)
    if a.ndim == 2:
        axis = 0 if channels_first else 1
        if a.shape[axis] == 0:
            raise ValueError(f"audio has no channels: shape {a.shape}")
        return a.mean(axis=axis).astype(np.float32, copy=False)
    raise ValueError(f"unsupported audio shape: {a.shape}")


_RESAMPLER_CACHE: dict = {}


def _resample_to_16k(
    audio_mono: "numpy.ndarray", orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE
) -> "numpy.ndarray":
    """Resample mono audio to ``target_sr``.

    Uses GPU torchaudio.transforms.Resample when CUDA is available (cached to avoid
    reconstructing polyphase filter banks on every request), or librosa with res_type="kaiser_fast"
    (falling back to "soxr_qq" if resampy is unavailable).
    """
    import numpy as np

    if orig_sr == target_sr:
        return np.asarray(audio_mono, dtype=np.float32)

    # 1. Try GPU resampling via torchaudio if CUDA is available
    try:
        import torch
        import torchaudio.transforms as T

        if torch.cuda.is_available():
            cache_key = (int(orig_sr), int(target_sr))
            resampler = _RESAMPLER_CACHE.get(cache_key)
            if resampler is None:
                resampler = T.Resample(orig_sr, target_sr).to("cuda")
                _RESAMPLER_CACHE[cache_key] = resampler
            tensor = torch.from_numpy(np.asarray(audio_mono, dtype=np.float32)).to("cuda")
            resampled = resampler(tensor)
            return resampled.cpu().numpy()
    except Exception:
        pass

    # 2. Fallback to librosa with kaiser_fast or soxr_qq
    import librosa

    try:
        return np.asarray(
            librosa.resample(
                np.asarray(audio_mono, dtype=np.float32),
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type="kaiser_fast",
            ),
            dtype=np.float32,
        )
    except Exception:
        return np.asarray(
            librosa.resample(
                np.asarray(audio_mono, dtype=np.float32),
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type="soxr_qq",
            ),
            dtype=np.float32,
        )


# Maximum bytes fetched from an ``audio_url``. Mirrors the validated-bytes
# cap (+1 so oversize downloads are detected, not silently truncated).
_MAX_URL_BYTES = 50 * 1024 * 1024 + 1


def _download_url(url: str) -> bytes:
    """Fetch ``url`` with a bounded read so a huge response cannot OOM the worker.

    Redirects away from http(s) (e.g. to file://) are refused by urllib
    itself (HTTPError); only http(s) targets are followed.
    """
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read(_MAX_URL_BYTES)
    if len(raw) >= _MAX_URL_BYTES:
        raise ValueError("audio_url response exceeds 50 MiB cap")
    return raw


def trim_audio_silence(
    audio_16k: "numpy.ndarray",
    top_db: float = 35.0,
    margin_samples: int = 1600,
) -> "numpy.ndarray":
    """Trim leading and trailing silence from 16 kHz mono audio while keeping margins.

    Uses ``librosa.effects.trim(audio_16k, top_db=top_db)`` or energy threshold.
    If trimmed non-silent segment is found:
      start_idx = max(0, start_idx - margin_samples)
      end_idx = min(len(audio_16k), end_idx + margin_samples)
    Returns trimmed slice audio_16k[start_idx:end_idx] if duration >= 0.2s, else original.
    """
    if audio_16k is None or len(audio_16k) == 0:
        return audio_16k

    try:
        import librosa

        _, index = librosa.effects.trim(audio_16k, top_db=top_db)
        start_idx = max(0, int(index[0]) - margin_samples)
        end_idx = min(len(audio_16k), int(index[1]) + margin_samples)
        if end_idx > start_idx and (end_idx - start_idx) >= int(0.2 * TARGET_SAMPLE_RATE):
            return audio_16k[start_idx:end_idx]
    except Exception:
        pass
    return audio_16k


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_audio_for_inference(validated: Dict[str, Any]) -> Tuple["numpy.ndarray", int, int]:
    """Load + resample + mono-mix the audio for one validated payload.

    Returns ``(audio_16k_mono, original_sample_rate, total_16k_samples)``.

    ``audio_16k_mono`` is always ``float32`` shape ``(n,)`` at 16 kHz. Any
    input format (wav / flac / ogg / mp3 / arbitrary sample rate / mono /
    multi-channel) is handled uniformly. The returned ``audio_seconds`` can
    be computed as ``len(audio_16k_mono) / 16000``.
    """
    kind = validated["kind"]

    if kind == "audio_bytes":
        raw: bytes = validated["audio_bytes"]
        audio_np, sr, channels_first = _decode_bytes_to_numpy(raw)
    elif kind == "audio_path":
        import librosa

        # librosa with mono=False returns (channels, samples).
        audio_np, sr = librosa.load(validated["audio_path"], sr=None, mono=False)
        channels_first = True
    elif kind == "audio_url":
        raw = _download_url(validated["audio_url"])
        audio_np, sr, channels_first = _decode_bytes_to_numpy(raw)
    else:
        raise ValueError(f"unsupported audio kind: {kind}")

    mono = _to_mono(audio_np, channels_first)
    audio_16k = _resample_to_16k(mono, sr, TARGET_SAMPLE_RATE)
    if len(audio_16k) > 0 and bool(validated.get("trim_silence", False)):
        audio_16k = trim_audio_silence(audio_16k, top_db=35.0, margin_samples=3200)
    return audio_16k, int(sr), int(audio_16k.shape[0])


# ---------------------------------------------------------------------------
# Buffer windowing (non-overlapping 30 s windows)
# ---------------------------------------------------------------------------
def split_windows(
    audio_16k: "numpy.ndarray",
    *,
    window_seconds: float = WINDOW_SECONDS,
) -> List["numpy.ndarray"]:
    """Split 16 kHz mono audio into non-overlapping windows ≤ ``window_seconds``.

    Returns a list with length ``>= 1``. Audio shorter than ``window_seconds``
    yields a single window covering the whole signal. Longer audio is split
    into contiguous ``window_seconds`` windows; the final window holds the
    remainder (possibly shorter than a full window).
    """
    import numpy as np

    audio = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
    total = audio.shape[0]
    window_samples = int(window_seconds * TARGET_SAMPLE_RATE)

    if total <= window_samples:
        return [audio]

    windows: List["numpy.ndarray"] = []
    start = 0
    while start < total:
        end = min(start + window_samples, total)
        windows.append(audio[start:end])
        if end == total:
            break
        start = end
    return windows


def take_full_windows(
    pcm: "numpy.ndarray",
    *,
    window_seconds: float = WINDOW_SECONDS,
) -> Tuple[List["numpy.ndarray"], int]:
    """Take complete windows from a session buffer.

    Returns ``(windows, consumed_samples)`` where ``windows`` holds every
    full ``window_seconds`` window available in ``pcm`` and
    ``consumed_samples`` is ``len(windows) * window_samples``. The caller
    keeps ``pcm[consumed_samples:]`` buffered; a trailing partial window is
    only flushed on ``is_last`` via :func:`split_windows`.
    """
    import numpy as np

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    window_samples = int(window_seconds * TARGET_SAMPLE_RATE)
    full = audio.shape[0] // window_samples
    windows = [
        audio[i * window_samples:(i + 1) * window_samples] for i in range(full)
    ]
    return windows, full * window_samples


def join_texts(parts: List[str]) -> str:
    """Concatenate per-window transcripts in order.

    Trims whitespace on each segment and joins with single spaces. Empty
    segments are skipped (defensive: a window that yields no transcription
    does not inject a stray space).
    """
    cleaned = [str(p).strip() for p in parts]
    cleaned = [c for c in cleaned if c]
    return " ".join(cleaned)


# ---------------------------------------------------------------------------
# Total audio duration helper
# ---------------------------------------------------------------------------
def audio_seconds(audio_16k: "numpy.ndarray") -> float:
    """Return total audio length in seconds at 16 kHz."""
    import numpy as np

    return float(np.asarray(audio_16k).shape[0]) / TARGET_SAMPLE_RATE


def make_16k(audio_16k: "numpy.ndarray") -> "numpy.ndarray":
    """Helper for callers that just need the canonical 16 kHz mono array."""
    import numpy as np

    return np.asarray(audio_16k, dtype=np.float32)