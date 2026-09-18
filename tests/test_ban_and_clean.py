"""Tests for the language lock: decoder script ban + transcript cleaner."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_settings  # noqa: E402
from src.inference import (  # noqa: E402
    _get_banned_ids_for_bundle,
    clean_transcript,
    detect_scripts,
)
from src.model import (  # noqa: E402
    _is_banned_char,
    _resolve_banned_token_ids,
    _token_has_banned_script,
)


def _base_env(**over):
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
    }
    env.update(over)
    return env


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------
def test_ban_and_clean_default_on():
    s = load_settings(_base_env())
    assert s.ban_script_tokens is True
    assert s.clean_transcript is True


def test_ban_and_clean_can_be_disabled():
    s = load_settings(_base_env(BAN_SCRIPT_TOKENS="0", CLEAN_TRANSCRIPT="false"))
    assert s.ban_script_tokens is False
    assert s.clean_transcript is False


# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------
def test_banned_char_ranges():
    assert _is_banned_char("अ")  # Devanagari
    assert _is_banned_char("ట")  # Telugu
    assert _is_banned_char("એ")  # Gujarati
    assert _is_banned_char("ফ")  # Bengali
    assert not _is_banned_char("A")
    assert not _is_banned_char(" ")


def test_token_scan():
    assert _token_has_banned_script("एक") is True
    assert _token_has_banned_script("▁ Tablet") is False
    assert _token_has_banned_script("") is False


def test_resolve_banned_ids_with_fake_tokenizer():
    class _Tok:
        def get_vocab(self):
            return {"▁Tablet": 1, "एक": 2, "టూ": 3, "hello": 4}

        def decode(self, ids):
            inv = {1: " Tablet", 2: "एक", 3: "టూ", 4: "hello"}
            return inv.get(ids[0], "")

    assert _resolve_banned_token_ids(_Tok()) == [2, 3]
    assert _resolve_banned_token_ids(None) == []
    assert _resolve_banned_token_ids(object()) == []


def test_get_banned_ids_prefers_bundle():
    bundle = SimpleNamespace(banned_token_ids=[7, 9])
    assert _get_banned_ids_for_bundle(bundle) == [7, 9]
    assert _get_banned_ids_for_bundle(SimpleNamespace()) == []


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------
def test_cleaner_strips_noise_tags():
    raw = "Tablet Pan 40 <bird_squawk> for ten days <Persistent-noise-start> hi <hi-en>"
    assert clean_transcript(raw) == "Tablet Pan 40 for ten days hi"


def test_cleaner_unwraps_bracket_gloss():
    assert clean_transcript("वन [One] सिक्स्टी [Sixty]") == "वन One सिक्स्टी Sixty"


def test_cleaner_collapses_whitespace_and_punct():
    assert clean_transcript("a   b ,  c") == "a b, c"
    assert clean_transcript("") == ""
    assert clean_transcript(None) == ""


def test_detect_scripts_flags():
    d = detect_scripts("Tablet एक गोली టూ")
    assert d["has_devanagari"] is True
    assert d["has_telugu"] is True
    assert d["has_indic"] is True
    clean = detect_scripts("Tablet one daily")
    assert clean["has_indic"] is False


# ---------------------------------------------------------------------------
# Inference wiring: ban passed to transcribe, fallback when unsupported
# ---------------------------------------------------------------------------
def test_decoder_passes_bad_words_ids():
    import numpy as np

    from src import inference as inf_mod

    calls = {}

    class _Speech:
        def transcribe(self, audio, sr, max_new_tokens=None, bad_words_ids=None, **kw):
            calls["bad"] = bad_words_ids
            return "ok <bird_squawk>"

    bundle = SimpleNamespace(
        speech_llm=_Speech(), banned_token_ids=[11, 22],
    )
    validated = {"kind": "audio_bytes", "audio_bytes": b"x", "sample_rate": 16000}
    settings = load_settings(_base_env())

    # Stub audio layer: 1s of silence, single window.
    import src.audio as audio_module

    real_load = audio_module.load_audio_for_inference
    audio_module.load_audio_for_inference = lambda v: (
        np.zeros(16000, dtype=np.float32), 16000, 16000,
    )
    try:
        out = inf_mod._run_decoder_real(bundle, validated, settings)
    finally:
        audio_module.load_audio_for_inference = real_load
    assert calls["bad"] == [[11], [22]]
    assert out["ban_applied"] is True
    assert out["banned_token_count"] == 2


def test_decoder_falls_back_without_bad_words_support():
    import numpy as np

    from src import inference as inf_mod

    class _OldSpeech:
        def transcribe(self, audio, sr, max_new_tokens=None, **kw):
            if "bad_words_ids" in kw:
                raise TypeError("unexpected keyword 'bad_words_ids'")
            return "plain"

    bundle = SimpleNamespace(speech_llm=_OldSpeech(), banned_token_ids=[5])
    validated = {"kind": "audio_bytes", "audio_bytes": b"x", "sample_rate": 16000}
    settings = load_settings(_base_env())
    import src.audio as audio_module

    real_load = audio_module.load_audio_for_inference
    audio_module.load_audio_for_inference = lambda v: (
        np.zeros(16000, dtype=np.float32), 16000, 16000,
    )
    try:
        out = inf_mod._run_decoder_real(bundle, validated, settings)
    finally:
        audio_module.load_audio_for_inference = real_load
    assert out["transcript"] == "plain"
    assert out["ban_applied"] is False


def test_postprocess_cleans_and_reports_scripts():
    from src.inference import _postprocess

    settings = load_settings(_base_env())
    out = _postprocess(
        {"transcript": "Tablet <bird_squawk> એક",
         "audio_seconds": 1.0, "native_sample_rate": 16000,
         "flushed_windows": 1, "ban_applied": True, "banned_token_count": 3},
        settings, 0.01,
    )
    assert out["transcript"] == "Tablet એક"
    assert out["transcript_raw"] == "Tablet <bird_squawk> એક"
    assert out["detected_scripts"]["has_gujarati"] is True
    assert out["ban_applied"] is True
    assert out["banned_token_count"] == 3
