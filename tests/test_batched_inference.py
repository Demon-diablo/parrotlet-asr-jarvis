"""Unit tests for batched inference and raw bytes pass-through."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_settings
from src.inference import _run_decoder_real, _transcribe_batched


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
    }
    env.update(kwargs)
    return load_settings(env)


def test_transcribe_batched_missing_submodules():
    settings = _get_test_settings()
    # Missing encoder
    fake_llm = SimpleNamespace(
        decoder=object(),
        projector=object(),
        processor=object(),
        tokenizer=object(),
    )
    with pytest.raises(AttributeError, match="missing required submodule"):
        _transcribe_batched(fake_llm, [np.zeros(16000)], settings)


def test_transcribe_batched_no_torch(monkeypatch):
    import src.inference as inf_mod

    settings = _get_test_settings()
    fake_llm = SimpleNamespace(
        encoder=object(),
        decoder=object(),
        projector=object(),
        processor=object(),
        tokenizer=object(),
    )
    monkeypatch.setattr(inf_mod, "torch", None)
    with pytest.raises(RuntimeError, match="torch is not available"):
        inf_mod._transcribe_batched(fake_llm, [np.zeros(16000)], settings)


def test_transcribe_batched_empty_windows():
    settings = _get_test_settings()
    fake_llm = SimpleNamespace(
        encoder=object(),
        decoder=object(),
        projector=object(),
        processor=object(),
        tokenizer=object(),
    )
    # When torch is mocked, empty chunks returns []
    fake_torch = SimpleNamespace()
    import src.inference as inf_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inf_mod, "torch", fake_torch)
    try:
        assert _transcribe_batched(fake_llm, [], settings) == []
    finally:
        monkeypatch.undo()


def test_transcribe_batched_end_to_end_mock(monkeypatch):
    import src.inference as inf_mod

    class _DummyTensor:
        def __init__(self, shape=(1, 10, 16)):
            self.shape = shape
            self.dtype = "float32"

        def to(self, *a, **kw):
            return self

        def unsqueeze(self, dim):
            return self

        def expand(self, *sizes):
            return _DummyTensor(shape=(sizes[0], self.shape[1], self.shape[2] if len(self.shape) > 2 else 16))

        def tolist(self):
            return [1, 2, 999, 4]  # 999 is audio token

        def __getitem__(self, idx):
            return self

        def __setitem__(self, idx, val):
            pass

    inference_mode_entered = []

    class _DummyTorch:
        float16 = "float16"
        int64 = "int64"
        cuda = SimpleNamespace(is_available=lambda: False)

        class inference_mode:
            def __enter__(self):
                inference_mode_entered.append(True)
                return self

            def __exit__(self, *a):
                pass

        @staticmethod
        def tensor(val, **kw):
            return _DummyTensor((1, 4))

        @staticmethod
        def ones_like(t, **kw):
            return _DummyTensor((1, 4))

        @staticmethod
        def zeros(*args, **kw):
            return _DummyTensor(shape=(args[0], args[1], args[2] if len(args) > 2 else 16))

    monkeypatch.setattr(inf_mod, "torch", _DummyTorch)

    class _Param:
        device = "cpu"
        dtype = "float32"

    recorded = {}

    class _Processor:
        @staticmethod
        def feature_extractor(audios, sampling_rate=None, return_tensors=None):
            recorded["audios_count"] = len(audios)
            recorded["sr"] = sampling_rate
            return SimpleNamespace(input_features=_DummyTensor((len(audios), 128, 3000)))

    class _Encoder:
        def parameters(self):
            return iter([_Param()])

        def __call__(self, features):
            return SimpleNamespace(last_hidden_state=_DummyTensor((features.shape[0], 1500, 128)))

    class _Projector:
        def __call__(self, embeddings):
            return _DummyTensor((embeddings.shape[0], 750, 64))

    class _Decoder:
        def parameters(self):
            return iter([_Param()])

        def get_input_embeddings(self):
            return lambda ids: _DummyTensor((ids.shape[0], 4, 64))

        def generate(self, **kwargs):
            recorded["gen_kwargs"] = kwargs
            return [[101, 102], [201, 202]]

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompt, **kw):
            return {"input_ids": [1, 2, 999, 4]}

        def convert_tokens_to_ids(self, token):
            return 999

        def decode(self, tokens, **kw):
            return f"decoded_{tokens[0]}"

    speech_llm = SimpleNamespace(
        encoder=_Encoder(),
        decoder=_Decoder(),
        projector=_Projector(),
        processor=_Processor(),
        tokenizer=_Tokenizer(),
        audio_token="<audio>",
    )

    settings = _get_test_settings(MAX_NEW_TOKENS="128")
    bad_words = [[42]]
    windows = [np.zeros(16000), np.zeros(16000)]
    out = _transcribe_batched(speech_llm, windows, settings, bad_words=bad_words)

    assert out == ["decoded_101", "decoded_201"]
    assert len(inference_mode_entered) == 1
    assert recorded["audios_count"] == 2
    assert recorded["sr"] == 16000
    assert recorded["gen_kwargs"]["max_new_tokens"] == 128
    assert recorded["gen_kwargs"]["do_sample"] is False
    assert recorded["gen_kwargs"]["repetition_penalty"] == 1.2
    assert recorded["gen_kwargs"]["use_cache"] is True
    assert recorded["gen_kwargs"]["bad_words_ids"] == [[42]]


def test_run_decoder_real_batched_success(monkeypatch):
    import src.inference as inf_mod
    import src.audio as audio_module

    class _BatchedLLM:
        encoder = object()
        decoder = object()
        projector = object()
        processor = object()
        tokenizer = object()

    bundle = SimpleNamespace(speech_llm=_BatchedLLM(), banned_token_ids=[10])
    validated = {"kind": "audio_bytes", "audio_bytes": b"x", "sample_rate": 16000}
    settings = _get_test_settings()

    monkeypatch.setattr(
        inf_mod,
        "_transcribe_batched",
        lambda speech_llm, windows, settings, bad_words=None: ["batched_1", "batched_2", "batched_3"],
    )

    real_load = audio_module.load_audio_for_inference
    audio_module.load_audio_for_inference = lambda v: (
        np.zeros(16000 * 65, dtype=np.float32), 16000, 16000
    )
    try:
        out = _run_decoder_real(bundle, validated, settings)
    finally:
        audio_module.load_audio_for_inference = real_load

    # 65 s buffers into [30 s, 30 s, 5 s] and flushes in one batched pass.
    assert out["transcript"] == "batched_1 batched_2 batched_3"
    assert out["ban_applied"] is True
    assert out["flushed_windows"] == 3
    assert out["decode_path"] == "buffered:B=3"


def test_run_decoder_real_batched_fallback_to_sequential():
    import src.audio as audio_module

    calls = []

    class _FallbackLLM:
        def transcribe(self, audio, sr, **kw):
            calls.append(len(audio))
            return f"window_{len(calls)}"

    bundle = SimpleNamespace(speech_llm=_FallbackLLM(), banned_token_ids=[])
    validated = {"kind": "audio_bytes", "audio_bytes": b"x", "sample_rate": 16000}
    settings = _get_test_settings()

    real_load = audio_module.load_audio_for_inference
    audio_module.load_audio_for_inference = lambda v: (
        np.zeros(16000 * 65, dtype=np.float32), 16000, 16000
    )
    try:
        out = _run_decoder_real(bundle, validated, settings)
    finally:
        audio_module.load_audio_for_inference = real_load

    assert out["flushed_windows"] == 3
    assert len(calls) == out["flushed_windows"]
    assert "window_1" in out["transcript"]
    assert out["decode_path"] == "fallback-sequential"


def test_transcribe_bytes_source_passes_raw_bytes():
    """Verify source of transcribe_bytes doesn't b64encode audio_bytes."""
    from src import server

    with open(server.__file__, "r", encoding="utf-8") as f:
        src = f.read()

    start = src.find("def transcribe_bytes(")
    assert start != -1
    end = src.find("def ", start + 1)
    method_src = src[start:end] if end != -1 else src[start:]

    assert '{"audio": audio_bytes}' in method_src
    assert "b64encode" not in method_src
