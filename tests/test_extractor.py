"""Unit tests for clinical schema and extractor module (no GPU, no network required)."""

import json
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.schema import (
    STANDARD_SYSTEM_PROMPT,
    normalize_frequency,
    parse_prescriptions_json,
    strip_fences,
)
from src.extractor import (
    extract_prescriptions,
    extract_prescriptions_stream,
    get_extractor_backend,
    get_medgemma_model_id,
    reset_extractor,
)


def test_strip_fences():
    raw = "```json\n{\"test\": 1}\n```"
    assert strip_fences(raw) == '{"test": 1}'

    raw2 = "```\n[1, 2, 3]\n```"
    assert strip_fences(raw2) == "[1, 2, 3]"

    raw3 = '  {"hello": "world"}  '
    assert strip_fences(raw3) == '{"hello": "world"}'


def test_normalize_frequency():
    assert normalize_frequency("OD") == "OD (daily)"
    assert normalize_frequency("BD") == "BD (twice a day)"
    assert normalize_frequency("tds") == "TDS (thrice a day)"
    assert normalize_frequency("qid") == "QID (4 times a day)"
    assert normalize_frequency("sos") == "SOS (as needed)"
    assert normalize_frequency("hs") == "HS (bedtime)"
    assert normalize_frequency("custom freq") == "custom freq"
    assert normalize_frequency(None) is None
    assert normalize_frequency("") is None


def test_parse_prescriptions_json():
    json_text = json.dumps({
        "medications": [
            {
                "spoken_name": "Augmentin",
                "strength": "625mg",
                "dosage_form": "tablet",
                "dose": "1",
                "route": "oral",
                "frequency": "BD",
                "duration": "5 days",
                "instructions": "after food",
                "source_text": "augmentin 625 bd 5 days",
            }
        ]
    })
    res = parse_prescriptions_json(json_text)
    assert res["valid_json"] is True
    assert res["medications_count"] == 1
    assert res["medications"][0]["spoken_name"] == "Augmentin"
    assert res["medications"][0]["frequency"] == "BD"
    assert res["medications"][0]["strength"] == "625mg"


def test_parse_prescriptions_invalid_json():
    res = parse_prescriptions_json("not valid json at all")
    assert res["valid_json"] is False
    assert res["medications"] == []
    assert res["medications_count"] == 0


def test_extract_prescriptions_empty():
    res = extract_prescriptions("")
    assert res["medications_count"] == 0
    assert res["raw_text"] == ""

    res_ws = extract_prescriptions("   \n  ")
    assert res_ws["medications_count"] == 0


def test_get_extractor_backend(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_BACKEND", "vllm")
    assert get_extractor_backend() == "vllm"

    monkeypatch.setenv("EXTRACTOR_BACKEND", "sglang")
    assert get_extractor_backend() == "sglang"

    monkeypatch.setenv("EXTRACTOR_BACKEND", "transformers")
    assert get_extractor_backend() == "transformers"

    monkeypatch.setenv("EXTRACTOR_BACKEND", "none")
    assert get_extractor_backend() == "none"

    monkeypatch.setenv("EXTRACTOR_BACKEND", "invalid_choice")
    assert get_extractor_backend() == "vllm"


def test_extract_prescriptions_backend_none(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_BACKEND", "none")
    res = extract_prescriptions("Prescribe Pan 40 OD")
    assert res["valid_json"] is False
    assert "disabled" in res.get("error", "")


def test_extract_prescriptions_mock():
    reset_extractor()
    sample_json = json.dumps({
        "medications": [
            {
                "spoken_name": "Pantoprazole",
                "strength": "40mg",
                "dosage_form": "tablet",
                "dose": "1",
                "route": "oral",
                "frequency": "OD",
                "duration": "14 days",
                "instructions": "before breakfast",
                "source_text": "pan 40 od empty stomach",
            }
        ]
    })

    mock_vllm = MagicMock()
    mock_engine = MagicMock()
    mock_output = MagicMock()
    mock_out_obj = MagicMock()
    mock_out_obj.text = f"```json\n{sample_json}\n```"
    mock_out_obj.token_ids = [1, 2, 3, 4]
    mock_output.outputs = [mock_out_obj]
    mock_engine.chat.return_value = [mock_output]

    with patch.dict(sys.modules, {"vllm": mock_vllm}):
        with patch("src.extractor.load_extractor", return_value=mock_engine):
            with patch.dict("src.extractor._EXTRACTOR_CACHE", {"backend": "vllm", "engine": mock_engine, "loaded": True}):
                res = extract_prescriptions("Prescribe Pan 40 OD")
                assert res["valid_json"] is True
                assert res["medications_count"] == 1
                assert res["medications"][0]["spoken_name"] == "Pantoprazole"


def test_extract_prescriptions_stream_mock():
    sample_json = json.dumps({"medications": []})

    with patch("src.extractor.extract_prescriptions", return_value={
        "valid_json": True,
        "medications": [],
        "medications_count": 0,
        "latency_seconds": 0.05,
        "tokens_generated": 10,
        "throughput_tok_s": 200.0,
        "raw_text": sample_json,
    }):
        events = list(extract_prescriptions_stream("Patient takes paracetamol"))
        assert len(events) == 2
        assert events[0]["event"] == "extraction_start"
        assert events[1]["event"] == "extraction_complete"
