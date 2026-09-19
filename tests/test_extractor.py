"""Unit tests for clinical schema and extractor module (no GPU, no network required)."""

import json
import sys
from unittest.mock import MagicMock, patch
import pytest

from src.schema import (
    DENSE_SYSTEM_PROMPT,
    STANDARD_SYSTEM_PROMPT,
    get_default_system_prompt,
    normalize_frequency,
    parse_dense_to_prescriptions,
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
    assert get_extractor_backend() == "sglang"


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


def test_extract_prescriptions_sglang_mock():
    reset_extractor()
    sample_json = json.dumps({
        "medications": [
            {
                "spoken_name": "Pan 40",
                "strength": "40mg",
                "dosage_form": "tablet",
                "frequency": "OD (daily)",
            }
        ]
    })

    mock_engine = MagicMock()
    mock_engine.generate.return_value = {
        "text": f"```json\n{sample_json}\n```",
        "meta_info": {"completion_tokens": 15},
    }

    with patch("src.extractor.load_extractor", return_value=mock_engine):
        with patch.dict("src.extractor._EXTRACTOR_CACHE", {"backend": "sglang", "engine": mock_engine, "loaded": True}):
            res = extract_prescriptions("Prescribe Pan 40 OD")
            assert res["valid_json"] is True
            assert res["medications_count"] == 1
            assert res["medications"][0]["spoken_name"] == "Pan 40"
            assert res["tokens_generated"] == 15


def test_extract_prescriptions_sglang_streaming_tokens():
    reset_extractor()
    sample_json = json.dumps({
        "medications": [
            {
                "spoken_name": "Paracetamol",
                "strength": "500mg",
            }
        ]
    })
    chunks = [
        {"text": "```json\n{"},
        {"text": f"```json\n{sample_json}"},
        {"text": f"```json\n{sample_json}\n```"},
    ]

    mock_engine = MagicMock()
    mock_engine.generate.return_value = iter(chunks)

    with patch("src.extractor.load_extractor", return_value=mock_engine):
        with patch.dict("src.extractor._EXTRACTOR_CACHE", {"backend": "sglang", "engine": mock_engine, "loaded": True}):
            events = list(extract_prescriptions_stream("Paracetamol 500mg"))
            event_types = [e["event"] for e in events]
            assert "extraction_start" in event_types
            assert "token" in event_types
            assert "extraction_complete" in event_types
            complete_evt = next(e for e in events if e["event"] == "extraction_complete")
            assert complete_evt["valid_json"] is True
            assert complete_evt["medications"][0]["spoken_name"] == "Paracetamol"


def test_parse_dense_to_prescriptions():
    raw = (
        "Pan|40 mg|Tablet|OD||ES|5 days|empty stomach\n"
        "Drotin MF||Tablet|BD||DF|5 days|with breakfast\n"
        "Tablet Buscopan|10 mg||BD|||5 days|with breakfast"
    )
    transcript = "Tablet Pan 40 OD with empty stomach ES. Tablet Drotin MF BD five days with breakfast DF. Tablet Buscopan 10 mg BD five days."
    meds = parse_dense_to_prescriptions(raw, transcript=transcript)
    assert len(meds) == 3
    assert meds[0]["spoken_name"] == "Pan"
    assert meds[0]["strength"] == "40 mg"
    assert meds[0]["dosage_form"] == "Tablet"
    assert meds[0]["frequency"] == "ES"
    assert meds[0]["duration"] == "5 days"
    assert meds[0]["instructions"] == "empty stomach"
    assert "Pan 40" in meds[0]["source_text"]

    # Auto-detected dosage form from prefix
    assert meds[2]["spoken_name"] == "Buscopan"
    assert meds[2]["dosage_form"] == "Tablet"


def test_parse_prescriptions_dense_fallback():
    raw = "Pan|40 mg|Tablet|OD||ES|5 days|empty stomach"
    res = parse_prescriptions_json(raw)
    assert res["valid_json"] is True
    assert res["medications_count"] == 1
    assert res["medications"][0]["spoken_name"] == "Pan"


def test_get_default_system_prompt(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_MODE", "dense")
    assert get_default_system_prompt() == DENSE_SYSTEM_PROMPT

    monkeypatch.setenv("EXTRACTOR_MODE", "json")
    assert get_default_system_prompt() == STANDARD_SYSTEM_PROMPT

