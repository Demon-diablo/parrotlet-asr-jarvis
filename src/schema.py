"""Clinical prescription schemas, prompts, and normalization utilities."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Frequency Normalization Map
# ---------------------------------------------------------------------------
_FREQUENCY_CANONICAL_MAP = {
    "od": "OD (daily)",
    "daily": "OD (daily)",
    "once daily": "OD (daily)",
    "bd": "BD (twice a day)",
    "twice daily": "BD (twice a day)",
    "tds": "TDS (thrice a day)",
    "thrice daily": "TDS (thrice a day)",
    "qid": "QID (4 times a day)",
    "hs": "HS (bedtime)",
    "bedtime": "HS (bedtime)",
    "sos": "SOS (as needed)",
    "as needed": "SOS (as needed)",
    "12h": "12 hourly",
    "12 hourly": "12 hourly",
    "15d": "every 15 days",
    "every 15 days": "every 15 days",
    "6x": "6 times a day",
    "3x": "3 times a day",
}

_FORM_PREFIX_RE = re.compile(
    r"^(?:tablet|tab\.?|capsule|cap\.?|syrup|syr\.?|injection|inj\.?|inhaler)\s+",
    re.IGNORECASE,
)


def normalize_frequency(freq: Optional[str]) -> Optional[str]:
    """Normalize short medical frequency abbreviations to full clinical canonical form."""
    if not freq:
        return None
    cleaned = freq.strip().strip("\"'")
    if not cleaned or cleaned.lower() in ("null", "none", ""):
        return None
    if "(" in cleaned and ")" in cleaned:
        return cleaned
    lookup = cleaned.lower()
    return _FREQUENCY_CANONICAL_MAP.get(lookup, cleaned)


# ---------------------------------------------------------------------------
# Clinical Extraction Prompts
# ---------------------------------------------------------------------------
STANDARD_SYSTEM_PROMPT = (
    "Extract literal medication mentions from the doctor transcript. "
    'Return valid JSON only in this format: {"medications":[{"spoken_name":"","strength":"","dosage_form":"","dose":"","route":"","frequency":"","duration":"","instructions":"","source_text":""}]}. '
    "Do not prescribe or correct medicine spellings."
)

DENSE_SYSTEM_PROMPT = (
    "Extract literal medication mentions from the doctor transcript. Do not prescribe or correct medicine spellings.\n"
    "Output exactly ONE line per medication in this pipe-delimited format:\n"
    "spoken_name|strength|dosage_form|dose|route|frequency|duration|instructions\n\n"
    "Rules:\n"
    "- Output ONLY the pipe-delimited lines. No markdown code fences, headers, or explanatory text.\n"
    "- Do not include form words (Tablet, Capsule, Inhaler, Syrup) in spoken_name if dosage_form is specified.\n"
    "- Leave omitted fields blank between pipes. Do not shift columns.\n"
    "- Extract literal spoken mentions. Do not invent or correct medicine spellings.\n\n"
    "Example output:\n"
    "Pan|40 mg|Tablet|OD||ES|5 days|empty stomach\n"
    "Drotin MF||Tablet|BD||DF|5 days|with breakfast\n"
    "Foracort|200 mg|Inhaler|two puffs||twelve hourly||two puffs twelve hourly"
)


def get_default_system_prompt() -> str:
    """Resolve default system prompt based on EXTRACTOR_MODE ('dense' vs 'json')."""
    mode = (os.getenv("EXTRACTOR_MODE") or os.getenv("EXTRACTOR_PROMPT_MODE") or "dense").lower()
    return STANDARD_SYSTEM_PROMPT if mode == "json" else DENSE_SYSTEM_PROMPT


MEDICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spoken_name": {"type": "string"},
                    "strength": {"type": "string"},
                    "dosage_form": {"type": "string"},
                    "dose": {"type": "string"},
                    "route": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "instructions": {"type": "string"},
                    "source_text": {"type": "string"},
                },
                "required": ["spoken_name"],
            },
        }
    },
    "required": ["medications"],
}


def strip_fences(text: str) -> str:
    """Strip markdown code blocks, language tags, and extract raw JSON object or array."""
    text = (text or "").strip()
    if "<unused95>" in text:
        text = text.split("<unused95>")[-1].strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
        if blocks:
            text = blocks[-1].strip()
        else:
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        last_bracket = text.rfind("]")
        if last_bracket > first_bracket:
            text = text[first_bracket : last_bracket + 1]
    elif first_brace != -1:
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            text = text[first_brace : last_brace + 1]
    return text


def parse_dense_to_prescriptions(raw_text: str, transcript: str = "") -> List[Dict[str, str]]:
    """Parse pipe-delimited lines into canonical 9-key clinical prescription dicts."""
    medications: List[Dict[str, str]] = []
    if not raw_text:
        return medications

    cleaned = raw_text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        pipe_parts = [p for p in parts if "|" in p]
        cleaned = "\n".join(pipe_parts) if pipe_parts else parts[1 if len(parts) > 1 else 0]
        if cleaned.startswith(("text", "csv", "markdown", "plaintext")):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned

    for line in cleaned.strip().splitlines():
        line = line.strip().strip("- ")
        if not line or "|" not in line:
            continue

        cols = [c.strip() for c in line.split("|")]
        if not cols or not cols[0]:
            continue

        spoken_name = cols[0]
        strength = cols[1] if len(cols) > 1 else ""
        dosage_form = cols[2] if len(cols) > 2 else ""
        dose = cols[3] if len(cols) > 3 else ""
        route = cols[4] if len(cols) > 4 else ""
        frequency = cols[5] if len(cols) > 5 else ""
        duration = cols[6] if len(cols) > 6 else ""
        instructions = cols[7] if len(cols) > 7 else ""
        source_text = cols[8] if len(cols) > 8 else ""

        # Auto-detect dosage_form if not explicitly in col 2 but in spoken_name
        if not dosage_form:
            m = _FORM_PREFIX_RE.match(spoken_name)
            if m:
                dosage_form = m.group(0).strip().capitalize()
                spoken_name = spoken_name[m.end():].strip()

        # Reconstruct source_text snippet from transcript if missing
        if not source_text and transcript:
            idx = transcript.lower().find(spoken_name.lower())
            if idx != -1:
                start = max(0, transcript.rfind(".", 0, idx) + 1)
                end = transcript.find(".", idx)
                if end == -1:
                    end = len(transcript)
                source_text = transcript[start:end].strip()
        if not source_text:
            parts_str = [p for p in [dosage_form, spoken_name, strength, dose, route, frequency, duration, instructions] if p]
            source_text = " ".join(parts_str)

        medications.append({
            "spoken_name": spoken_name,
            "strength": strength,
            "dosage_form": dosage_form,
            "dose": dose,
            "route": route,
            "frequency": frequency,
            "duration": duration,
            "instructions": instructions,
            "source_text": source_text,
        })
    return medications


def parse_prescriptions_json(raw_text: str, transcript: str = "") -> Dict[str, Any]:
    """Parse raw LLM output text into structured clinical medications dictionary.

    Supports:
    1. Standard JSON object with 'medications' key
    2. Dense pipe-delimited format (name|dosage|form|...) auto-converted to 9-key schema
    3. rx_norm and raw list formats
    """
    clean = strip_fences(raw_text)
    parsed = None
    if "{" in clean or "[" in clean:
        try:
            parsed = json.loads(clean)
        except Exception:
            parsed = None

    if isinstance(parsed, dict) and "medications" in parsed:
        medications = parsed["medications"]
    elif isinstance(parsed, dict) and "rx_norm" in parsed:
        medications = []
        for item in parsed.get("rx_norm", []):
            if not isinstance(item, dict):
                continue
            medications.append({
                "spoken_name": item.get("drug") or item.get("name") or "",
                "strength": str(item.get("strength")) if item.get("strength") is not None else "",
                "dosage_form": str(item.get("form") or item.get("dosage_form") or ""),
                "dose": str(item.get("dose") or ""),
                "route": str(item.get("route") or ""),
                "frequency": str(item.get("freq_en") or item.get("frequency") or ""),
                "duration": str(item.get("duration") or ""),
                "instructions": str(item.get("instructions") or item.get("qty_en") or ""),
                "source_text": str(item.get("source_text") or ""),
            })
    elif isinstance(parsed, list):
        medications = parsed
    else:
        # Check if output is dense pipe-delimited format
        if "|" in raw_text:
            medications = parse_dense_to_prescriptions(raw_text, transcript=transcript)
            if medications:
                return {
                    "valid_json": True,
                    "medications": medications,
                    "medications_count": len(medications),
                    "raw_text": raw_text,
                }
        medications = []

    return {
        "valid_json": parsed is not None,
        "medications": medications if isinstance(medications, list) else [],
        "medications_count": len(medications) if isinstance(medications, list) else 0,
        "raw_text": raw_text,
    }
