"""Clinical prescription schemas, prompts, and normalization utilities."""

from __future__ import annotations

import json
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
# Standard Production Clinical Extraction Prompt (Full JSON Schema)
# ---------------------------------------------------------------------------
STANDARD_SYSTEM_PROMPT = (
    "Extract literal medication mentions from the doctor transcript. "
    'Return valid JSON only in this format: {"medications":[{"spoken_name":"","strength":"","dosage_form":"","dose":"","route":"","frequency":"","duration":"","instructions":"","source_text":""}]}. '
    "Do not prescribe or correct medicine spellings."
)

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


def parse_prescriptions_json(raw_text: str) -> Dict[str, Any]:
    """Parse raw LLM output text into structured clinical medications dictionary."""
    clean = strip_fences(raw_text)
    try:
        parsed = json.loads(clean)
    except Exception:
        # Fallback to loose JSON regex or empty dict
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
        medications = []

    return {
        "valid_json": parsed is not None,
        "medications": medications if isinstance(medications, list) else [],
        "medications_count": len(medications) if isinstance(medications, list) else 0,
        "raw_text": raw_text,
    }
