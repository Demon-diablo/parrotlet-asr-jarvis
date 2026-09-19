#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/l4_benchmark_results/audio6_complete")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "id": "Demondiablo/medgemma-4b-it-int4-rtn-w4a16",
        "name": "INT4 RTN W4A16",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-fp8-w8a8",
        "name": "FP8 W8A8 Dynamic",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-int8-w8a8",
        "name": "INT8 W8A8 Dynamic",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-mxfp8",
        "name": "MXFP8 Microscaling",
    },
]

WORKER_SCRIPT = Path("/home/ubuntu/vllm_audio6_worker.py")
WORKER_CODE = '''#!/usr/bin/env python3
import os
import sys
import json
import time
import re
import torch
from vllm import LLM, SamplingParams

model_id = sys.argv[1]
model_name = sys.argv[2]
output_file = sys.argv[3]

SYSTEM = (
    "You are an expert medical transcriptionist and clinical pharmacologist. "
    "Your task is to extract all medications, strengths, frequencies, routes, quantities, "
    "and instructions mentioned in clinical conversation transcripts. "
    "Output strictly valid JSON matching this schema:\\n"
    "{\\n"
    '  "rx_norm": [\\n'
    "    {\\n"
    '      "drug": "string (brand or generic drug name)",\\n'
    '      "strength": "string or null",\\n'
    '      "freq_en": "string (e.g. OD, BD, TDS, PRN, etc.)",\\n'
    '      "qty_en": "string (quantity, instructions, or null)",\\n'
    '      "route": "string (Oral, Topical, Inhalation, etc.)",\\n'
    '      "needs_review": false\\n'
    "    }\\n"
    "  ]\\n"
    "}\\n"
    "Do not include any conversational filler, markdown explanations, or commentary outside the JSON."
)

TRANSCRIPT = (
    "Tablet Pan 40 OD with empty stomach ES. Tablet Drotin MF BD 5 days with breakfast. "
    "Tablet Buscopan 10 mg BD 5 days with breakfast. Tablet Glycomet 500 mg OD thirty minute before breakfast "
    "tablet Glycomet GP1 OD 20 mg at 30 minutes before breakfast tablet glycomet GP2 BD 30 minutes before meal "
    "tablet amlovas M 5 mg OD before breakfast twenty minutes tablet Tazloc forty OD thirty minutes before breakfast "
    "tablet Tazloc CT 40/12.5 mg BD 30 minutes before breakfast. Tablet Triolmesar 40 oblique 5 oblique 12.5 mg OD 30 minutes before breakfast. "
    "Tablet Allegra 120 mg OD with breakfast. Tablet Montina LC 10/5 OD HS after meal "
    "Tablet Ecosprin 75 mg OD with breakfast. Tablet Clopitab 75 75 mg OD with breakfast. "
    "Tablet Acyclosera BD with breakfast. Tablet Clonafit 0.25 mg OD HS. "
    "Tablet RHEFD OD with empty stomach. Tablet Ondem 8 mg BD five days. "
    "Tablet Emeset 4 mg BD five days. Tablet Doxypheylin 400 mg BD tablet Deriphyllin 150 mg BD "
    "tablet Mucinex 600 mg OD in half glass water inhaler Foracort two hundred two hundred mg two puffs twelve hourly "
    "inhaler Tiova two puffs twelve hourly inhaler Duolin two puffs twelve hourly "
    "Tablet Atorva 40 mg with breakfast. Tablet Atormac TG OD with breakfast. "
    "Tablet Sitaglo 100 mg OD. one hour before breakfast or meal tablet Sitaglo GM oblique GM2 OD thirty minutes before Breakfast or meal "
    "tablet Dapanorm 10 mg before 30 minutes before meal or breakfast tablet GP 200 BD seven days "
    "tablet Clavum 625 mg OD seven days tablet Mero 30 mg BD seven days"
)

def strip_fences(text):
    text = (text or "").strip()
    if "<unused95>" in text:
        text = text.split("<unused95>")[-1].strip()
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)\\s*```", text, re.I)
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
            text = text[first_bracket:last_bracket+1]
    elif first_brace != -1:
        last_brace = text.rfind("}")
        if last_brace > first_brace:
            text = text[first_brace:last_brace+1]
    return text

def map_items(parsed):
    rx = parsed if isinstance(parsed, list) else (parsed.get("rx_norm", []) if isinstance(parsed, dict) else [])
    items = []
    for e in rx:
        if not isinstance(e, dict):
            continue
        strength = e.get("strength")
        if strength is not None and not isinstance(strength, str):
            strength = str(strength)
        strength = strength.strip() if isinstance(strength, str) else None
        items.append({
            "name": e.get("drug") or "",
            "dosage": strength or None,
            "frequency": e.get("freq_en") or e.get("frequency") or e.get("freq"),
            "quantity": e.get("qty_en") or e.get("quantity") or e.get("qty") or e.get("duration"),
            "instruction": e.get("route") or e.get("instruction") or e.get("form"),
            "needs_review": bool(e.get("needs_review", False)),
        })
    return items

result = {
    "model_id": model_id,
    "model_name": model_name,
    "clip_id": "audio6.ogg",
    "transcript": TRANSCRIPT,
    "supported_on_l4": False,
    "error": None,
    "load_time_s": None,
    "tokens": None,
    "latency_s": None,
    "tokens_per_s": None,
    "json_valid": False,
    "n_extracted_items": 0,
    "raw_output": None,
    "extracted_items": [],
    "peak_vram_gib": None,
}

try:
    print(f"=== Initializing vLLM for {model_name} ({model_id}) ===")
    t_load0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    load_time = round(time.perf_counter() - t_load0, 2)
    result["supported_on_l4"] = True
    result["load_time_s"] = load_time
    print(f"Model loaded in {load_time}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2500,
    )

    prompt_content = f"Extract medications from this transcript according to the exact schema in the system instruction. Transcript:\\n\\n{TRANSCRIPT}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt_content},
    ]

    print("Generating complete clinical extractions for audio6.ogg (max_tokens=2500)...")
    t0 = time.perf_counter()
    outputs = llm.chat(messages=messages, sampling_params=sampling_params)
    gen_s = round(time.perf_counter() - t0, 3)

    out_obj = outputs[0].outputs[0]
    raw_text = out_obj.text
    tokens = len(out_obj.token_ids)
    tok_s = round(tokens / max(gen_s, 1e-6), 2)

    try:
        clean = strip_fences(raw_text)
        parsed = json.loads(clean)
        valid = True
    except Exception as e:
        print(f"JSON parsing error: {e}")
        parsed = None
        valid = False

    extracted_items = map_items(parsed) if valid else []

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    result["peak_vram_gib"] = peak_vram
    result["tokens"] = tokens
    result["latency_s"] = gen_s
    result["tokens_per_s"] = tok_s
    result["json_valid"] = valid
    result["n_extracted_items"] = len(extracted_items)
    result["raw_output"] = raw_text
    result["extracted_items"] = extracted_items

    print(f"DONE: tokens={tokens} | latency={gen_s}s | tok/s={tok_s} | items={len(extracted_items)} | valid={valid} | peak_vram={peak_vram}GiB")

except Exception as exc:
    print(f"Error evaluating {model_name}: {exc}")
    result["error"] = str(exc)
    result["supported_on_l4"] = False

finally:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")
'''

def main():
    print("Writing worker script on L4 VM...")
    WORKER_SCRIPT.write_text(WORKER_CODE)

    all_results = []

    for m in MODELS:
        print("\n" + "=" * 80)
        print(f"  BENCHMARKING AUDIO6 ON MODEL: {m['name']} ({m['id']})")
        print("=" * 80 + "\n", flush=True)

        out_file = OUTPUT_DIR / f"{m['name'].lower().replace(' ', '_')}.json"
        cmd = [
            "/home/ubuntu/venv/bin/python3",
            str(WORKER_SCRIPT),
            m["id"],
            m["name"],
            str(out_file),
        ]

        res = subprocess.run(cmd)
        print(f"Worker exited with code: {res.returncode}")

        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append(data)
        else:
            all_results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "supported_on_l4": False,
                "error": f"Process exited with code {res.returncode}",
            })

    # Summary
    summary_file = OUTPUT_DIR / "audio6_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  AUDIO6 COMPLETE BENCHMARK COMPLETED ON NVIDIA L4 GPU")
    print("=" * 80 + "\n")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
