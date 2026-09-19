#!/usr/bin/env python3
"""
Comprehensive vLLM Benchmark on NVIDIA L4 GPU (24GB VRAM)
Tests all MedGemma 4B quantized models:
  1. FP8 W8A8 Dynamic:     Demondiablo/medgemma-4b-it-fp8-w8a8
  2. INT8 W8A8 Dynamic:    Demondiablo/medgemma-4b-it-int8-w8a8
  3. INT4 RTN W4A16:       Demondiablo/medgemma-4b-it-int4-rtn-w4a16
  4. MXFP8 Microscaling:   Demondiablo/medgemma-4b-it-mxfp8 (capability test)

Outputs:
  - Detailed performance metrics (load time, generation latency, tokens/s, peak VRAM)
  - Raw generation output for every clip
  - Parsed clinical medication extractions (JSON)
  - Schema validity and entity counts
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MODELS = [
    {
        "id": "Demondiablo/medgemma-4b-it-int4-rtn-w4a16",
        "name": "INT4 RTN W4A16",
        "format": "compressed-tensors (W4A16 / Marlin)",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-fp8-w8a8",
        "name": "FP8 W8A8 Dynamic",
        "format": "compressed-tensors (FP8_DYNAMIC)",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-int8-w8a8",
        "name": "INT8 W8A8 Dynamic",
        "format": "compressed-tensors (W8A8)",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-mxfp8",
        "name": "MXFP8 Microscaling",
        "format": "compressed-tensors (MXFP8 / OCP)",
    },
]

OUTPUT_DIR = Path("/home/ubuntu/l4_benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKER_SCRIPT = Path("/home/ubuntu/vllm_single_model_worker.py")

WORKER_CODE = r'''#!/usr/bin/env python3
import gc
import json
import os
import re
import sys
import time
import torch
from vllm import LLM, SamplingParams

model_id = sys.argv[1]
model_name = sys.argv[2]
output_file = sys.argv[3]

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

SYSTEM = """You are a medical transcription extraction engine. Extract every medication instruction explicitly present in the transcript. Do not invent missing information. Preserve medication names as spoken, including brand names. Return VALID JSON ONLY, with no markdown fences and no explanatory text. The top-level schema is: {"rx_norm":[{"drug":string,"strength":string|null,"freq_en":string|null,"qty_en":string|null,"route":string|null,"needs_review":boolean}],"needs_review":boolean}. Use null when a field is not stated or cannot be read confidently. Use needs_review=true when the transcript is unclear, ambiguous, malformed, or missing a clinically important field. Do not omit a medication merely because some fields are missing."""

INLINE = [
    ("audio 2 47s", "Capsule Desula 21 HS Tab 81 2 mg 1 HS Tab Telista CL 1 BD, Thyronorm 50 mg OD, Istamelt 5500 BD. Prolomet XR 50 OD, CTD 12.5 one OD, Ecosprin AV 75/10 one HS Capsule Calbona 3D 15 दिन में 1 बार. Tablet Zifi 200 BD. Tablet Pexican 214"),
    ("Audio 3", "Tablet Nexito Forte 1 HS. Capsule Rabica Gold 1 OD. Telista CL 1 BD. Thyronorm 50 mg 1 OD. Tab Difi M 10/500 1 OD. Sanxit 0.5/2.5 OD. Istamelt 5/500 one HS. Tolomet 50 one HS. CTD 12.5 OD. Neurotto 3D one OD. Sempraz D one OD. Calbone T 1 OD. Capol ER 50 BD. Ecosprin AV 50/10 HS."),
    ("Audio 4", "Dazula 30 ek goli raat mein. Prothen 75 ek goli raat mein. Quetain 25 ek goli raat mein. Tablet Lopez 2 mg ek goli raat mein aur Ativan 2 mg ek goli raat mein. Thyronorm 75 ek goli din mein subah-subah khaali pet. Tablet Prolomet XL 100 mg ek goli subah. Capsule Ecosprin AV 75/10 ek goli raat mein. Calbone 3D din 15 din mein ek baar. Telista CL Trio ek goli din mein. Cilacar 20 ek goli raat mein. Tablet Sanpraz IT ek raat mein. [Unintelligible]"),
    ("Audio 6 1min13secx", "Diabetes Type 2 plus hypothyroidism, obese mild. CBC, hemoglobin, HbA1c, lipid profile, LFT, KFT, thyroid. Tablet Thyronorm 112.5 ek goli subah. Sempas IT ek goli. Ozempic 1 mg. Lantus 26. Glycomet GP2 BD. Glyco Bay 50 TDS. Justoza 10. Capsule Unistar A 50. [Unclear] 75. Carnous 40 BD. Calbonat E3 15 din mein ek baar. Tadalafil 10 mg one HS."),
    ("audio6.ogg", "Tablet Pan 40 OD with empty stomach ES. Tablet Drotin MF BD 5 days with breakfast. Tablet Buscopan 10 mg BD 5 days with breakfast. Tablet Glycomet 500 mg OD thirty minute before breakfast tablet Glycomet GP1 OD 20 mg at 30 minutes before breakfast tablet glycomet GP2 BD 30 minutes before meal tablet amlovas M 5 mg OD before breakfast twenty minutes tablet Tazloc forty OD thirty minutes before breakfast tablet Tazloc CT 40/12.5 mg BD 30 minutes before breakfast. Tablet Triolmesar 40 oblique 5 oblique 12.5 mg OD 30 minutes before breakfast. Tablet Allegra 120 mg OD with breakfast. Tablet Montina LC 10/5 OD HS after meal Tablet Ecosprin 75 mg OD with breakfast. Tablet Clopitab 75 75 mg OD with breakfast. Tablet Acyclosera BD with breakfast. Tablet Clonafit 0.25 mg OD HS. Tablet RHEFD OD with empty stomach. Tablet Ondem 8 mg BD five days. Tablet Emeset 4 mg BD five days. Tablet Doxypheylin 400 mg BD tablet Deriphyllin 150 mg BD tablet Mucinex 600 mg OD in half glass water inhaler Foracort two hundred two hundred mg two puffs twelve hourly inhaler Tiova two puffs twelve hourly inhaler Duolin two puffs twelve hourly Tablet Atorva 40 mg with breakfast. Tablet Atormac TG OD with breakfast. Tablet Sitaglo 100 mg OD. one hour before breakfast or meal tablet Sitaglo GM oblique GM2 OD thirty minutes before Breakfast or meal tablet Dapanorm 10 mg before 30 minutes before meal or breakfast tablet GP 200 BD seven days tablet Clavum 625 mg OD seven days tablet Mero 30 mg BD seven days"),
]

def strip_fences(text):
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
    "supported_on_l4": False,
    "error": None,
    "load_time_s": None,
    "peak_vram_gib": None,
    "clips": [],
    "summary": {},
}

try:
    print(f"=== Initializing vLLM for {model_name} ({model_id}) ===")
    t_load0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
    )
    load_time = round(time.perf_counter() - t_load0, 2)
    result["supported_on_l4"] = True
    result["load_time_s"] = load_time
    print(f"Model loaded in {load_time}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    total_tokens = 0
    total_time = 0.0

    for idx, (clip_id, transcript) in enumerate(INLINE, 1):
        prompt_content = f"Extract medications from this transcript according to the exact schema in the system instruction. Transcript:\\n\\n{transcript}"
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt_content},
        ]
        
        t0 = time.perf_counter()
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        gen_s = round(time.perf_counter() - t0, 3)

        out_obj = outputs[0].outputs[0]
        raw_text = out_obj.text
        tokens = len(out_obj.token_ids)
        tok_s = round(tokens / max(gen_s, 1e-6), 2)
        total_tokens += tokens
        total_time += gen_s

        try:
            parsed = json.loads(strip_fences(raw_text))
            valid = True
        except Exception:
            parsed = None
            valid = False

        extracted_items = map_items(parsed) if valid else []

        print(f"[{idx}/5] {clip_id[:20]:20s} | tokens: {tokens:4d} | time: {gen_s:5.2f}s | tok/s: {tok_s:6.1f} | items: {len(extracted_items):2d} | valid: {str(valid):5s}")

        result["clips"].append({
            "clip_id": clip_id,
            "transcript": transcript,
            "latency_s": gen_s,
            "tokens": tokens,
            "tokens_per_s": tok_s,
            "json_valid": valid,
            "n_extracted_items": len(extracted_items),
            "raw_output": raw_text,
            "extracted_items": extracted_items,
        })

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
    result["peak_vram_gib"] = peak_vram
    result["summary"] = {
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "avg_tok_s": round(total_tokens / max(total_time, 1e-6), 2),
        "total_items_extracted": sum(c["n_extracted_items"] for c in result["clips"]),
        "valid_json_count": sum(1 for c in result["clips"] if c["json_valid"]),
    }

except Exception as exc:
    print(f"Error evaluating {model_name}: {exc}")
    result["error"] = str(exc)
    result["supported_on_l4"] = False

finally:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_file}")
'''

def main():
    print("Writing worker script on L4 VM...")
    WORKER_SCRIPT.write_text(WORKER_CODE)

    all_results = []

    for m in MODELS:
        print("\n" + "=" * 80)
        print(f"  BENCHMARKING MODEL: {m['name']} ({m['id']})")
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

    # Generate unified summary report
    summary_file = OUTPUT_DIR / "l4_vllm_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  ALL BENCHMARKS COMPLETED ON NVIDIA L4 GPU")
    print("=" * 80 + "\n")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
