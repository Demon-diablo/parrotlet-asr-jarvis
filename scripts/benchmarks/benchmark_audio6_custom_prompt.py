#!/usr/bin/env python3
import os
import sys
import json
import csv
import time
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/root/audio6_custom_prompt_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "id": "google/medgemma-4b-it",
        "name": "BF16 Base (Unquantized)",
        "slug": "bf16_base",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-int4-rtn-w4a16",
        "name": "INT4 RTN W4A16",
        "slug": "int4_rtn_w4a16",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-fp8-w8a8",
        "name": "FP8 W8A8 Dynamic",
        "slug": "fp8_w8a8_dynamic",
    },
]

WORKER_SCRIPT = Path("/tmp/audio6_custom_worker.py")
WORKER_CODE = '''#!/usr/bin/env python3
import os
import sys

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = "9.0"

import json
import time
import re
import torch
from pathlib import Path
from vllm import LLM, SamplingParams

model_id = sys.argv[1]
model_name = sys.argv[2]
model_slug = sys.argv[3]
output_dir = Path(sys.argv[4])

SYSTEM_CONTENT = (
    "Extract literal medication mentions from the doctor transcript. "
    'Return valid JSON only in this format: {"medications":[{"spoken_name":"","strength":"","dosage_form":"","dose":"","route":"","frequency":"","duration":"","instructions":"","source_text":""}]}. '
    "Do not prescribe or correct medicine spellings."
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

result = {
    "model_id": model_id,
    "model_name": model_name,
    "model_slug": model_slug,
    "prompt_config": {
        "system_prompt": SYSTEM_CONTENT,
        "temperature": 0.0,
        "max_tokens": 8000,
    },
    "load_time_s": None,
    "latency_s": None,
    "tokens": None,
    "tokens_per_s": None,
    "json_valid": False,
    "n_medications": 0,
    "peak_vram_gib": None,
    "raw_output": None,
    "extracted_json": None,
    "error": None,
}

try:
    print(f"=== Initializing vLLM for {model_name} ({model_id}) ===")
    t_load0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    load_time = round(time.perf_counter() - t_load0, 2)
    result["load_time_s"] = load_time
    print(f"Model loaded in {load_time}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8000,
    )

    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": TRANSCRIPT},
    ]

    print("Generating extractions for audio6.ogg with prompt (temperature=0, max_tokens=8000)...")
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

    medications = []
    if isinstance(parsed, dict) and "medications" in parsed:
        medications = parsed["medications"]
    elif isinstance(parsed, list):
        medications = parsed

    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

    result["latency_s"] = gen_s
    result["tokens"] = tokens
    result["tokens_per_s"] = tok_s
    result["json_valid"] = valid
    result["n_medications"] = len(medications)
    result["peak_vram_gib"] = peak_vram
    result["raw_output"] = raw_text
    result["extracted_json"] = parsed

    print(f"RESULTS for {model_name}:")
    print(f"  Tokens: {tokens} | Latency: {gen_s}s | Speed: {tok_s} tok/s | Medications: {len(medications)} | Valid JSON: {valid}")

    # Save model specific raw and extracted files
    raw_path = output_dir / f"{model_slug}_audio6_raw.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    ext_path = output_dir / f"{model_slug}_audio6_extracted.json"
    with open(ext_path, "w", encoding="utf-8") as f:
        json.dump(result["extracted_json"] or {"raw_text": raw_text}, f, indent=2, ensure_ascii=False)

except Exception as exc:
    print(f"Error evaluating {model_name}: {exc}")
    result["error"] = str(exc)

finally:
    res_file = output_dir / f"{model_slug}_full_result.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Full result written to {res_file}")
'''

def main():
    print("Writing worker script on RTX PRO 6000...")
    WORKER_SCRIPT.write_text(WORKER_CODE)

    all_results = []

    for m in MODELS:
        print("\n" + "=" * 80)
        print(f"  BENCHMARKING MODEL: {m['name']} ({m['id']})")
        print("=" * 80 + "\n", flush=True)

        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            m["id"],
            m["name"],
            m["slug"],
            str(OUTPUT_DIR),
        ]

        res = subprocess.run(cmd)
        print(f"Worker for {m['name']} exited with code: {res.returncode}")

        res_file = OUTPUT_DIR / f"{m['slug']}_full_result.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append(data)
        else:
            all_results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "model_slug": m["slug"],
                "error": f"Process exited with code {res.returncode}",
            })

    # Summary JSON
    summary_file = OUTPUT_DIR / "audio6_custom_prompt_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Metrics CSV
    csv_file = OUTPUT_DIR / "audio6_custom_prompt_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Name", "Model ID", "Tokens Generated", "Latency (s)", "Throughput (tok/s)", "Medications Extracted", "Valid JSON", "Peak VRAM (GiB)"])
        for r in all_results:
            writer.writerow([
                r.get("model_name"),
                r.get("model_id"),
                r.get("tokens"),
                r.get("latency_s"),
                r.get("tokens_per_s"),
                r.get("n_medications"),
                r.get("json_valid"),
                r.get("peak_vram_gib"),
            ])

    print("\n" + "=" * 80)
    print("  AUDIO6 BENCHMARK WITH CUSTOM PROMPT COMPLETED ON RTX PRO 6000")
    print("=" * 80 + "\n")
    print(f"Summary JSON: {summary_file}")
    print(f"Metrics CSV:  {csv_file}")

if __name__ == "__main__":
    main()
