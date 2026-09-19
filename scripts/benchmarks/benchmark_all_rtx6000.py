#!/usr/bin/env python3
import os
import sys
import json
import csv
import time
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/home/root/rtx6000_benchmark_results" if Path("/home/root").exists() else "/root/rtx6000_benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

(OUTPUT_DIR / "per_audio").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "per_model").mkdir(parents=True, exist_ok=True)

MODELS = [
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
    {
        "id": "Demondiablo/medgemma-4b-it-int8-w8a8",
        "name": "INT8 W8A8 Dynamic",
        "slug": "int8_w8a8_dynamic",
    },
    {
        "id": "Demondiablo/medgemma-4b-it-mxfp8",
        "name": "MXFP8 Microscaling",
        "slug": "mxfp8_microscaling",
    },
    {
        "id": "google/medgemma-4b-it",
        "name": "BF16 Base (Unquantized)",
        "slug": "bf16_base",
    },
]

WORKER_SCRIPT = Path("/tmp/rtx6000_vllm_worker.py")

WORKER_CODE = '''#!/usr/bin/env python3
import os
import sys

# Critical environment settings for Blackwell SM 12.0
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

CLIPS = [
    (
        "audio6.ogg",
        "audio6_ogg",
        "Tablet Pan 40 OD with empty stomach ES. Tablet Drotin MF BD 5 days with breakfast. Tablet Buscopan 10 mg BD 5 days with breakfast. Tablet Glycomet 500 mg OD thirty minute before breakfast tablet Glycomet GP1 OD 20 mg at 30 minutes before breakfast tablet glycomet GP2 BD 30 minutes before meal tablet amlovas M 5 mg OD before breakfast twenty minutes tablet Tazloc forty OD thirty minutes before breakfast tablet Tazloc CT 40/12.5 mg BD 30 minutes before breakfast. Tablet Triolmesar 40 oblique 5 oblique 12.5 mg OD 30 minutes before breakfast. Tablet Allegra 120 mg OD with breakfast. Tablet Montina LC 10/5 OD HS after meal Tablet Ecosprin 75 mg OD with breakfast. Tablet Clopitab 75 75 mg OD with breakfast. Tablet Acyclosera BD with breakfast. Tablet Clonafit 0.25 mg OD HS. Tablet RHEFD OD with empty stomach. Tablet Ondem 8 mg BD five days. Tablet Emeset 4 mg BD five days. Tablet Doxypheylin 400 mg BD tablet Deriphyllin 150 mg BD tablet Mucinex 600 mg OD in half glass water inhaler Foracort two hundred two hundred mg two puffs twelve hourly inhaler Tiova two puffs twelve hourly inhaler Duolin two puffs twelve hourly Tablet Atorva 40 mg with breakfast. Tablet Atormac TG OD with breakfast. Tablet Sitaglo 100 mg OD. one hour before breakfast or meal tablet Sitaglo GM oblique GM2 OD thirty minutes before Breakfast or meal tablet Dapanorm 10 mg before 30 minutes before meal or breakfast tablet GP 200 BD seven days tablet Clavum 625 mg OD seven days tablet Mero 30 mg BD seven days"
    ),
    (
        "audio 2 47s",
        "audio_2_47s",
        "Capsule Desula 21 HS Tab 81 2 mg 1 HS Tab Telista CL 1 BD, Thyronorm 50 mg OD, Istamelt 5500 BD. Prolomet XR 50 OD, CTD 12.5 one OD, Ecosprin AV 75/10 one HS Capsule Calbona 3D 15 दिन में 1 बार. Tablet Zifi 200 BD. Tablet Pexican 214"
    ),
    (
        "Audio 3",
        "audio_3",
        "Tablet Nexito Forte 1 HS. Capsule Rabica Gold 1 OD. Telista CL 1 BD. Thyronorm 50 mg 1 OD. Tab Difi M 10/500 1 OD. Sanxit 0.5/2.5 OD. Istamelt 5/500 one HS. Tolomet 50 one HS. CTD 12.5 OD. Neurotto 3D one OD. Sempraz D one OD. Calbone T 1 OD. Capol ER 50 BD. Ecosprin AV 50/10 HS."
    ),
    (
        "Audio 4",
        "audio_4",
        "Dazula 30 ek goli raat mein. Prothen 75 ek goli raat mein. Quetain 25 ek goli raat mein. Tablet Lopez 2 mg ek goli raat mein aur Ativan 2 mg ek goli raat mein. Thyronorm 75 ek goli din mein subah-subah khaali pet. Tablet Prolomet XL 100 mg ek goli subah. Capsule Ecosprin AV 75/10 ek goli raat mein. Calbone 3D din 15 din mein ek baar. Telista CL Trio ek goli din mein. Cilacar 20 ek goli raat mein. Tablet Sanpraz IT ek raat mein. [Unintelligible]"
    ),
    (
        "Audio 6 1min13secx",
        "audio_6_1min13s",
        "Diabetes Type 2 plus hypothyroidism, obese mild. CBC, hemoglobin, HbA1c, lipid profile, LFT, KFT, thyroid. Tablet Thyronorm 112.5 ek goli subah. Sempas IT ek goli. Ozempic 1 mg. Lantus 26. Glycomet GP2 BD. Glyco Bay 50 TDS. Justoza 10. Capsule Unistar A 50. [Unclear] 75. Carnous 40 BD. Calbonat E3 15 din mein ek baar. Tadalafil 10 mg one HS."
    ),
    (
        "audio 5 18s",
        "audio_5_18s",
        "Most of the slip I drop is drop image saber, spread foot, I drop it, drop image saber, I drop, I drop, I drop in machine bar, from machine bar, I drop it, I drop it."
    ),
]

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

model_res = {
    "model_id": model_id,
    "model_name": model_name,
    "model_slug": model_slug,
    "supported_on_rtx6000": False,
    "error": None,
    "load_time_s": None,
    "peak_vram_gib": None,
    "clips": [],
    "summary": {},
}

model_out_dir = output_dir / "per_model" / model_slug
model_out_dir.mkdir(parents=True, exist_ok=True)

try:
    print(f"=== Initializing vLLM for {model_name} ({model_id}) ===")
    t_load0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    load_time = round(time.perf_counter() - t_load0, 2)
    model_res["supported_on_rtx6000"] = True
    model_res["load_time_s"] = load_time
    print(f"Model loaded in {load_time}s")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2500,
    )

    total_tokens = 0
    total_time = 0.0

    for idx, (clip_title, clip_slug, transcript) in enumerate(CLIPS, 1):
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
            clean = strip_fences(raw_text)
            parsed = json.loads(clean)
            valid = True
        except Exception:
            parsed = None
            valid = False

        extracted_items = map_items(parsed) if valid else []

        print(f"[{idx}/6] {clip_title[:20]:20s} | tokens: {tokens:4d} | time: {gen_s:5.2f}s | tok/s: {tok_s:6.1f} | items: {len(extracted_items):2d} | valid: {str(valid):5s}")

        # Save model-specific raw and extracted files
        raw_file = model_out_dir / f"{clip_slug}_raw.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        ext_file = model_out_dir / f"{clip_slug}_extracted.json"
        with open(ext_file, "w", encoding="utf-8") as f:
            json.dump({
                "clip_title": clip_title,
                "clip_slug": clip_slug,
                "model_name": model_name,
                "model_id": model_id,
                "valid_json": valid,
                "items_count": len(extracted_items),
                "latency_seconds": gen_s,
                "tokens_generated": tokens,
                "throughput_tok_s": tok_s,
                "medications": extracted_items,
            }, f, indent=2, ensure_ascii=False)

        model_res["clips"].append({
            "clip_title": clip_title,
            "clip_slug": clip_slug,
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
    model_res["peak_vram_gib"] = peak_vram
    model_res["summary"] = {
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "avg_tok_s": round(total_tokens / max(total_time, 1e-6), 2),
        "total_items_extracted": sum(c["n_extracted_items"] for c in model_res["clips"]),
        "valid_json_count": sum(1 for c in model_res["clips"] if c["json_valid"]),
    }

except Exception as exc:
    print(f"Error evaluating {model_name}: {exc}")
    model_res["error"] = str(exc)
    model_res["supported_on_rtx6000"] = False

finally:
    res_file = output_dir / f"{model_slug}.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(model_res, f, indent=2, ensure_ascii=False)
    print(f"Model results written to {res_file}")
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
            "/root/miniconda3/envs/py3.10/bin/python3" if Path("/root/miniconda3/envs/py3.10/bin/python3").exists() else sys.executable,
            str(WORKER_SCRIPT),
            m["id"],
            m["name"],
            m["slug"],
            str(OUTPUT_DIR),
        ]

        res = subprocess.run(cmd)
        print(f"Worker for {m['name']} exited with code: {res.returncode}")

        out_file = OUTPUT_DIR / f"{m['slug']}.json"
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_results.append(data)
        else:
            all_results.append({
                "model_id": m["id"],
                "model_name": m["name"],
                "model_slug": m["slug"],
                "supported_on_rtx6000": False,
                "error": f"Process exited with code {res.returncode}",
            })

    # Generate 2 files for each audio across all models
    per_audio_dir = OUTPUT_DIR / "per_audio"
    
    # Identify unique clips from first successful result
    sample_clips = []
    for r in all_results:
        if r.get("clips"):
            sample_clips = [(c["clip_title"], c["clip_slug"], c["transcript"]) for c in r["clips"]]
            break

    for clip_title, clip_slug, transcript in sample_clips:
        # File 1: Pure raw output file
        raw_filename = per_audio_dir / f"{clip_slug}_raw.txt"
        with open(raw_filename, "w", encoding="utf-8") as f:
            f.write(f"=== TRANSCRIPT: {clip_title} ===\n{transcript}\n\n" + "="*80 + "\n\n")
            for r in all_results:
                m_name = r.get("model_name", "Unknown")
                m_clip = next((c for c in r.get("clips", []) if c["clip_slug"] == clip_slug), None)
                f.write(f"--- MODEL: {m_name} ---\n")
                if m_clip:
                    f.write(m_clip.get("raw_output", "") + "\n\n")
                else:
                    f.write(f"Error or not run: {r.get('error')}\n\n")

        # File 2: Correct extraction JSON file
        ext_filename = per_audio_dir / f"{clip_slug}_extracted.json"
        clip_extractions = {
            "clip_title": clip_title,
            "clip_slug": clip_slug,
            "transcript": transcript,
            "models": {},
        }
        for r in all_results:
            m_slug = r.get("model_slug", "unknown")
            m_name = r.get("model_name", "Unknown")
            m_clip = next((c for c in r.get("clips", []) if c["clip_slug"] == clip_slug), None)
            if m_clip:
                clip_extractions["models"][m_slug] = {
                    "model_name": m_name,
                    "valid_json": m_clip["json_valid"],
                    "latency_s": m_clip["latency_s"],
                    "tokens": m_clip["tokens"],
                    "tokens_per_s": m_clip["tokens_per_s"],
                    "items_count": m_clip["n_extracted_items"],
                    "medications": m_clip["extracted_items"],
                }
            else:
                clip_extractions["models"][m_slug] = {"error": r.get("error")}

        with open(ext_filename, "w", encoding="utf-8") as f:
            json.dump(clip_extractions, f, indent=2, ensure_ascii=False)

    # Save consolidated summary JSON
    summary_file = OUTPUT_DIR / "rtx6000_vllm_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save consolidated metrics CSV
    csv_file = OUTPUT_DIR / "rtx6000_vllm_benchmark_metrics.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Model ID", "Clip Title", "Tokens", "Latency (s)", "Throughput (tok/s)", "Extracted Items", "Valid JSON"])
        for r in all_results:
            m_name = r.get("model_name", "Unknown")
            m_id = r.get("model_id", "Unknown")
            for c in r.get("clips", []):
                writer.writerow([
                    m_name,
                    m_id,
                    c["clip_title"],
                    c["tokens"],
                    c["latency_s"],
                    c["tokens_per_s"],
                    c["n_extracted_items"],
                    c["json_valid"],
                ])

    print("\n" + "=" * 80)
    print("  ALL BENCHMARKS COMPLETED ON NVIDIA RTX PRO 6000 GPU")
    print("=" * 80 + "\n")
    print(f"Summary saved to: {summary_file}")
    print(f"Metrics CSV saved to: {csv_file}")
    print(f"Per-audio outputs saved to: {per_audio_dir}")

if __name__ == "__main__":
    main()
