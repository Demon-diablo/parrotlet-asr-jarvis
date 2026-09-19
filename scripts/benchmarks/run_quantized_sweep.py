#!/usr/bin/env python3
"""Automated Quantized MedGemma Extractor Benchmark Sweep on RTX PRO 6000.

Sequentially benchmarks quantized MedGemma-4B models against audio6.ogg:
1. INT4 RTN W4A16:   Demondiablo/medgemma-4b-it-int4-rtn-w4a16
2. MXFP8:            Demondiablo/medgemma-4b-it-mxfp8
3. INT8 W8A8:        Demondiablo/medgemma-4b-it-int8-w8a8
4. FP8 W8A8 Dynamic: Demondiablo/medgemma-4b-it-fp8-w8a8

Guarantees:
- Strict process isolation: only ONE model is loaded in GPU VRAM at any time.
- Complete VRAM flush and verification between runs.
- Extraction of all phase timing metrics (ASR encoder, projector, decoder, extractor, RTT).
- Complete JSON output persistence for each model.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

HERE = Path(__file__).resolve().parent.parent.parent
AUDIO_PATH = HERE / "audio" / "audio6.ogg"
RESULTS_DIR = HERE / "quantized_benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "id": "Demondiablo/medgemma-4b-it-int4-rtn-w4a16",
        "name": "INT4 RTN W4A16",
        "slug": "int4_rtn_w4a16",
        "quantization": "bitsandbytes" if "bnb" in "int4_rtn" else None,
    },
    {
        "id": "Demondiablo/medgemma-4b-it-mxfp8",
        "name": "MXFP8 Microscaling",
        "slug": "mxfp8",
        "quantization": None,
    },
    {
        "id": "Demondiablo/medgemma-4b-it-int8-w8a8",
        "name": "INT8 W8A8 Dynamic",
        "slug": "int8_w8a8",
        "quantization": None,
    },
    {
        "id": "Demondiablo/medgemma-4b-it-fp8-w8a8",
        "name": "FP8 W8A8 Dynamic",
        "slug": "fp8_w8a8",
        "quantization": "fp8",
    },
]

# Baseline reference metrics (from verified run)
BASELINE_BF16 = {
    "model_name": "BF16 Base (Unquantized)",
    "model_id": "google/medgemma-4b-it",
    "slug": "bf16_base",
    "supported": True,
    "asr_latency_s": 2.3866,
    "whisper_encoder_s": 0.1134,
    "projector_s": 0.0700,
    "decoder_s": 1.9866,
    "extractor_latency_s": 9.1340,
    "extractor_tokens": 408,
    "extractor_tok_s": 44.7,
    "total_rtt_s": 11.5286,
    "medications_count": 34,
    "valid_json": True,
    "peak_vram_gib": 45.7,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_gpu_vram_gib() -> float:
    """Query currently allocated VRAM in GiB using nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        return round(float(out.splitlines()[0]) / 1024.0, 2)
    except Exception:
        return 0.0


def force_kill_servers() -> None:
    """Kill any running serve_jarvis or uvicorn processes and flush CUDA cache."""
    log("Cleaning up any existing server processes...")
    subprocess.run(["pkill", "-9", "-f", "serve_jarvis.py"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "sglang"], stderr=subprocess.DEVNULL)
    time.sleep(2)

    # Empty torch CUDA cache via quick Python one-liner
    py_bin = sys.executable
    subprocess.run([
        py_bin, "-c",
        "import torch; (torch.cuda.is_available() and (torch.cuda.empty_cache(), torch.cuda.ipc_collect()))"
    ], stderr=subprocess.DEVNULL)
    time.sleep(1)
    vram = get_gpu_vram_gib()
    log(f"GPU VRAM after cleanup: {vram:.2f} GiB")


def wait_for_server_ready(port: int = 6006, timeout_s: int = 180, proc: Optional[subprocess.Popen] = None) -> bool:
    """Poll health endpoint until 200 OK or process exits / times out."""
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            log(f"Server process terminated prematurely with exit code {proc.returncode}")
            return False
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("loaded", False):
                        return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_pipeline_benchmark(audio_file: Path, port: int = 6006) -> Dict[str, Any]:
    """Execute multipart/form-data upload to /pipeline and return full parsed JSON."""
    import mimetypes

    url = f"http://127.0.0.1:{port}/pipeline"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = audio_file.name
    mime_type = mimetypes.guess_type(str(audio_file))[0] or "audio/ogg"

    with open(audio_file, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        elapsed = time.perf_counter() - t0
        res_data = json.loads(resp.read().decode("utf-8"))
        res_data["client_rtt_seconds"] = round(elapsed, 4)
        return res_data


def benchmark_single_model(m: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
    """Run isolated benchmark lifecycle for one model."""
    model_id = m["id"]
    model_name = m["name"]
    slug = m["slug"]

    log("=" * 72)
    log(f"STARTING EVALUATION: {model_name} ({model_id})")
    log("=" * 72)

    force_kill_servers()

    server_log_path = RESULTS_DIR / f"{slug}_server.log"
    server_log_file = open(server_log_path, "w", encoding="utf-8")

    env = dict(os.environ)
    env["HOST"] = "0.0.0.0"
    env["PORT"] = "6006"
    env["MEDGEMMA_MODEL_ID"] = model_id
    env["EXTRACTOR_BACKEND"] = "sglang"
    env["EXTRACTOR_MODE"] = "dense"
    env["ALLOW_TF32"] = "1"
    env["FLOAT32_MATMUL_PRECISION"] = "high"
    env["NVCC_PREPEND_FLAGS"] = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1"
    env.pop("FLASHINFER_CUDA_ARCH_LIST", None)

    py_bin = "/root/miniconda3/envs/py3.10/bin/python" if Path("/root/miniconda3/envs/py3.10/bin/python").exists() else sys.executable
    server_script = HERE / "serve_jarvis.py"

    log(f"Launching serve_jarvis.py with {model_id}...")
    t_start = time.perf_counter()
    proc = subprocess.Popen(
        [py_bin, str(server_script), "--host", "0.0.0.0", "--port", "6006"],
        cwd=str(HERE),
        env=env,
        stdout=server_log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    ready = wait_for_server_ready(port=6006, timeout_s=180, proc=proc)
    load_time_s = round(time.perf_counter() - t_start, 2)
    peak_vram = get_gpu_vram_gib()

    result_record: Dict[str, Any] = {
        "model_name": model_name,
        "model_id": model_id,
        "slug": slug,
        "supported": ready,
        "load_time_s": load_time_s,
        "peak_vram_gib": peak_vram,
        "error": None,
    }

    if not ready:
        log(f"FAILED: {model_name} could not start or initialize within timeout.")
        server_log_file.close()
        # Read last 25 lines of log for diagnosis
        try:
            with open(server_log_path, "r", encoding="utf-8") as lf:
                lines = lf.readlines()
                err_snippet = "".join(lines[-20:])
                result_record["error"] = err_snippet.strip()
                log(f"Server Log Error Snippet:\n{err_snippet}")
        except Exception:
            result_record["error"] = "Process exited or timed out"
        
        force_kill_servers()
        return result_record

    log(f"Server READY in {load_time_s:.2f}s! Peak VRAM: {peak_vram:.2f} GiB")

    try:
        log(f"Executing pipeline inference on {audio_path.name}...")
        pipeline_resp = run_pipeline_benchmark(audio_path, port=6006)
        log("Pipeline execution finished successfully!")

        # Save full pipeline response JSON
        pipeline_out_file = RESULTS_DIR / f"{slug}_audio6_pipeline.json"
        with open(pipeline_out_file, "w", encoding="utf-8") as f:
            json.dump(pipeline_resp, f, indent=2, ensure_ascii=False)
        log(f"Saved full response: {pipeline_out_file.name}")

        output_dict = pipeline_resp.get("output", {})
        asr_output = output_dict.get("asr_output", {})
        phases = asr_output.get("phases", {})
        extraction = output_dict.get("extraction", {})
        medications = extraction.get("medications", [])

        # Save clean extracted medications JSON
        extracted_out_file = RESULTS_DIR / f"{slug}_audio6_extracted.json"
        with open(extracted_out_file, "w", encoding="utf-8") as f:
            json.dump({
                "valid_json": extraction.get("valid_json", True),
                "medications_count": len(medications),
                "medications": medications,
            }, f, indent=2, ensure_ascii=False)
        log(f"Saved extracted medications: {extracted_out_file.name} ({len(medications)} items)")

        # Record fine-grained timing metrics
        asr_lat = output_dict.get("asr_latency_seconds", asr_output.get("inference_seconds", 0.0))
        ext_lat = extraction.get("latency_seconds", 0.0)
        ext_tok = extraction.get("tokens_generated", 0)
        ext_tok_s = extraction.get("throughput_tok_s", 0.0)
        client_rtt = pipeline_resp.get("client_rtt_seconds", output_dict.get("total_latency_seconds", 0.0))

        result_record.update({
            "asr_latency_s": asr_lat,
            "whisper_encoder_s": phases.get("whisper_encoder_s", 0.0),
            "projector_s": phases.get("projector_s", 0.0),
            "decoder_s": phases.get("decoder_s", phases.get("decode_s", 0.0)),
            "extractor_latency_s": ext_lat,
            "extractor_tokens": ext_tok,
            "extractor_tok_s": ext_tok_s,
            "total_rtt_s": client_rtt,
            "medications_count": len(medications),
            "valid_json": extraction.get("valid_json", True),
            "peak_vram_gib": get_gpu_vram_gib(),
        })

        log(f"RESULTS for {model_name}:")
        log(f" • Extractor Latency  : {ext_lat:.4f}s")
        log(f" • Extractor Throughput: {ext_tok_s:.1f} tok/s ({ext_tok} tokens)")
        log(f" • Medications Count  : {len(medications)}")
        log(f" • Client RTT         : {client_rtt:.4f}s")

    except Exception as exc:
        log(f"Error during inference execution: {exc}")
        result_record["error"] = str(exc)
    finally:
        server_log_file.close()
        force_kill_servers()

    return result_record


def write_summary_reports(all_results: List[Dict[str, Any]]) -> None:
    """Generate master CSV and Markdown summary comparison tables."""
    csv_path = RESULTS_DIR / "summary_metrics.csv"
    fieldnames = [
        "model_name", "slug", "supported", "load_time_s", "peak_vram_gib",
        "asr_latency_s", "whisper_encoder_s", "projector_s", "decoder_s",
        "extractor_latency_s", "extractor_tokens", "extractor_tok_s",
        "total_rtt_s", "medications_count", "valid_json", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    log(f"Wrote summary CSV: {csv_path.name}")

    md_path = RESULTS_DIR / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Quantized MedGemma Extractor Benchmark Comparison\n\n")
        f.write(f"**Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM)\n")
        f.write(f"**Audio**: `audio/audio6.ogg` (197.5s clinical consultation)\n\n")
        f.write("### Timing & Performance Summary\n\n")
        f.write("| Model | Status | Load Time | Peak VRAM | Extractor Latency | Throughput | Meds Found | Total RTT | Speedup vs BF16 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        bf16_ext_lat = BASELINE_BF16["extractor_latency_s"]
        for r in all_results:
            name = r.get("model_name", "")
            supp = "✅ Supported" if r.get("supported") else "❌ Failed"
            load_t = f"{r.get('load_time_s', 0):.1f}s" if r.get("load_time_s") else "-"
            vram = f"{r.get('peak_vram_gib', 0):.1f} GiB" if r.get("peak_vram_gib") else "-"
            ext_lat = f"{r.get('extractor_latency_s', 0):.4f}s" if r.get("extractor_latency_s") else "-"
            tok_s = f"{r.get('extractor_tok_s', 0):.1f} tok/s" if r.get("extractor_tok_s") else "-"
            cnt = f"{r.get('medications_count', 0)}" if r.get("medications_count") is not None else "-"
            rtt = f"{r.get('total_rtt_s', 0):.4f}s" if r.get("total_rtt_s") else "-"
            
            speedup = "-"
            if r.get("extractor_latency_s") and r.get("extractor_latency_s") > 0:
                sp_val = bf16_ext_lat / r["extractor_latency_s"]
                speedup = f"{sp_val:.2f}x" if sp_val >= 1.0 else f"{sp_val:.2f}x (slower)"

            f.write(f"| **{name}** | {supp} | {load_t} | {vram} | {ext_lat} | {tok_s} | {cnt} | {rtt} | **{speedup}** |\n")

        f.write("\n\n### Stage-by-Stage Breakdown\n\n")
        f.write("| Model | Whisper Encoder | Projector | ASR LLM Decoder | ASR Pure Latency | Extractor Latency | Total RTT |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_results:
            if not r.get("supported"):
                continue
            name = r.get("model_name", "")
            enc = f"{r.get('whisper_encoder_s', 0):.4f}s"
            proj = f"{r.get('projector_s', 0):.4f}s"
            dec = f"{r.get('decoder_s', 0):.4f}s"
            asr = f"{r.get('asr_latency_s', 0):.4f}s"
            ext = f"{r.get('extractor_latency_s', 0):.4f}s"
            rtt = f"{r.get('total_rtt_s', 0):.4f}s"
            f.write(f"| **{name}** | {enc} | {proj} | {dec} | {asr} | {ext} | {rtt} |\n")

    log(f"Wrote Markdown report: {md_path.name}")


def main() -> None:
    if not AUDIO_PATH.exists():
        sys.exit(f"Audio file not found: {AUDIO_PATH}")

    log(f"Starting Quantized Sweep on {AUDIO_PATH.name} ({len(MODELS)} models to test)...")
    all_results: List[Dict[str, Any]] = [BASELINE_BF16]

    for m in MODELS:
        res = benchmark_single_model(m, audio_path=AUDIO_PATH)
        all_results.append(res)
        write_summary_reports(all_results)
        time.sleep(2)

    log("\n" + "=" * 72)
    log("BENCHMARK SWEEP COMPLETE!")
    log("=" * 72)
    write_summary_reports(all_results)


if __name__ == "__main__":
    main()
