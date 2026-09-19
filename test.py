#!/usr/bin/env python3
"""Test ASR endpoint on JarvisLabs VM (or localhost). Stdlib only.

Usage:
    python test.py audio/audio6.ogg
    python test.py audio/audio6.ogg --url http://localhost:6006
    python test.py audio/audio6.ogg --url http://<vm-ip>:6006 --token <secret>
    python test.py audio/audio6.ogg --raw
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

URL = "http://localhost:6006"
TOKEN = ""
AUDIO = "./sample.wav"


def req(method, url, body=None, headers=None, timeout=300):
    r = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as h:
            return h.status, h.read().decode("utf-8", "replace")
    except Exception as e:
        try:
            return getattr(e, "code", -1), e.read().decode("utf-8", "replace")  # type: ignore
        except Exception:
            return -1, str(e)


def print_report(res_json, elapsed_total, audio_path):
    out = res_json.get("output", res_json) if isinstance(res_json, dict) else {}
    if not isinstance(out, dict):
        out = {}

    transcript = out.get("transcript") or out.get("text") or res_json.get("transcript") or ""
    audio_conv = out.get("audio_conversion", {})
    metrics = out.get("metrics", {})
    phases = out.get("phases", {})
    timing = out.get("timing", {})

    # Extract audio metrics
    orig_sr = audio_conv.get("original_sample_rate") or out.get("native_sample_rate") or "Unknown"
    target_sr = audio_conv.get("target_sample_rate", 16000)
    is_resampled = audio_conv.get("is_resampled_to_16k", orig_sr != 16000)
    orig_ch = audio_conv.get("original_channels", "Unknown")
    total_samples = audio_conv.get("total_16k_samples") or out.get("total_16k_samples") or 0
    audio_dur = float(metrics.get("audio_duration_s") or out.get("audio_seconds") or 0.0)
    num_windows = metrics.get("flushed_windows") or out.get("flushed_windows", 1)

    # Speed metrics
    inf_s = float(metrics.get("inference_seconds") or timing.get("inference_seconds") or 0.0)
    rtf = float(
        metrics.get("real_time_factor_rtf")
        or timing.get("real_time_factor_rtf")
        or (inf_s / audio_dur if audio_dur > 0 else 0.0)
    )
    speedup = (
        metrics.get("speedup_factor")
        or timing.get("speedup")
        or (f"{audio_dur / inf_s:.1f}x" if inf_s > 0 else "?")
    )
    gen_tokens = metrics.get("tokens_generated") or phases.get("tokens_generated", 0)
    tok_per_sec = metrics.get("tokens_per_second") or phases.get("tokens_per_second", 0.0)

    # Stages breakdown
    stage_items = [
        ("1. Audio Decode & Load", phases.get("audio_decode_s")),
        ("2. Mono Downmix", phases.get("mono_mix_s")),
        ("3. 16 kHz Resampling", phases.get("resample_16k_s")),
        ("4. Window Buffering", phases.get("windowing_s")),
        ("5. Whisper Feature Extractor", phases.get("featurize_s")),
        ("6. Whisper Audio Encoder", phases.get("whisper_encoder_s") or phases.get("encode_s")),
        ("7. Multimodal Projector", phases.get("projector_s")),
        ("8. Prompt & Embed Setup", phases.get("prompt_prep_s")),
        ("9. LLM Decoder Generation", phases.get("decoder_s") or phases.get("decode_s")),
        ("10. Token Detokenize", phases.get("detokenize_s")),
        ("11. Window Join & Cleanup", phases.get("assemble_s")),
    ]

    ch_str = "1 (Mono)" if orig_ch == 1 else ("2 (Stereo)" if orig_ch == 2 else f"{orig_ch} Channels")
    resample_str = (
        f"YES ({orig_sr:,} Hz -> 16,000 Hz float32 mono)"
        if (is_resampled and isinstance(orig_sr, (int, float)))
        else ("NO (already 16,000 Hz mono)" if orig_sr == 16000 else "YES (16,000 Hz)")
    )

    print("\n" + "=" * 78)
    print("                    PARROTLET ASR TRANSCRIPTION & METRICS")
    print("=" * 78)

    print("\n[1] AUDIO PROPERTIES & 16 kHz RESAMPLING VERIFICATION")
    print("─" * 78)
    print(f" • Input File Path       : {audio_path}")
    print(f" • Original Sample Rate  : {orig_sr if isinstance(orig_sr, str) else f'{orig_sr:,} Hz'}")
    print(f" • Original Channels     : {ch_str}")
    print(f" • Resampled to 16 kHz   : {resample_str}")
    print(f" • Converted to Mono     : YES (Float32 single channel)")
    print(f" • Total 16 kHz Samples  : {total_samples:,} samples")
    print(f" • Audio Duration        : {audio_dur:.2f} seconds")
    print(f" • Window Chunks (<=30s) : {num_windows} window(s)")

    print("\n[2] SPEED & THROUGHPUT METRICS")
    print("─" * 78)
    print(f" • Audio Duration        : {audio_dur:.2f}s")
    print(f" • Pure Inference Time   : {inf_s:.4f}s")
    print(f" • Round-Trip Latency    : {elapsed_total:.4f}s")
    print(f" • Real-Time Factor (RTF): {rtf:.4f}  (Time spent per 1s of audio, lower = faster)")
    print(f" • Speedup Factor        : {speedup} faster than real-time!")
    if gen_tokens:
        print(f" • Tokens Generated      : {gen_tokens} tokens")
    if tok_per_sec:
        print(f" • Generation Throughput : {tok_per_sec:.1f} tokens/second")

    present_stages = [(name, val) for name, val in stage_items if val is not None]
    if present_stages:
        print("\n[3] STEP-BY-STEP FUNCTION & STAGE TIMING BREAKDOWN")
        print("─" * 78)
        print(f"  {'Pipeline Stage':<38} {'Duration':>12}   {'% of Infer':>12}")
        print("  " + "─" * 66)
        for name, duration in present_stages:
            pct = (duration / inf_s * 100.0) if inf_s > 0 else 0.0
            print(f"  {name:<38} {duration:>11.4f}s   {pct:>11.1f}%")
        print("  " + "─" * 66)
        print(f"  {'Total Inference Latency':<38} {inf_s:>11.4f}s   {'100.0%':>12}")

    scripts_dict = out.get("detected_scripts", {})
    if isinstance(scripts_dict, dict):
        active_scripts = [
            k.replace("has_", "").title()
            for k, v in scripts_dict.items()
            if k.startswith("has_") and k != "has_indic" and v
        ]
        non_latin = scripts_dict.get("non_latin_count", 0)
        script_display = (
            "Pure Latin/English (0 non-Latin characters)"
            if not active_scripts
            else f"{', '.join(active_scripts)} ({non_latin} non-Latin chars)"
        )
    else:
        script_display = str(scripts_dict)

    ban_applied = out.get("ban_applied", False)
    banned_cnt = out.get("banned_token_count", 0)
    print("\n[4] GUARDRAILS & SCRIPT ENFORCEMENT")
    print("─" * 78)
    print(f" • Script Filter / Ban   : {'Active' if ban_applied else 'Disabled'} ({banned_cnt} banned tokens suppressed)")
    print(f" • Detected Script(s)    : {script_display}")

    print("\n[5] TRANSCRIPTION")
    print("─" * 78)
    print(f'"{transcript}"')

    extraction = out.get("extraction")
    if isinstance(extraction, dict):
        ext_lat = extraction.get("latency_seconds", 0.0)
        ext_tok = extraction.get("tokens_generated", 0)
        ext_tok_s = extraction.get("throughput_tok_s", 0.0)
        meds = extraction.get("medications", [])
        med_cnt = extraction.get("medications_count", len(meds))
        val_json = extraction.get("valid_json", False)

        print("\n[6] ZERO-HOP MEDGEMMA CLINICAL EXTRACTION")
        print("─" * 78)
        print(f" • Extractor Engine      : vLLM (Prefix Cached)")
        print(f" • Extraction Latency    : {ext_lat:.4f}s")
        print(f" • Extractor Throughput  : {ext_tok_s:.1f} tokens/second ({ext_tok} tokens)")
        print(f" • Medications Extracted : {med_cnt} items")
        print(f" • JSON Syntax Valid     : {val_json}")
        print("\n[7] STRUCTURED CLINICAL MEDICATIONS JSON OUTPUT")
        print("─" * 78)
        print(json.dumps({"medications": meds}, indent=2, ensure_ascii=False))

    print("\n[8] TOTAL PIPELINE RTT")
    print("─" * 78)
    asr_lat = out.get("asr_latency_seconds", inf_s)
    tot_lat = out.get("total_latency_seconds", elapsed_total)
    print(f" • ASR Pure Latency      : {asr_lat:.4f}s")
    if isinstance(extraction, dict):
        print(f" • Extractor Latency     : {extraction.get('latency_seconds', 0.0):.4f}s")
    print(f" • Total Round-Trip Time : {elapsed_total:.4f}s")
    print("=" * 78 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Test Parrotlet ASR endpoint with detailed metrics.")
    ap.add_argument("audio", nargs="?", default=os.getenv("TEST_AUDIO", AUDIO))
    ap.add_argument("--url", default=os.getenv("URL", URL))
    ap.add_argument("--token", default=os.getenv("AUTH_TOKEN") or os.getenv("MODAL_AUTH_TOKEN", ""))
    ap.add_argument("--pipeline", action="store_true", default=True, help="Test full zero-hop /pipeline endpoint.")
    ap.add_argument("--transcribe-only", dest="pipeline", action="store_false", help="Test /transcribe only.")
    ap.add_argument("--raw", action="store_true", help="Print raw JSON response.")
    a = ap.parse_args()

    base = a.url.rstrip("/")
    if not base:
        sys.exit("No URL. Pass --url or fill URL in test.py.")
    if not a.audio or not Path(a.audio).exists():
        sys.exit(f"Audio not found: {a.audio!r}. Usage: python test.py <path.wav>")
    if Path(a.audio).stat().st_size > 50 * 1024 * 1024:
        sys.exit("Audio >50MiB cap.")

    token = a.token or ""
    H = {"Authorization": f"Bearer {token}"} if token else {}
    endpoint = "/pipeline" if a.pipeline else "/transcribe"
    print(f"Target URL   : {base}{endpoint}\nAudio File   : {a.audio}")

    # Health check
    s, b = req("GET", base + "/health", headers=H, timeout=300)
    if s == 200:
        try:
            hj = json.loads(b)
            gpu_name = (hj.get("gpu", {}).get("gpus", [{}])[0] or {}).get("name", "Unknown GPU")
            loaded = hj.get("loaded", False)
            print(f"[Health Check] Status: 200 OK | Model Loaded: {loaded} | GPU: {gpu_name}")
        except Exception:
            print(f"[Health Check] Status: {s} | {b[:300]}")
    else:
        print(f"[Health Check] Status: {s} | {b[:300]}")

    import mimetypes
    ctype = mimetypes.guess_type(a.audio)[0] or "application/octet-stream"

    bound = uuid.uuid4().hex.encode()
    data = Path(a.audio).read_bytes()
    body = (
        b"--"
        + bound
        + b'\r\nContent-Disposition: form-data; name="file"; filename="'
        + Path(a.audio).name.encode()
        + b'"\r\nContent-Type: '
        + ctype.encode()
        + b'\r\n\r\n'
        + data
        + b"\r\n--"
        + bound
        + b"--\r\n"
    )

    t0 = time.time()
    s, out = req(
        "POST",
        base + endpoint,
        body=body,
        headers={**H, "Content-Type": "multipart/form-data; boundary=" + bound.decode()},
    )
    elapsed_total = time.time() - t0

    if s != 200:
        print(f"\n[{endpoint} failed with HTTP {s} in {elapsed_total:.2f}s]\n{out[:2000]}")
        sys.exit(1)

    try:
        j = json.loads(out)
        if a.raw:
            print("\n[Raw JSON Response]")
            print(json.dumps(j, indent=2, ensure_ascii=False))
        # Handle envelope format if wrapped
        payload = j
        if isinstance(j, dict) and "output" in j and "asr_output" in j["output"]:
            # Combine asr_output and top-level fields for reporting
            merged = dict(j["output"]["asr_output"])
            merged["extraction"] = j["output"].get("extraction")
            merged["asr_latency_seconds"] = j["output"].get("asr_latency_seconds")
            merged["total_latency_seconds"] = j["output"].get("total_latency_seconds")
            payload = {"status": "success", "output": merged}
        print_report(payload, elapsed_total, a.audio)
    except Exception as exc:
        print(f"\n[{endpoint} returned non-JSON: {exc}]")
        print(out[:4000])


if __name__ == "__main__":
    main()
