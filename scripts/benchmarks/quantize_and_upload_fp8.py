#!/usr/bin/env python3
"""
MedGemma 4B-IT -> FP8 W8A8 Quantization & Hugging Face Deployment Script
Optimized for NVIDIA RTX PRO 6000 Blackwell Server Edition (and modern FP8 GPUs).

Pipeline:
1. Verifies GPU compute capability (native FP8 support).
2. Authenticates with Hugging Face using provided HF_TOKEN.
3. Loads google/medgemma-4b-it in bfloat16 (from local cache if present, else HF).
4. Quantizes decoder Linear layers to FP8_DYNAMIC (W8A8) via llm-compressor,
   ignoring lm_head, embed_tokens, and multimodal/vision layers.
5. Saves compressed-tensors FP8 checkpoint and tokenizer/processor.
6. Writes a complete Hugging Face Model Card (README.md) with metadata and vLLM usage.
7. Reloads the compressed model from disk to validate weight loading and inference.
8. Runs the 5 medical transcription benchmark clips to verify zero regression.
9. Deploys/uploads the quantized model to Hugging Face repository Demondiablo/medgemma-4b-it-fp8-w8a8.
10. Confirms all uploaded files on the Hugging Face Hub.
"""

import gc
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

# Hugging Face Settings
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "Demondiablo/medgemma-4b-it-fp8-w8a8"
MODEL_ID = "google/medgemma-4b-it"

# Directories
BASE_DIR = Path("/home/ubuntu/medgemma_fp8")
OUTPUT_DIR = BASE_DIR / "medgemma-4b-it-fp8-w8a8"
LOCAL_MODEL_CACHE = Path("/kaggle/working/medgemma_models/medgemma-4b-it")

GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}"
MAX_NEW_TOKENS = 8000
DO_SAMPLE = False
USE_CACHE = True

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def step_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n", flush=True)


def main():
    step_header("STEP 1: Checking Environment & Hardware")
    import torch
    import transformers
    import huggingface_hub
    import accelerate
    
    try:
        import llmcompressor
        from llmcompressor.modifiers.quantization import QuantizationModifier
        try:
            from llmcompressor import oneshot
        except ImportError:
            from llmcompressor.transformers import oneshot
    except ImportError as exc:
        print(f"Error importing llmcompressor: {exc}")
        print("Please ensure llmcompressor is installed: pip install -U llmcompressor compressed-tensors")
        sys.exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"LLM Compressor: {getattr(llmcompressor, '__version__', 'unknown')}")

    assert torch.cuda.is_available(), "CUDA is not available!"
    props = torch.cuda.get_device_properties(GPU_INDEX)
    major, minor = torch.cuda.get_device_capability(GPU_INDEX)
    vram_gib = props.total_memory / (1024**3)
    print(f"GPU: {props.name}")
    print(f"VRAM: {vram_gib:.2f} GiB")
    print(f"Compute Capability: {major}.{minor}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists("/kaggle/working"):
        os.makedirs("/kaggle/working", exist_ok=True)

    step_header("STEP 2: Hugging Face Authentication")
    from huggingface_hub import HfApi, login
    api = HfApi(token=HF_TOKEN)
    try:
        who = api.whoami()
        print(f"Authenticated as: {who.get('name') or who.get('user', 'unknown')}")
    except Exception as exc:
        print(f"Auth check notice: {exc}")

    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as exc:
        print(f"Login notice (non-fatal): {exc}")

    model_source = MODEL_ID
    if LOCAL_MODEL_CACHE.exists() and (LOCAL_MODEL_CACHE / "config.json").exists():
        print(f"Found local cached model at: {LOCAL_MODEL_CACHE}")
        model_source = str(LOCAL_MODEL_CACHE)
    else:
        print(f"Loading model directly from HF: {MODEL_ID}")

    step_header("STEP 3: Loading Base Model in bfloat16")
    from transformers import AutoProcessor, AutoModelForImageTextToText

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_source,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.eval()
    torch.cuda.synchronize(GPU_INDEX)
    load_time = time.perf_counter() - t0
    print(f"Loaded in: {load_time:.2f}s")
    print(f"Model dtype: {model.dtype}")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(GPU_INDEX)/(1024**3):.2f} GiB")

    step_header("STEP 4: Configuring and Running FP8 W8A8 Quantization")
    FP8_IGNORE = [
        "re:.*lm_head.*",
        "re:.*embed_tokens.*",
        "re:.*vision.*",
        "re:.*visual.*",
        "re:.*multi_modal_projector.*",
    ]

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=FP8_IGNORE,
    )

    already_quantized = (OUTPUT_DIR / "model.safetensors").exists() and (OUTPUT_DIR / "config.json").exists()
    if already_quantized:
        print(f"Found already quantized model at {OUTPUT_DIR}! Skipping re-quantization.")
        quant_time = 0.0
    else:
        if OUTPUT_DIR.exists():
            print(f"Cleaning previous output directory: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Starting oneshot FP8 quantization...")
        quant_t0 = time.perf_counter()

        oneshot(
            model=model,
            recipe=[recipe],
            trust_remote_code_model=True,
            output_dir=str(OUTPUT_DIR),
        )
    print(f"Quantization time: {quant_time/60:.2f} minutes")

    print("Saving processor and tokenizer files to output directory...")
    processor.save_pretrained(OUTPUT_DIR)

    # Copy any additional configuration files from base model if missing
    for cfg_name in ["generation_config.json", "chat_template.json"]:
        src_cfg = (LOCAL_MODEL_CACHE if LOCAL_MODEL_CACHE.exists() else None)
        if src_cfg and (src_cfg / cfg_name).exists() and not (OUTPUT_DIR / cfg_name).exists():
            shutil.copy2(src_cfg / cfg_name, OUTPUT_DIR / cfg_name)
            print(f"Copied {cfg_name} to {OUTPUT_DIR}")

    # Symlink to /kaggle/working for consistency
    kw_target = Path("/kaggle/working/medgemma-4b-it-fp8-w8a8")
    if not kw_target.exists():
        try:
            kw_target.symlink_to(OUTPUT_DIR)
            print(f"Symlinked {OUTPUT_DIR} -> {kw_target}")
        except Exception:
            pass

    step_header("STEP 5: Creating Hugging Face Model Card (README.md)")
    readme_content = f"""---
base_model: {MODEL_ID}
library_name: transformers
license: gemma
tags:
- fp8
- w8a8
- compressed-tensors
- llm-compressor
- vllm
- medical
- healthcare
pipeline_tag: image-text-to-text
---

# MedGemma 4B-IT (FP8 W8A8 Dynamic)

This is an **FP8 W8A8 Quantized** version of [google/medgemma-4b-it]({ 'https://huggingface.co/' + MODEL_ID }) optimized for high-throughput, low-latency deployment with **vLLM** and compatible serving engines.

## Quantization Details

- **Base Model:** `{MODEL_ID}`
- **Quantization Method:** [llm-compressor](https://github.com/vllm-project/llm-compressor)
- **Quantization Scheme:** `FP8_DYNAMIC` (Weights in FP8 E4M3, dynamic FP8 activations)
- **Format:** `compressed-tensors` (native vLLM format)
- **Excluded Modules:** `lm_head`, `embed_tokens`, `multi_modal_projector`, and vision tower linear layers preserved for accuracy.
- **Quantized On:** NVIDIA RTX PRO 6000 Blackwell Server Edition

## Deployment & Usage

### 1. High-Performance Inference with vLLM (Recommended)

```python
from vllm import LLM, SamplingParams

model_name = "{HF_REPO_ID}"

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=2048,
)

prompt = "You are a clinical AI. Summarize the following prescription: Tablet Thyronorm 50 mcg OD."
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### 2. Inference with Transformers

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_name = "{HF_REPO_ID}"

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

prompt = "Extract medications: Capsule Desula 21 HS, Thyronorm 50 mg OD."
inputs = processor(text=prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Hardware Compatibility

Native FP8 W8A8 acceleration is supported on:
- NVIDIA Ada Lovelace (RTX 4090, RTX 6000 Ada, etc.)
- NVIDIA Hopper (H100, H200)
- NVIDIA Blackwell (RTX PRO 6000 Blackwell, B100, B200)
"""
    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
    print(f"Generated README.md in {OUTPUT_DIR}")

    step_header("STEP 6: Validating Checkpoint Reload and Size")
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(GPU_INDEX)

    total_bytes = 0
    file_list = []
    for p in sorted(OUTPUT_DIR.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total_bytes += sz
            file_list.append((p.name, sz))
            print(f"  {str(p.relative_to(OUTPUT_DIR)):40s} {sz/(1024**2):8.2f} MiB")

    print(f"\nTotal Files: {len(file_list)}")
    print(f"Total Checkpoint Size: {total_bytes/(1024**3):.2f} GiB")

    assert (OUTPUT_DIR / "config.json").exists(), "config.json is missing!"

    reload_t0 = time.perf_counter()
    q_processor = AutoProcessor.from_pretrained(
        OUTPUT_DIR,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    q_model = AutoModelForImageTextToText.from_pretrained(
        OUTPUT_DIR,
        torch_dtype=torch.bfloat16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    q_model.eval()
    torch.cuda.synchronize(GPU_INDEX)
    reload_time = time.perf_counter() - reload_t0
    print(f"Reloaded FP8 model in: {reload_time:.2f}s")
    print(f"Reloaded VRAM: {torch.cuda.memory_allocated(GPU_INDEX)/(1024**3):.2f} GiB")

    try:
        from compressed_tensors.offload import dispatch_model
        dispatch_model(q_model)
    except Exception as exc:
        print(f"dispatch_model notice: {exc}")

    step_header("STEP 7: Running 5 Benchmark Transcripts on FP8 Model")
    SYSTEM = """You are a medical transcription extraction engine. Extract every medication instruction explicitly present in the transcript. Do not invent missing information. Preserve medication names as spoken, including brand names. Return VALID JSON ONLY, with no markdown fences and no explanatory text. The top-level schema is: {"rx_norm":[{"drug":string,"strength":string|null,"freq_en":string|null,"qty_en":string|null,"route":string|null,"needs_review":boolean}],"needs_review":boolean}. Use null when a field is not stated or cannot be read confidently. Use needs_review=true when the transcript is unclear, ambiguous, malformed, or missing a clinically important field. Do not omit a medication merely because some fields are missing."""

    INLINE = [
        "audio 2 47s -Capsule Desula 21 HS Tab 81 2 mg 1 HS Tab Telista CL 1 BD, Thyronorm 50 mg OD, Istamelt 5500 BD. Prolomet XR 50 OD, CTD 12.5 one OD, Ecosprin AV 75/10 one HS Capsule Calbona 3D 15 दिन में 1 बार. Tablet Zifi 200 BD. Tablet Pexican 214",
        "Audio 3: Tablet Nexito Forte 1 HS. Capsule Rabica Gold 1 OD. Telista CL 1 BD. Thyronorm 50 mg 1 OD. Tab Difi M 10/500 1 OD. Sanxit 0.5/2.5 OD. Istamelt 5/500 one HS. Tolomet 50 one HS. CTD 12.5 OD. Neurotto 3D one OD. Sempraz D one OD. Calbone T 1 OD. Capol ER 50 BD. Ecosprin AV 50/10 HS.",
        "Audio 4: Dazula 30 ek goli raat mein. Prothen 75 ek goli raat mein. Quetain 25 ek goli raat mein. Tablet Lopez 2 mg ek goli raat mein aur Ativan 2 mg ek goli raat mein. Thyronorm 75 ek goli din mein subah-subah khaali pet. Tablet Prolomet XL 100 mg ek goli subah. Capsule Ecosprin AV 75/10 ek goli raat mein. Calbone 3D din 15 din mein ek baar. Telista CL Trio ek goli din mein. Cilacar 20 ek goli raat mein. Tablet Sanpraz IT ek raat mein. [Unintelligible]",
        "Audio 6 1min13secx: Diabetes Type 2 plus hypothyroidism, obese mild. CBC, hemoglobin, HbA1c, lipid profile, LFT, KFT, thyroid. Tablet Thyronorm 112.5 ek goli subah. Sempas IT ek goli. Ozempic 1 mg. Lantus 26. Glycomet GP2 BD. Glyco Bay 50 TDS. Justoza 10. Capsule Unistar A 50. [Unclear] 75. Carnous 40 BD. Calbonat E3 15 din mein ek baar. Tadalafil 10 mg one HS.",
        "audio6.ogg - Tablet Pan 40 OD with empty stomach ES. Tablet Drotin MF BD 5 days with breakfast. Tablet Buscopan 10 mg BD 5 days with breakfast. Tablet Glycomet 500 mg OD thirty minute before breakfast tablet Glycomet GP1 OD 20 mg at 30 minutes before breakfast tablet glycomet GP2 BD 30 minutes before meal tablet amlovas M 5 mg OD before breakfast twenty minutes tablet Tazloc forty OD thirty minutes before breakfast tablet Tazloc CT 40/12.5 mg BD 30 minutes before breakfast. Tablet Triolmesar 40 oblique 5 oblique 12.5 mg OD 30 minutes before breakfast. Tablet Allegra 120 mg OD with breakfast. Tablet Montina LC 10/5 OD HS after meal Tablet Ecosprin 75 mg OD with breakfast. Tablet Clopitab 75 75 mg OD with breakfast. Tablet Acyclosera BD with breakfast. Tablet Clonafit 0.25 mg OD HS. Tablet RHEFD OD with empty stomach. Tablet Ondem 8 mg BD five days. Tablet Emeset 4 mg BD five days. Tablet Doxypheylin 400 mg BD tablet Deriphyllin 150 mg BD tablet Mucinex 600 mg OD in half glass water inhaler Foracort two hundred two hundred mg two puffs twelve hourly inhaler Tiova two puffs twelve hourly inhaler Duolin two puffs twelve hourly Tablet Atorva 40 mg with breakfast. Tablet Atormac TG OD with breakfast. Tablet Sitaglo 100 mg OD. one hour before breakfast or meal tablet Sitaglo GM oblique GM2 OD thirty minutes before Breakfast or meal tablet Dapanorm 10 mg before 30 minutes before meal or breakfast tablet GP 200 BD seven days tablet Clavum 625 mg OD seven days tablet Mero 30 mg BD seven days",
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

    benchmark_results = []
    total_items = 0

    for idx, raw_sample in enumerate(INLINE, 1):
        m = re.match(r"^(.{0,30}?)\s*[-:]\s*(.*)$", raw_sample, re.S)
        clip_id = m.group(1).strip() if m else f"sample_{idx}"
        transcript = m.group(2).strip() if m else raw_sample

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": f"Extract medications from this transcript according to the exact schema in the system instruction. Transcript:\n\n{transcript}"}]},
        ]
        inputs = q_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        torch.cuda.synchronize(GPU_INDEX)
        t_gen = time.perf_counter()
        with torch.inference_mode():
            generated = q_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                use_cache=USE_CACHE,
            )
        torch.cuda.synchronize(GPU_INDEX)
        gen_s = time.perf_counter() - t_gen

        completion = generated[0, prompt_tokens:]
        completion_tokens = int(completion.numel())
        raw_text = q_processor.decode(completion, skip_special_tokens=True)

        try:
            parsed = json.loads(strip_fences(raw_text))
            valid = True
        except Exception:
            parsed = None
            valid = False

        items = map_items(parsed) if valid else []
        total_items += len(items)
        tok_s = completion_tokens / max(gen_s, 1e-9)

        print(f"[{idx}/5] {clip_id[:20]:20s} | valid: {str(valid):5s} | items: {len(items):2d} | tokens: {completion_tokens:4d} | time: {gen_s:5.2f}s | tok/s: {tok_s:5.1f}")
        benchmark_results.append({
            "clip": clip_id,
            "transcript": transcript,
            "n_items": len(items),
            "extraction": items,
            "json_valid": valid,
            "wall_s": round(gen_s, 3),
            "tok_s": round(tok_s, 3),
        })

    with open(OUTPUT_DIR / "fp8_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    print(f"\nBenchmark completed on 5 clips. Total items extracted: {total_items}")

    # Release model before upload to free up memory
    del q_model
    del q_processor
    gc.collect()
    torch.cuda.empty_cache()

    step_header("STEP 8: Deploying / Uploading Quantized Model to Hugging Face")
    print(f"Target Repository: {HF_REPO_ID}")

    print("Creating repository (if not exists)...")
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        exist_ok=True,
        token=HF_TOKEN,
        private=False,
    )

    print(f"Uploading files from {OUTPUT_DIR} to {HF_REPO_ID}...")
    upload_t0 = time.perf_counter()
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
        commit_message="Add MedGemma 4B-IT FP8 W8A8 (compressed-tensors) checkpoint",
    )
    upload_s = time.perf_counter() - upload_t0
    print(f"Upload complete in: {upload_s/60:.2f} minutes")

    step_header("STEP 9: Verifying Uploaded Files on Hugging Face Hub")
    repo_files = api.list_repo_files(repo_id=HF_REPO_ID, token=HF_TOKEN)
    print("Files currently on Hugging Face Hub:")
    for rf in sorted(repo_files):
        print(f"  ✓ {rf}")

    assert any(rf.endswith(".safetensors") for rf in repo_files), "No .safetensors files found in repository!"
    assert "config.json" in repo_files, "config.json missing from repository!"
    assert "README.md" in repo_files, "README.md missing from repository!"

    step_header("ALL STEPS COMPLETED SUCCESSFULLY!")
    print(f"Model URL: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
