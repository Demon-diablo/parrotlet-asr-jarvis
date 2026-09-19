#!/usr/bin/env python3
"""
MedGemma 4B-IT -> INT8 W8A8 Quantization & Hugging Face Deployment Script
Optimized for NVIDIA GPUs (RTX PRO 6000 Blackwell, Ada, Hopper, Ampere, Turing).

Pipeline:
1. Verifies GPU compute capability & environment.
2. Authenticates with Hugging Face using provided HF_TOKEN.
3. Loads google/medgemma-4b-it in bfloat16 (from local cache if present, else HF).
4. Quantizes decoder Linear layers to INT8 W8A8 (scheme="W8A8") via llm-compressor,
   preserving lm_head, embed_tokens, and vision/multimodal projections.
5. Saves compressed-tensors INT8 checkpoint and tokenizer/processor.
6. Bundles all required processor/tokenizer/preprocessor files.
7. Writes a complete Hugging Face Model Card (README.md) with metadata and vLLM usage.
8. Validates checkpoint files, config, and size.
9. Deploys/uploads the quantized model to Hugging Face repository Demondiablo/medgemma-4b-it-int8-w8a8.
10. Confirms all uploaded files on the Hugging Face Hub.
"""

import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Hugging Face Settings
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "Demondiablo/medgemma-4b-it-int8-w8a8"
MODEL_ID = "google/medgemma-4b-it"

# Directories
BASE_DIR = Path("/home/ubuntu/medgemma_int8")
OUTPUT_DIR = BASE_DIR / "medgemma-4b-it-int8-w8a8"
LOCAL_MODEL_CACHE = Path("/kaggle/working/medgemma_models/medgemma-4b-it")

GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}"

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
    import compressed_tensors
    
    try:
        import llmcompressor
        from llmcompressor.modifiers.quantization import QuantizationModifier
        try:
            from llmcompressor import oneshot
        except ImportError:
            from llmcompressor.transformers import oneshot
    except ImportError as exc:
        print(f"Error importing llmcompressor: {exc}")
        sys.exit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Hugging Face Hub: {huggingface_hub.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"LLM Compressor: {getattr(llmcompressor, '__version__', 'unknown')}")
    print(f"compressed-tensors: {getattr(compressed_tensors, '__version__', 'unknown')}")

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

    step_header("STEP 4: Configuring and Running INT8 W8A8 Quantization")
    INT8_IGNORE = [
        "re:.*lm_head.*",
        "re:.*embed_tokens.*",
        "re:.*vision.*",
        "re:.*visual.*",
        "re:.*multi_modal_projector.*",
    ]

    # scheme="W8A8" -> INT8 channel-wise weights + dynamic INT8 token-wise activations
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=INT8_IGNORE,
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

        print("Starting oneshot INT8 W8A8 quantization...")
        quant_t0 = time.perf_counter()

        oneshot(
            model=model,
            recipe=[recipe],
            trust_remote_code_model=True,
            output_dir=str(OUTPUT_DIR),
        )
        quant_time = time.perf_counter() - quant_t0

    print(f"Quantization finished in: {quant_time/60:.2f} minutes")

    print("Saving processor and tokenizer files to output directory...")
    processor.save_pretrained(OUTPUT_DIR)

    # Copy any additional configuration and tokenizer files from base model if present
    extra_files = [
        "generation_config.json",
        "chat_template.jinja",
        "added_tokens.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
    ]
    src_dir = LOCAL_MODEL_CACHE if LOCAL_MODEL_CACHE.exists() else None
    if src_dir:
        for fname in extra_files:
            if (src_dir / fname).exists():
                shutil.copy2(src_dir / fname, OUTPUT_DIR / fname)
                print(f"  ✓ Bundled {fname}")

    # Symlink to /kaggle/working for consistency
    kw_target = Path("/kaggle/working/medgemma-4b-it-int8-w8a8")
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
- int8
- w8a8
- compressed-tensors
- llm-compressor
- vllm
- medical
- healthcare
pipeline_tag: image-text-to-text
---

# MedGemma 4B-IT (INT8 W8A8 Dynamic)

This is an **INT8 W8A8 Quantized** version of [google/medgemma-4b-it](https://huggingface.co/{MODEL_ID}) optimized for high-throughput, low-latency deployment with **vLLM** and compatible serving engines across all modern GPU architectures.

## Quantization Details

- **Base Model:** `{MODEL_ID}`
- **Quantization Method:** [llm-compressor](https://github.com/vllm-project/llm-compressor)
- **Quantization Scheme:** `W8A8` (Weights in INT8 per-channel symmetric, activations in dynamic INT8 per-token)
- **Format:** `compressed-tensors` (native vLLM format)
- **Excluded Modules:** `lm_head`, `embed_tokens`, `multi_modal_projector`, and vision tower linear layers preserved for clinical accuracy.
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

INT8 W8A8 acceleration is supported across virtually all modern NVIDIA architectures:
- NVIDIA Turing (T4, RTX 2080 Ti)
- NVIDIA Ampere (A100, A10, RTX 3090, A6000)
- NVIDIA Ada Lovelace (RTX 4090, L4, L40S, RTX 6000 Ada)
- NVIDIA Hopper (H100, H200)
- NVIDIA Blackwell (RTX PRO 6000 Blackwell, B100, B200)
"""
    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content.strip() + "\n")
    print(f"Generated README.md in {OUTPUT_DIR}")

    step_header("STEP 6: Validating Checkpoint Size and Files")
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

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
    assert (OUTPUT_DIR / "model.safetensors").exists(), "model.safetensors is missing!"

    step_header("STEP 7: Deploying / Uploading INT8 Model to Hugging Face")
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
        commit_message="Add MedGemma 4B-IT INT8 W8A8 (compressed-tensors) checkpoint",
    )
    upload_s = time.perf_counter() - upload_t0
    print(f"Upload complete in: {upload_s/60:.2f} minutes ({upload_s:.1f}s)")

    step_header("STEP 8: Verifying Uploaded Files on Hugging Face Hub")
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
