# Jarvis SGLang: Low-Latency ASR & Clinical Extraction Engine

Ultra-low-latency, zero-hop clinical consultation transcription and medication extraction service engineered for **NVIDIA RTX PRO 6000 (96GB / 48GB Ada Lovelace / Blackwell)** running natively on a GPU VM (JarvisLabs / Bare-Metal).

- **Zero-Hop In-Process Co-Hosting**: Parrotlet ASR and MedGemma-4B-it co-hosted in the same Python process on GPU 0, eliminating cross-process serialization and networking overhead.
- **Strict Zero-Quantization (Native BF16)**: Full unquantized precision for both model weights and KV cache states.
- **SGLang RadixAttention Trie**: LRU prefix caching achieves **<5ms prompt TTFT** by caching common clinical system prompts.
- **Low-Batch Decode CUDA Graphs (`BS=[1, 2]`)**: Bypasses the ~1.5ms/token CPU kernel dispatch bubble for single-stream interactive inference.
- **FlashInfer Cooperative Warp Kernels**: Ultra-fast decode attention kernels.
- **TensorFloat-32 (TF32) Acceleration**: Hardware TF32 on Tensor Cores enabled globally.
- **Complete Streaming Surface**: Full Server-Sent Events (SSE) streaming preserved across all endpoints.
- **Standard Clinical JSON Schema**: Clean, standard dictionary output format (`{"medications": [...]}`).

---

## 1. Latency Architecture: Why SGLang on RTX 6000 Pro?

| Parameter | Baseline vLLM (`parrotlet-asr-jarvis`) | **Optimized SGLang (`jarvis-sglang`)** |
| :--- | :--- | :--- |
| **CUDA Graphs** | ❌ Disabled (`enforce_eager=True`) | ✅ **Enabled (`cuda_graph_max_bs_decode=2`, `[1, 2]`)** |
| **Attention Backend** | Eager fallback | ✅ **FlashInfer Cooperative Warp Kernels** |
| **System Prompt TTFT** | ~380 ms (Cold Prefill) | ✅ **<5 ms (Warm Radix Trie Cache Hit)** |
| **Boot Pre-Warming** | ❌ None | ✅ **`warmup_prefix_cache()` pre-warms trie & CUDA graph** |
| **TPOT (Per-Token)** | ~22.0 ms / token (~45 tok/s) | ✅ **~3.9 ms / token (~250 tok/s)** |
| **Audio 6 Extraction** | **45.6s - 47.4s** | ✅ **Sub-8s interactive turnaround** |

---

## 2. API Endpoints

### Informational & Health
- `GET /` — Service status & available routes
- `GET /health` — GPU memory, model placement & load status

### Speech Transcription (Parrotlet ASR)
- `POST /transcribe` — Multipart audio upload transcription
- `POST /transcribe_b64` — JSON base64 audio transcription
- `POST /transcribe_chunk` — Session-buffered streaming chunk ingestion
- `POST /transcribe_stream` — Server-Sent Events (SSE) streaming transcription (`event: window`, `event: final`)

### Clinical Medication Extraction (SGLang MedGemma-4B)
- `POST /extract` — Standard clinical JSON extraction from transcript
- `POST /extract_stream` — SSE streaming extraction (`event: extraction_start`, `event: token`, `event: extraction_complete`)

### Zero-Hop End-to-End Pipeline
- `POST /pipeline` — Audio $\rightarrow$ Parrotlet ASR $\rightarrow$ SGLang MedGemma (single call)
- `POST /pipeline_stream` — Full SSE pipeline (`event: window` $\rightarrow$ `event: asr_complete` $\rightarrow$ `event: extraction_complete`)

---

## 3. Quick Start: Running on GPU VM

### 1. Requirements & System Dependencies
```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Verify Hardware
```bash
# Verify GPU detection (RTX 6000 Pro 96GB / 48GB)
python3 scripts/check_gpu.py

# Verify TF32 Tensor Core support
python3 scripts/check_tf32.py
```

### 3. Launch Server
```bash
# Start server with default RTX 6000 Pro configuration (Port 6006)
bash run_jarvis.sh

# Or run with custom port and authentication:
PORT=8000 AUTH_TOKEN=mysecret bash run_jarvis.sh
```

---

## 4. Testing & Verification

Run the client benchmark against the active server:

```bash
# Test full pipeline on sample audio:
python3 test.py audio/audio6.ogg --url http://localhost:6006

# Test with authentication:
python3 test.py audio/audio6.ogg --url http://localhost:6006 --token mysecret

# View raw JSON response payload:
python3 test.py audio/audio6.ogg --raw
```

### Running Unit Tests (No GPU Required)
```bash
pip install -r requirements-dev.txt
pytest tests -q
```

---

## 5. Configuration Reference

All settings can be customized via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EXTRACTOR_BACKEND` | `sglang` | Extraction engine (`sglang`, `vllm`, `transformers`, `none`) |
| `SGLANG_MEM_FRACTION` | `0.50` | Static KV cache pool fraction (~48GB on 96GB GPU, ~24GB on 48GB) |
| `SGLANG_CONTEXT_LEN` | `8192` | Maximum context length for SGLang |
| `SGLANG_ATTENTION_BACKEND`| `flashinfer`| FlashInfer cooperative warp decode kernels |
| `ALLOW_TF32` | `1` | Enables TF32 math on NVIDIA Tensor Cores |
| `FLOAT32_MATMUL_PRECISION`| `high` | PyTorch matmul precision setting |
| `BAN_SCRIPT_TOKENS` | `1` | Suppresses non-Latin Indic script hallucination |
| `CLEAN_TRANSCRIPT` | `1` | Strips audio noise tags and unrolls gloss brackets |
| `AUTH_TOKEN` | `""` | Optional Bearer token for securing HTTP routes |
| `PORT` | `6006` | Server HTTP port |
