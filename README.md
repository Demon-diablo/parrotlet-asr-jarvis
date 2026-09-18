# Parrotlet ASR on JarvisLabs GPU VM

High-performance Parrotlet ASR service optimized for NVIDIA GPUs (RTX 6000 Ada / Ampere / A100) running natively on a JarvisLabs VM.

- **Direct in-process execution**: Fast inference with no serverless overhead.
- **Pre-warmed GPU pipelines**: Eliminates cold-start wait times.
- **TensorFloat-32 (TF32) acceleration**: Native Tensor Core speedups on Ada Lovelace / Ampere GPUs.
- **Indic script token ban**: Eliminates hallucinated script tokens, keeping clinical output in Latin English.
- **Complete ASR HTTP API**: Batch files, base64 payloads, buffered chunks, and Server-Sent Events (SSE) streaming.

## Endpoints

- `GET /` — Service status & available routes
- `GET /health` — GPU memory, model placement & load status
- `POST /transcribe` — Multipart audio upload transcription
- `POST /transcribe_b64` — JSON base64 audio transcription
- `POST /transcribe_chunk` — Streaming chunk ingestion & buffer management
- `POST /transcribe_stream` — Server-Sent Events (SSE) streaming transcription

---

## Running on JarvisLabs VM (or any GPU VM)

### 1. Launch Instance
- In [JarvisLabs.ai](https://jarvislabs.ai), launch an instance with an NVIDIA GPU (e.g. **RTX 6000 Ada / RTX 5000 / A100**).
- Choose the **PyTorch** template (PyTorch with CUDA 12.x).
- Connect via SSH or open the JupyterLab Terminal.

### 2. Setup Code & Dependencies
```bash
# Clone or copy repo onto the VM, then navigate into the directory:
cd "rtx 6000 pro og"

# Install system audio libraries (if not already installed)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# Install Python dependencies (FastAPI, Uvicorn, Hugging Face stack)
pip install -r requirements.txt
```

### 3. Verify Hardware & Download Weights
```bash
# Verify GPU detection
python3 scripts/check_gpu.py

# Verify TensorFloat-32 (TF32) support
python3 scripts/check_tf32.py

# (Optional) Pre-download model weights (~10GB) with progress indicator
python3 scripts/download_model.py
```

### 4. Start the Server
Run using the startup script:
```bash
bash run_jarvis.sh
```
Or directly with Python / Uvicorn:
```bash
# Default: runs on 0.0.0.0:6006 without authentication
python3 serve_jarvis.py

# With custom port and optional bearer token:
AUTH_TOKEN=mysecret PORT=6006 python3 serve_jarvis.py
```

### 5. Accessing & Testing the Endpoint
JarvisLabs routes port `6006` directly. You can find your endpoint URL in the JarvisLabs dashboard or use the public IP:
- URL: `http://<your-jarvis-instance-id-or-ip>:6006`

Test transcription from your local machine or terminal:
```bash
# Basic test (health check + file transcription)
python3 test.py /path/to/sample.wav --url http://<vm-ip>:6006

# If you configured an AUTH_TOKEN:
python3 test.py /path/to/sample.wav --url http://<vm-ip>:6006 --token mysecret
```

---

## Running Unit Tests

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

---

## TensorFloat-32 (TF32) Architecture Acceleration

The NVIDIA RTX 6000 Pro (48GB Ada Lovelace sm_89 / Ampere sm_86) features dedicated Tensor Cores supporting **TensorFloat-32 (TF32)** math mode.

### What is TF32?
- **Format**: 1 sign bit, 8 exponent bits (same dynamic range as FP32/BF16), and 10 mantissa bits (same precision as FP16) = 19 bits total.
- **Benefit**: Executes 32-bit floating-point matrix multiplications (GEMM) and convolutions on Tensor Cores at up to **4x-8x higher throughput** than standard FP32 CUDA cores, with zero loss in clinical transcription accuracy.

### Configuration
TF32 is fully configurable via environment variables:

| Variable | Default | Allowed | Description |
| :--- | :--- | :--- | :--- |
| `ALLOW_TF32` | `1` | `1`, `0` | Enables TF32 math mode for CUDA matmuls and cuDNN. |
| `FLOAT32_MATMUL_PRECISION` | `high` | `highest`, `high`, `medium` | PyTorch precision setting (`high` enables TF32 Tensor Cores). |
| `DTYPE` | `auto` | `auto`, `tf32`, `fp32`, `bf16`, `fp16` | Model dtype setting (`tf32` runs FP32 weights on TF32 Tensor Cores). |

### Verifying TF32 Support
Run the TF32 diagnostic and GEMM benchmark script:
```bash
python3 scripts/check_tf32.py
```
