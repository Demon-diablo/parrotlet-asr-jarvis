#!/usr/bin/env bash
# scripts/setup_vm_env.sh - One-shot VM environment setup & persistence verification
#
# Ensures system libraries (ffmpeg, libsndfile1), Python packages (sglang, flashinfer),
# FlashInfer CCCL compatibility patch, and CUDA 13 symlinks are all configured.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo " Verifying JarvisLabs VM Environment for SGLang & Parrotlet"
echo "============================================================"

# 1. System packages (ffmpeg and libsndfile1 for audio decoding)
if ! command -v ffmpeg &>/dev/null || [ ! -f /usr/lib/x86_64-linux-gnu/libsndfile.so.1 ]; then
    echo "[1/4] Installing system audio libraries (ffmpeg, libsndfile1)..."
    apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1
else
    echo "[1/4] System audio libraries verified (ffmpeg + libsndfile1)."
fi

# 2. Hugging Face authenticated token
mkdir -p /root/.cache/huggingface
if [ -f "/home/.cache/huggingface/token" ] && [ ! -f "/root/.cache/huggingface/token" ]; then
    cp /home/.cache/huggingface/token /root/.cache/huggingface/token
    echo "[2/4] Hugging Face token synced from persistent storage."
elif [ -f "/root/.cache/huggingface/token" ]; then
    echo "[2/4] Hugging Face token verified."
else
    echo "[2/4] Warning: HF_TOKEN not found. Gated models like MedGemma may require authentication."
fi

# 3. Conda Python 3.10 environment
if [ -f "/root/miniconda3/bin/activate" ]; then
    source "/root/miniconda3/bin/activate" py3.10
    export PATH="/root/miniconda3/envs/py3.10/bin:$PATH"
fi

# 4. Check core python packages
NEED_INSTALL=0
python3 -c "import sglang, flashinfer, fastapi, uvicorn, soundfile, librosa, resampy, nest_asyncio" 2>/dev/null || NEED_INSTALL=1

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[3/4] Installing missing Python packages from local wheel cache..."
    pip install -q --no-index --find-links=/home/.cache/pip/wheels \
        'sglang==0.5.20' 'flashinfer-python==0.6.18' fastapi uvicorn python-multipart soundfile librosa resampy nest_asyncio 2>/dev/null || \
    pip install -q 'sglang==0.5.20' 'flashinfer-python==0.6.18' fastapi uvicorn python-multipart soundfile librosa resampy nest_asyncio
else
    echo "[3/4] Python ML & server stack verified (SGLang, FlashInfer, FastAPI, Librosa)."
fi

# 5. FlashInfer CCCL compatibility patch for Blackwell / CUDA 13
CCCL_HEADER="/root/miniconda3/envs/py3.10/lib/python3.10/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h"
if [ -f "$CCCL_HEADER" ]; then
    if ! head -n 1 "$CCCL_HEADER" | grep -q "CCCL_DISABLE_CTK_COMPATIBILITY_CHECK"; then
        sed -i '1i #define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 1' "$CCCL_HEADER"
        echo "[4/4] FlashInfer CCCL compatibility check patched."
        # Clear any failed build cache from previous incompatible attempts
        rm -rf /home/.cache/sglang/.cache/flashinfer ~/.cache/flashinfer
    else
        echo "[4/4] FlashInfer CCCL patch verified."
    fi
fi

# 6. CUDA 13 cu13 linker symlinks for FlashInfer JIT
CU13_DIR="/root/miniconda3/envs/py3.10/lib/python3.10/site-packages/nvidia/cu13"
if [ -d "$CU13_DIR" ]; then
    cd "$CU13_DIR"
    [ ! -d lib64 ] && ln -sf lib lib64
    mkdir -p lib64/stubs
    [ -f /usr/lib/x86_64-linux-gnu/libcuda.so ] && ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so lib64/stubs/libcuda.so
    [ -f lib/libcudart.so.13 ] && ln -sf libcudart.so.13 lib/libcudart.so
    cd "$DIR"
fi

echo "============================================================"
echo " VM Environment Ready for Execution!"
echo "============================================================"
