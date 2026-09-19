#!/usr/bin/env bash
# run_jarvis.sh - One-command launcher for Parrotlet ASR + SGLang on RTX 6000 Pro (96GB / 48GB).
#
# Usage:
#   bash run_jarvis.sh
#   PORT=6006 AUTH_TOKEN=mysecret bash run_jarvis.sh

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================================"
echo " Starting Parrotlet ASR + SGLang on GPU VM (RTX 6000 Pro)"
echo "============================================================"

# Auto-verify/bootstrap environment if script exists
if [ -f "$DIR/scripts/setup_vm_env.sh" ]; then
    bash "$DIR/scripts/setup_vm_env.sh"
fi

PYTHON_BIN="python3"
# Activate virtual environment or conda if present
if [ -d "/root/miniconda3/envs/py3.10" ]; then
    export PATH="/root/miniconda3/envs/py3.10/bin:$PATH"
    PYTHON_BIN="/root/miniconda3/envs/py3.10/bin/python"
    [ -f "/root/miniconda3/bin/activate" ] && source "/root/miniconda3/bin/activate" py3.10 || true
elif [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
    PYTHON_BIN="python"
elif [ -d "$DIR/venv" ]; then
    source "$DIR/venv/bin/activate"
    PYTHON_BIN="python"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[Venv] Using active virtualenv: $VIRTUAL_ENV"
    PYTHON_BIN="python"
fi

# Set CUDA 13.x compiler and library paths if available
if [ -d "/root/miniconda3/envs/py3.10/lib/python3.10/site-packages/nvidia/cu13/bin" ]; then
    export PATH="/root/miniconda3/envs/py3.10/lib/python3.10/site-packages/nvidia/cu13/bin:$PATH"
    export CUDA_HOME="/root/miniconda3/envs/py3.10/lib/python3.10/site-packages/nvidia/cu13"
fi

# Set Hugging Face cache and token if available
export HF_HOME="${HF_HOME:-/home/.cache/huggingface}"
if [ -f "/root/.cache/huggingface/token" ] && [ -z "$HF_TOKEN" ]; then
    export HF_TOKEN="$(cat /root/.cache/huggingface/token)"
fi

# Verify GPU is accessible
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU Detection]"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are loaded."
fi

# Set host/port defaults
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-6006}"

# Optimization defaults for RTX 6000 Pro (96GB / 48GB)
unset FLASHINFER_CUDA_ARCH_LIST
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1 $NVCC_PREPEND_FLAGS"
export ALLOW_TF32="${ALLOW_TF32:-1}"
export FLOAT32_MATMUL_PRECISION="${FLOAT32_MATMUL_PRECISION:-high}"
export EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-sglang}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.45}"
export SGLANG_CONTEXT_LEN="${SGLANG_CONTEXT_LEN:-8192}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"

# Optional Auth Token
if [ -n "$AUTH_TOKEN" ]; then
    echo "[Auth] Authentication ENABLED via AUTH_TOKEN"
elif [ -n "$MODAL_AUTH_TOKEN" ]; then
    export AUTH_TOKEN="$MODAL_AUTH_TOKEN"
    echo "[Auth] Authentication ENABLED via legacy MODAL_AUTH_TOKEN"
else
    echo "[Auth] Authentication DISABLED (Open endpoint). Set AUTH_TOKEN=... to protect it."
fi

# Run the server
exec "$PYTHON_BIN" serve_jarvis.py --host "$HOST" --port "$PORT"
