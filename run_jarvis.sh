#!/usr/bin/env bash
# run_jarvis.sh - One-command launcher for Parrotlet ASR on JarvisLabs VM.
#
# Usage:
#   bash run_jarvis.sh
#   PORT=8000 AUTH_TOKEN=mysecret bash run_jarvis.sh

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "============================================================"
echo " Starting Parrotlet ASR on JarvisLabs VM"
echo "============================================================"

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
exec python3 serve_jarvis.py --host "$HOST" --port "$PORT"
