#!/usr/bin/env python3
"""Serve Parrotlet ASR on a JarvisLabs GPU VM (or any Linux GPU server).

Runs FastAPI + Uvicorn with in-process GPU inference:
- HTTP routes, auth, validation mapping
- ParrotletWorker: inference, session buffering, SSE streaming

Run on the VM::

    python serve_jarvis.py --host 0.0.0.0 --port 6006

Or with authentication::

    AUTH_TOKEN=<secret> python serve_jarvis.py --port 6006

Test from your laptop::

    python test.py /path/to/audio.wav --url http://<vm-ip>:6006
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import asynccontextmanager

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.server import ParrotletWorker, build_fastapi_app


def build_app(worker_cls=None):
    """Build the FastAPI app with lifespan hooks for pre-warming."""
    app = build_fastapi_app(worker_cls=worker_cls)

    _prev_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _lifespan(api):
        from src.config import get_settings
        from src.gpu import configure_tf32, gpu_info
        from src.model import ensure_banned_ids, load_model

        # 1. TF32 acceleration for RTX 6000 Ada / Ampere GPUs
        try:
            settings = get_settings()
            tf32_status = configure_tf32(
                enabled=settings.allow_tf32,
                precision=settings.float32_matmul_precision,
            )
            print(f"[serve_jarvis] TF32 architecture configured: {tf32_status}", flush=True)
        except Exception as exc:
            print(f"[serve_jarvis] TF32 init warning: {exc}", flush=True)

        # 2. Warm the model at startup so the first request isn't paying load time
        t0 = time.perf_counter()
        print("[serve_jarvis] Pre-warming Parrotlet model on GPU...", flush=True)
        bundle = load_model()
        gpus = [g.get("name") for g in gpu_info().get("gpus", [])]
        print(
            f"[serve_jarvis] Model ready in {time.perf_counter() - t0:.1f}s | "
            f"placement={(getattr(bundle, 'metadata', None) or {}).get('placement')} | "
            f"gpus={gpus}",
            flush=True,
        )

        # 3. Pre-warm banned Indic token IDs
        try:
            ensure_banned_ids()
        except Exception as exc:  # non-fatal; ban resolves lazily
            print(f"[serve_jarvis] banned-ids prewarm skipped: {exc}", flush=True)

        if _prev_lifespan is not None:
            async with _prev_lifespan(api):
                yield
        else:
            yield

    app.router.lifespan_context = _lifespan
    return app


app = build_app()


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve Parrotlet ASR on JarvisLabs VM.")
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "6006")),
        help="Port to listen on (default: 6006)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("AUTH_TOKEN") or os.getenv("MODAL_AUTH_TOKEN", ""),
        help="Optional bearer token for API auth",
    )
    args = parser.parse_args()

    if args.token:
        os.environ["AUTH_TOKEN"] = args.token

    configured_token = os.getenv("AUTH_TOKEN") or os.getenv("MODAL_AUTH_TOKEN")
    auth_status = "ENABLED (Bearer token configured)" if configured_token else "DISABLED (Open endpoint)"

    print("=" * 66, flush=True)
    print("  Parrotlet ASR Service — JarvisLabs GPU VM", flush=True)
    print("=" * 66, flush=True)
    print(f"  Host:        http://{args.host}:{args.port}", flush=True)
    print(f"  Auth:        {auth_status}", flush=True)
    print(f"  Health URL:  http://{args.host}:{args.port}/health", flush=True)
    print(f"  Transcribe:  http://{args.host}:{args.port}/transcribe", flush=True)
    print("=" * 66, flush=True)

    uvicorn.run(
        "serve_jarvis:app",
        host=args.host,
        port=args.port,
    )
