#!/usr/bin/env python3
"""Pre-download the Parrotlet model snapshot to the local Hugging Face cache.

Run on the VM before launching the server if you want to monitor
the download progress (~10 GB) with a progress bar::

    python scripts/download_model.py
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.config import get_settings


def main():
    settings = get_settings()
    repo_id = settings.model_id
    local_dir = settings.model_dir or None
    cache_dir = settings.model_cache_dir or None

    print("=" * 66)
    print(" Parrotlet Model Downloader")
    print("=" * 66)
    print(f" Target model:       {repo_id}")
    if local_dir:
        print(f" Destination dir:    {local_dir}")
    if cache_dir:
        print(f" Cache dir:          {cache_dir}")
    if settings.model_revision:
        print(f" Model revision:     {settings.model_revision}")
    print("=" * 66)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Please run:\n"
            "pip install -r requirements.txt"
        )

    kwargs = {}
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    if settings.model_revision:
        kwargs["revision"] = settings.model_revision
    if local_dir:
        kwargs["local_dir"] = local_dir
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    print("\nStarting snapshot download (this may take a few minutes for ~10 GB)...")
    t0 = time.time()
    try:
        path = snapshot_download(repo_id=repo_id, **kwargs)
    except Exception as exc:
        sys.exit(f"\n[ERROR] Download failed: {exc}")

    print(f"\n[SUCCESS] Download completed in {time.time() - t0:.1f}s.")
    print(f"Model files cached at: {path}\n")


if __name__ == "__main__":
    main()
