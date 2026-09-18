#!/usr/bin/env python3
"""Test ASR endpoint on JarvisLabs VM (or localhost). Stdlib only.

Usage:
    python test.py ./sample.wav
    python test.py ./sample.wav --url http://<vm-ip>:6006
    python test.py ./sample.wav --url http://<vm-ip>:6006 --token <secret>
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


def main():
    ap = argparse.ArgumentParser(description="Test Parrotlet ASR endpoint.")
    ap.add_argument("audio", nargs="?", default=os.getenv("TEST_AUDIO", AUDIO))
    ap.add_argument("--url", default=os.getenv("URL", URL))
    ap.add_argument("--token", default=os.getenv("AUTH_TOKEN") or os.getenv("MODAL_AUTH_TOKEN", ""))
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
    print(f"url: {base}\naudio: {a.audio}")
    s, b = req("GET", base + "/health", headers=H, timeout=300)
    print(f"\n[health {s}]\n{b[:1000]}")

    bound = uuid.uuid4().hex.encode()
    data = Path(a.audio).read_bytes()
    body = (b"--" + bound + b'\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            + data + b"\r\n--" + bound + b"--\r\n")
    t0 = time.time()
    s, out = req("POST", base + "/transcribe", body=body,
                 headers={**H, "Content-Type": "multipart/form-data; boundary=" + bound.decode()})
    print(f"\n[/transcribe {s} in {time.time()-t0:.1f}s]")
    try:
        j = json.loads(out)
        print(json.dumps(j, indent=2, ensure_ascii=False)[:8000])
        print("\n--- summary ---")
        print("transcript:", str(j.get("transcript") or j.get("text") or "")[:1000])
    except Exception:
        print(out[:4000])


if __name__ == "__main__":
    main()
