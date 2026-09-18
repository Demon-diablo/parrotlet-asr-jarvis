"""Loader-only diagnostic: verify everything BEFORE the model weights are
pulled into VRAM/RAM.

This is the right Phase 4 diagnostic when the local GPU is too small for the
real model. It exercises:

  - snapshot_download resolution against MODEL_ID + MODEL_REVISION
  - presence of modelling_speech-llm.py + auto_map wiring
  - presence + contents of decoder/, encoder/, projector/ subdirs
  - AutoConfig/AutoModel registration of SpeechLLMConfig/SpeechLLM
  - resolved config: dtype, model_type, vocab, expected sampling_rate

It deliberately does NOT call ``SpeechLLM.from_pretrained`` so the weights
are never allocated.

Usage:
    python scripts/check_loader.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings  # noqa: E402
from src.gpu import gpu_info  # noqa: E402


def _download_snapshot() -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    s = get_settings()
    kwargs: Dict[str, Any] = {"repo_id": s.model_source}
    if s.hf_token:
        kwargs["token"] = s.hf_token
    if s.model_revision:
        kwargs["revision"] = s.model_revision
    if s.model_cache_dir:
        kwargs["cache_dir"] = s.model_cache_dir
    else:
        kwargs["cache_dir"] = tempfile.mkdtemp(prefix="parrotlet-loader-check-")
    # Skip the heavy weight blobs. We want the modelling file + configs only.
    kwargs["allow_patterns"] = [
        "*.json",
        "*.jinja",
        "*.txt",
        "*.py",
        "modelling_speech-llm.py",
        "README.md",
    ]
    return snapshot_download(**kwargs)


def main() -> int:
    settings = get_settings()
    report: Dict[str, Any] = {
        "settings": settings.as_dict(),
        "runtime_gpu": gpu_info(),
    }

    # --- snapshot ------------------------------------------------------ #
    try:
        local_dir = _download_snapshot()
    except Exception as exc:  # pragma: no cover
        report["snapshot"] = {"ok": False, "error": str(exc)}
        print(json.dumps(report, indent=2, default=str))
        return 1
    report["snapshot"] = {"ok": True, "local_dir": local_dir}

    # --- file presence ------------------------------------------------- #
    expected_files = [
        "modelling_speech-llm.py",
        "config.json",
        "encoder/config.json",
        "encoder/preprocessor_config.json",
        "decoder/config.json",
        "decoder/preprocessor_config.json",
        "decoder/generation_config.json",
        "decoder/tokenizer_config.json",
        "projector/config.json",
    ]
    presence = {name: os.path.isfile(os.path.join(local_dir, name)) for name in expected_files}
    report["files_present"] = presence
    report["files_all_present"] = all(presence.values())

    # --- auto_map wiring ---------------------------------------------- #
    with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as f:
        root_cfg = json.load(f)
    report["auto_map"] = root_cfg.get("auto_map")
    report["root_model_type"] = root_cfg.get("model_type")

    # --- component configs -------------------------------------------- #
    def _load(name: str) -> Dict[str, Any]:
        with open(os.path.join(local_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    enc_cfg = _load("encoder/config.json")
    dec_cfg = _load("decoder/config.json")
    proj_cfg = _load("projector/config.json")

    report["encoder"] = {
        "model_type": enc_cfg.get("model_type"),
        "architectures": enc_cfg.get("architectures"),
        "d_model": enc_cfg.get("d_model"),
        "num_hidden_layers": enc_cfg.get("num_hidden_layers"),
        "torch_dtype": enc_cfg.get("torch_dtype"),
        "sampling_rate": enc_cfg.get("sampling_rate"),
    }
    enc_pre = _load("encoder/preprocessor_config.json")
    report["encoder_preprocessor"] = {
        "feature_extractor_type": enc_pre.get("feature_extractor_type"),
        "processor_class": enc_pre.get("processor_class"),
        "sampling_rate": enc_pre.get("sampling_rate"),
        "chunk_length": enc_pre.get("chunk_length"),
    }

    text_cfg = dec_cfg.get("text_config", {})
    report["decoder"] = {
        "model_type": dec_cfg.get("model_type"),
        "architectures": dec_cfg.get("architectures"),
        "torch_dtype": dec_cfg.get("torch_dtype"),
        "text_layers": text_cfg.get("num_hidden_layers"),
        "text_hidden": text_cfg.get("hidden_size"),
        "text_vocab": text_cfg.get("vocab_size"),
        "vision_layers": (dec_cfg.get("vision_config") or {}).get("num_hidden_layers"),
    }
    report["projector"] = {
        "model_type": proj_cfg.get("model_type"),
        "architectures": proj_cfg.get("architectures"),
        "encoder_dim": proj_cfg.get("encoder_dim"),
        "llm_dim": proj_cfg.get("llm_dim"),
        "linear_hidden_dim": proj_cfg.get("linear_hidden_dim"),
        "k": proj_cfg.get("encoder_projector_ds_rate"),
        "torch_dtype": proj_cfg.get("torch_dtype"),
    }

    # --- AutoConfig/AutoModel registration check --------------------- #
    reg: Dict[str, Any] = {}
    try:
        import importlib.util

        module_path = os.path.join(local_dir, "modelling_speech-llm.py")
        spec = importlib.util.spec_from_file_location("parrotlet_mod", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import custom modelling module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from transformers import AutoConfig, AutoModel  # type: ignore

        SpeechLLMConfig = getattr(mod, "SpeechLLMConfig")
        SpeechLLM = getattr(mod, "SpeechLLM")

        # AutoModel is a Placeholder when torch is not installed. We surface
        # that as a clean diagnostic instead of an opaque stack trace.
        if not hasattr(AutoModel, "register"):
            reg["error"] = (
                "torch is not installed; AutoModel is a Placeholder and "
                "model weights cannot be loaded until torch is available "
                "(the Docker image installs torch automatically)"
            )
            reg["registered_with_autoconfig"] = False
            reg["registered_with_automodel"] = False
        else:
            AutoConfig.register("speech-llm", SpeechLLMConfig)
            AutoModel.register(SpeechLLMConfig, SpeechLLM)
            # Confirm AutoConfig can read the repo root.
            resolved_config = AutoConfig.from_pretrained(local_dir)
            reg["resolved_config_class"] = type(resolved_config).__name__
            reg["resolved_model_type"] = getattr(resolved_config, "model_type", None)
            reg["registered_with_autoconfig"] = True
            reg["registered_with_automodel"] = True
    except Exception as exc:  # pragma: no cover
        reg["error"] = str(exc)
        reg["registered_with_autoconfig"] = False
        reg["registered_with_automodel"] = False
    report["auto_registration"] = reg

    print(json.dumps(report, indent=2, default=str))
    return 0 if (report.get("files_all_present") and reg.get("registered_with_autoconfig")) else 1


if __name__ == "__main__":
    raise SystemExit(main())