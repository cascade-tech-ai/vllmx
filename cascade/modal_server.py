"""Modal deployment for vLLM (baseline).

This baseline mirrors Modal's example and stands up a vLLM server inside a
GPU-backed container. It intentionally does NOT wire in custom vllmx/static
text behavior yet — we will extend it in a follow-up.

Usage:
  modal serve cascade/modal_server.py

Environment overrides (optional):
  MODEL_ID               Hugging Face model id/path
  VLLM_PORT              Port to bind (default: 8000)
  N_GPU                  GPU count (default: 1)
  GPU_TYPE               GPU type (default: H100)
  FAST_BOOT              "1" for eager; "0" to allow compile (default: 1)
"""

import os
import modal

app = modal.App("example-vllm-inference")

# Volumes to avoid re-downloading weights / re-JITing artifacts
hf_cache_vol = modal.Volume.from_name("huggingface-cache",
                                      create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache",
                                       create_if_missing=True)

VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
N_GPU = int(os.getenv("N_GPU", "1"))
GPU_TYPE = os.getenv("GPU_TYPE", "H100")
MINUTES = 60

# FAST_BOOT=1 -> --enforce-eager, faster cold starts
FAST_BOOT = os.getenv("FAST_BOOT", "1") in ("1", "true", "True")

# Default model (tweak to your needs)
MODEL_ID = os.getenv("MODEL_ID",
                     "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8")

# Base CUDA image with Python.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04",
                              add_python="3.12")
    # Install latest packages (no pinned versions per request)
    .uv_pip_install(
        "vllm",
        "torch",
        "flashinfer-python",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Ensure v1 path in recent vLLM builds
        "VLLM_USE_V1": "1",
    })
)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    # Snapshotting (optional but speeds up cold starts)
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    # Typical server options
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
    ]

    # Note: we use Popen so the web_server decorator can keep the container
    # alive while vLLM runs in the background.
    subprocess.Popen(" ".join(cmd), shell=True)

