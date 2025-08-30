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

CLI overrides (preferred):
  --model-id <id>        Model id/path
  --vllm-port <port>     Port to bind
  --n-gpu <N>            GPU count
  --gpu-type <name>      GPU type (e.g., H100, A100)

Any additional CLI args are forwarded to `vllm serve ...` unchanged. We also
enforce a default `--served-model-name predict` unless already provided.
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
    # Propagate local overrides into the container's env for dev `serve`.
    .env({
        "MODEL_ID": MODEL_ID,
        "VLLM_PORT": str(VLLM_PORT),
        "N_GPU": str(N_GPU),
        "GPU_TYPE": GPU_TYPE,
        "VLLM_ARGS": os.getenv("VLLM_ARGS", ""),
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

    # Echo effective configuration
    print(
        f"[serve] MODEL_ID={MODEL_ID} N_GPU={N_GPU} GPU_TYPE={GPU_TYPE} VLLM_PORT={VLLM_PORT}",
        flush=True,
    )
    print(f"[serve] extra VLLM_ARGS={os.getenv('VLLM_ARGS','')}", flush=True)

    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    import shlex
    extra_args = shlex.split(os.getenv("VLLM_ARGS", ""))
    # If user did not specify TP explicitly, default TP to N_GPU.
    def _has_opt(args: list[str], names: list[str]) -> bool:
        for n in names:
            if n in args:
                return True
            if any(a.startswith(n + "=") for a in args):
                return True
        return False
    if not _has_opt(extra_args, ["--tensor-parallel-size", "-tp"]):
        cmd += ["--tensor-parallel-size", str(N_GPU)]
    # Add default served-model-name if not provided by user.
    if "--served-model-name" not in extra_args:
        cmd += ["--served-model-name", "predict"]

    # Append any user-provided vLLM args at the end so they can override
    # defaults (port/host/tensor-parallel/etc.).
    cmd += extra_args

    # Note: we use Popen so the web_server decorator can keep the container
    # alive while vLLM runs in the background.
    full_cmd = " ".join(cmd)
    print(f"[serve] Launching vllm: {full_cmd}", flush=True)
    subprocess.Popen(full_cmd, shell=True)
