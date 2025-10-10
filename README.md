<!-- markdownlint-disable MD001 MD041 -->
<h1 style="font-family: 'JetBrains Mono', 'Fira Code', 'Source Code Pro', 'IBM Plex Mono', 'Courier New', monospace; color: #24dcff; letter-spacing: 0.18em; text-transform: uppercase; text-shadow: 0 0 12px rgba(36, 220, 255, 0.6);">CASCADE TECHNOLOGIES</h1>

# Welcome to vLLMX (Cascade Technologies Fork)

This is a fork of the very excellent <a href="https://github.com/vllm-project/vllm" target="_blank" rel="noreferrer">VLLM</a> project to hold experimental features by Cascade Technologies. All work that we don't intend to upstream (demos, documents, etc.) lives in the `./cascade` folder.

## Usage

The easiest way is just to use our vLLM-compatible Docker image: [`docker.io/alvion427/vllmx-openai:v0.11.0`](https://hub.docker.com/r/alvion427/vllmx-openai).

```bash
docker pull docker.io/alvion427/vllmx-openai:v0.11.0
```

Otherwise, follow the upstream vLLM build instructions to create wheels, etc.

> **Note**
> To use `VLLM_USE_PRECOMPILED=1` you will also need to specify the wheel location.

```bash
export VLLM_COMMIT=b761df963c2032144468a99bcb39a11e73e16ca4  # v0.11.0 merge-base
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/b761df963c2032144468a99bcb39a11e73e16ca4/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl" \
uv pip install -e .
```

## Features

- [Predicted Outputs](https://cascadetech.ai/blog/vllm-predicted-outputs/)
