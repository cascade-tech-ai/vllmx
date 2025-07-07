"""Light integration test that runs a *real* model offline.

Requirements:
1. The model `HuggingFaceTB/SmolLM2-360M-Instruct` must already exist in the
   local Hugging Face cache – i.e. was downloaded once on this machine.
2. No internet access: we set the environment variables Hugging Face honours
   (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`) so the loader never attempts to
   reach the hub.  If the model is **not** cached, the test is skipped.

The test verifies only that our patched draft worker stops proposing after the
first divergence.  We do *not* assert on latency or exact text because the
360 M model’s outputs can vary across hardware / kernels.
"""

import os
from pathlib import Path
from typing import Set

import torch

# ---------------------------------------------------------------------------
# Offline / CPU environment configuration BEFORE importing transformers/vLLM
# ---------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# If CI has GPUs attached we still allow CUDA; the tiny 360 M model is fine.
# But we ensure the test can run on CPU‐only machines by clearing CUDA devices
# if the user sets VLLM_CPU_ONLY=1.
if os.getenv("VLLM_CPU_ONLY") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# ---------------------------------------------------------------------------
# Patch imports (must precede vLLM imports to monkey-patch drafter)
# ---------------------------------------------------------------------------

import sys  # noqa: E402  (import after env vars set)

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import predicted  # noqa: F401,E402  side-effect patches

# vLLM imports – safe now
from vllm.entrypoints.llm import LLM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.config import SpeculativeConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _has_model_cached(model_id: str) -> bool:
    """Return True if *any* revision of *model_id* is present in the HF cache."""

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # fast but crude: model subdirs are of the form models--<id>--<hash>
    prefix = f"models--{model_id.replace('/', '--')}"
    return any(p.name.startswith(prefix) for p in cache_dir.iterdir()) if cache_dir.exists() else False


# ---------------------------------------------------------------------------
# The actual test
# ---------------------------------------------------------------------------


def test_offline_generation_with_predicted_tokens():
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

    if not _has_model_cached(model_id):
        import pytest  # local import so that pytest is only required when running this test
        pytest.skip(f"{model_id} not present in local HF cache – skipping offline test")

    spec_cfg = SpeculativeConfig(
        method="ngram",
        prompt_lookup_max=1,
        prompt_lookup_min=1,
        num_speculative_tokens=32,
    )

    llm = LLM(
        model=model_id,
        # enforce eager on CPU works fine; on GPU the model is small anyway
        enforce_eager=True,
        speculative_config=spec_cfg,
        gpu_memory_utilization=0.8,
    )

    sampling_params = SamplingParams(
        max_tokens=16,
        extra_args={"predicted_text": " Paris."},
    )

    outputs = llm.generate("The capital of France is", sampling_params=sampling_params)

    txt = outputs[0].text
    assert len(txt) > 0  # Ensure we got some output

    # Grab internal flag to confirm speculation was disabled after divergence.
    extra = sampling_params.extra_args or {}
    assert extra.get("disable_predicted", False) is True
