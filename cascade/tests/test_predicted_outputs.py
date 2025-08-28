"""Slow end-to-end tests for Smart Predicted Outputs (static-text proposer).

These tests intentionally spin up a real vLLM engine and exercise the
`predicted_outputs` API with and without a supplied prediction.

Notes:
- Model path is taken from env var `VLLMX_TEST_MODEL`, defaulting to
  'HuggingFaceTB/SmolLM2-360M-Instruct'.
- We keep a single lazy LLM instance across tests with static-text prediction
  enabled so the engine initialization cost is paid once.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pytest


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_ENV = "VLLMX_TEST_MODEL"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


_LLM = None  # lazy global LLM instance


def get_llm():
    """Return a cached LLM with static-text prediction enabled.

    Ensures we use the V1 engine so metrics are available via `get_metrics()`
    and per-request `RequestOutput.metrics` fields are populated.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    # Use V0 engine to access per-request metrics in RequestOutput.metrics
    os.environ.setdefault("VLLM_USE_V1", "0")

    # Local import so environments without vLLM installed can still import the
    # test module (collection time) without crashing.
    from vllm.entrypoints.llm import LLM

    model_id = os.environ.get(MODEL_ENV, DEFAULT_MODEL)

    # Allow overriding GPU util via env for CI/local tuning
    gpu_util = float(os.getenv("VLLMX_GPU_UTIL", "0.25"))

    _LLM = LLM(
        model=model_id,
        speculative_config={
            "method": "static_text",
            "num_speculative_tokens": 128,
            "disable_mqa_scorer": True,
        },
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=gpu_util,
        max_model_len=1024,
        disable_log_stats=False,
    )
    return _LLM


def _token_len(text: str) -> int:
    llm = get_llm()
    tokenizer = llm.get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _build_smol_chat_prompt(user_text: str) -> str:
    # Matches docs example for SmolLM2 chat template
    return (
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_chat(
    user_prompt: str,
    *,
    max_tokens: int,
    predicted_text: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Generate up to `max_tokens` and return (text, prediction_stats).

    - Applies chat template automatically via LLM.chat().
    - If `predicted_text` is provided, attaches it as the per-request
      prediction using SamplingParams.predicted_outputs.
    - Returns the assistant text and a dict of stats derived from
      RequestOutput.metrics (per-request, not aggregated).
    """
    from vllm.sampling_params import SamplingParams
    from vllm.spec_decode.predicted_output_params import PredictedOutputParams

    llm = get_llm()
    use_v1 = os.getenv("VLLM_USE_V1", "0") == "1"

    # Interpret non-positive as "until end"; use a generous cap to avoid
    # unbounded generation in CI while behaving intuitively for tests.
    capped_max = max_tokens if max_tokens > 0 else 512

    sampling_kwargs = dict(temperature=0.0, max_tokens=capped_max)
    if predicted_text is not None:
        # Provide pre-tokenized prediction to ensure the V1 worker has
        # predicted_token_ids available without re-tokenizing inside GPU proc.
        tokenizer = llm.get_tokenizer()
        normalized_text = predicted_text.replace("\r\n", "\n").replace("\r", "\n")
        pred_ids = tokenizer.encode(normalized_text, add_special_tokens=False)
        sampling_kwargs["predicted_outputs"] = PredictedOutputParams(
            predicted_text=predicted_text, predicted_token_ids=pred_ids)

    sampling = SamplingParams(**sampling_kwargs)

    messages = [
        {"role": "user", "content": user_prompt},
    ]

    # Pre-metrics snapshot for V1 (prometheus-based)
    pre_metrics = None
    if use_v1 and hasattr(llm, "get_metrics"):
        try:
            pre_metrics = llm.get_metrics()
        except Exception:
            pre_metrics = None

    model_id = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    use_raw_chat = "SmolLM2-360M-Instruct" in model_id

    if use_raw_chat:
        # Build chat prompt explicitly to avoid template mismatches
        prompt_str = _build_smol_chat_prompt(user_prompt)
        outputs = llm.generate([prompt_str], [sampling], use_tqdm=False)
    else:
        outputs = llm.chat(messages,
                           sampling_params=sampling,
                           add_generation_prompt=True,
                           use_tqdm=False)

    assert len(outputs) == 1
    req = outputs[0]
    assert len(req.outputs) >= 1
    out = req.outputs[0]

    # Per-request speculative acceptance counts (may be None if speculation
    # was not engaged, or 0-length if disabled internally for the request).
    acceptance_counts = None
    predicted_accepted = 0

    if not use_v1 and req.metrics is not None:
        acceptance_counts = req.metrics.spec_token_acceptance_counts
        if acceptance_counts:
            predicted_accepted = sum(acceptance_counts[1:])
    elif use_v1 and hasattr(llm, "get_metrics"):
        try:
            post_metrics = llm.get_metrics()
        except Exception:
            post_metrics = None
        # Compute delta of accepted tokens across the request window.
        if pre_metrics is not None and post_metrics is not None:
            def total_counter(metrics, name):
                from vllm.v1.metrics.reader import Counter
                return sum(m.value for m in metrics if isinstance(m, Counter)
                           and m.name == name)

            accepted_before = total_counter(pre_metrics,
                                            "vllm:spec_decode_num_accepted_tokens")
            accepted_after = total_counter(post_metrics,
                                           "vllm:spec_decode_num_accepted_tokens")
            drafts_before = total_counter(pre_metrics,
                                          "vllm:spec_decode_num_drafts")
            drafts_after = total_counter(post_metrics,
                                         "vllm:spec_decode_num_drafts")
            draft_tokens_before = total_counter(pre_metrics,
                                               "vllm:spec_decode_num_draft_tokens")
            draft_tokens_after = total_counter(post_metrics,
                                              "vllm:spec_decode_num_draft_tokens")
            predicted_accepted = max(0, accepted_after - accepted_before)
            print(
                "[DEBUG V1 metrics] drafts:", drafts_after - drafts_before,
                "draft_tokens:", draft_tokens_after - draft_tokens_before,
                "accepted:", predicted_accepted,
            )

    stats = {
        "acceptance_per_pos": acceptance_counts or [],
        "predicted_accepted_tokens": predicted_accepted,
        "generated_tokens": len(out.token_ids),
    }

    return out.text, stats


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


HAIKU = (
    "Autumn moonlight—\n"
    "a worm digs silently\n"
    "into the chestnut."
)


def test_echo_haiku_no_prediction():
    # Ask the model to echo the haiku. We do not attach any prediction.
    max_tokens = _token_len(HAIKU)
    prompt = (
        "Repeat back the following haiku verbatim.\n"
        "Do not add any commentary or extra text.\n\n"
        f"{HAIKU}"
    )

    _, stats = generate_chat(prompt, max_tokens=max_tokens, predicted_text=None)

    assert stats["predicted_accepted_tokens"] == 0


def test_echo_haiku_with_prediction():
    # First, get the model's exact deterministic output for this prompt.
    base_prompt = (
        "Repeat back the following haiku verbatim.\n"
        "Do not add any commentary or extra text.\n\n"
        f"{HAIKU}"
    )

    observed_text, _ = generate_chat(base_prompt, max_tokens=-1,
                                      predicted_text=None)

    # Now, attach that exact output as the prediction and request exactly the
    # same number of tokens to be generated.
    predicted_text = observed_text
    predicted_len = _token_len(predicted_text)

    _, stats = generate_chat(base_prompt,
                             max_tokens=predicted_len,
                             predicted_text=predicted_text)

    # All generated tokens should be accepted speculative tokens.
    assert stats["predicted_accepted_tokens"] == predicted_len
