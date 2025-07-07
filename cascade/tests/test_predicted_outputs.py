"""Integration test for *predicted outputs* via StaticTextWorker.

The test spins up a local ``LLM`` using the *SmolLM2-360M-Instruct* model that
is assumed to be present in the HuggingFace cache.  Network access is disabled
via the ``HF_HUB_OFFLINE`` environment variable to prevent accidental
downloads during CI.
"""

from __future__ import annotations

import os
# -----------------------------------------------------------------------------
# Runtime requirements ---------------------------------------------------------
# -----------------------------------------------------------------------------

import shutil

import torch

import pytest

# -----------------------------------------------------------------------------
# Configuration – disable any outbound network traffic to HF.
# -----------------------------------------------------------------------------

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


# Skip test when a CUDA device is not usable by PyTorch (some CI containers
# expose `nvidia-smi` but block `cudaGetDeviceCount`, which raises a runtime
# error instead of simply returning 0).
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reports CUDA unavailable – speculative decoding requires GPU",
)
def test_static_text_worker_generates_with_predicted(monkeypatch):
    # Import *after* setting env vars so that HF picks them up.
    # (No monkey-patch required – predicted outputs are now a first-class
    # field on SamplingParams.)
    from vllm.entrypoints.llm import LLM
    from vllm.sampling_params import SamplingParams
    from vllm.spec_decode.predicted_output_params import PredictedOutputParams

    # The text we expect / predict the model to emit first.
    predicted_text = "The tallest mountain in the world is Mount Everest."

    # We know the exact results of what this model will output
    prompt = """<|im_start|>user
What is the tallest mountain in the world?<|im_end|>
<|im_start|>assistant
"""

    sampling_params = SamplingParams(
        temperature=0.0,  # ensure deterministic continuation
        max_tokens=20,    # allow model to produce after the prediction
        predicted_outputs=PredictedOutputParams(predicted_text=predicted_text),
    )

    # Build the LLM in *offline* mode using our StaticTextWorker as the draft.
    llm = LLM(
        model=MODEL_NAME,
        # Use our new static_text speculative method.
        speculative_config={
            "method": "static_text",
            # Provide a large *k* so the entire prediction fits into the first
            # proposal window.  This lets us verify that the request finishes
            # in a *single* model forward pass when the prediction matches.
            "num_speculative_tokens": 128,
            "disable_mqa_scorer": True,
        },
        trust_remote_code=True,
        enforce_eager=True,  # avoid cuda-graph for quicker init in CI
    )

    outputs = llm.generate([prompt], [sampling_params], use_tqdm=False)

    # Basic sanity checks ------------------------------------------------------
    assert len(outputs) == 1
    out = outputs[0]

    # There should be exactly one completion (n=1).
    assert len(out.outputs) == 1

    # Spec-decode metrics should be present when the pipeline is engaged.
    assert out.metrics is not None
    assert out.metrics.spec_token_acceptance_counts is not None

    # ------------------------------------------------------------------
    # Human-readable diagnostics (visible with -s) ----------------------
    # ------------------------------------------------------------------

    from time import perf_counter

    total_tokens = len(out.outputs[0].token_ids)
    text_out = out.outputs[0].text

    # Capture basic timing info from metrics if present.
    if out.metrics.first_token_time and out.metrics.arrival_time:
        latency_ms = (out.metrics.first_token_time - out.metrics.arrival_time) * 1000
    else:
        latency_ms = None

    print("\n========== DEBUG SUMMARY ==========")
    print(f"Generated tokens: {total_tokens}")
    print(f"Output text    : {text_out!r}")
    print(f"Spec-decode    : {out.metrics.spec_token_acceptance_counts}")
    if latency_ms is not None:
        print(f"Time to first token: {latency_ms:.1f} ms")
    print("===================================\n")

    # --- Additional validations -------------------------------------------
    # We expect the entire prediction to be accepted in the *first* (and only)
    # speculative step so the model performs exactly one forward pass after
    # the prompt.

    acceptance_counts = out.metrics.spec_token_acceptance_counts

    # The first decode step should accept at least the entire prediction.  All
    # subsequent steps still include the *target-model* token, therefore they
    # will register exactly one accepted token each.  We only care that no
    # *additional* speculative tokens are accepted after step-0.

    tokenizer = llm.get_tokenizer()

    accepted_tokens = acceptance_counts[0]

    predicted_len = len(tokenizer.encode(predicted_text, add_special_tokens=False))

    assert accepted_tokens >= predicted_len, (
        "Speculative pipeline did not accept the full predicted prefix.")

    assert all(v == 1 for v in acceptance_counts[1:predicted_len-1]), (
        f"Unexpected number of accepted tokens after the first decoding step: {acceptance_counts[1:predicted_len-1]}")

    # Convert token-ids to strings so we can inspect them easily.
    gen_token_ids = list(out.outputs[0].token_ids)
    gen_tokens = [tokenizer.decode([tid], skip_special_tokens=False)
                  for tid in gen_token_ids]

    # Print helpful debug info (visible with `pytest -s`).
    print("\n[DEBUG] generated token count:", len(gen_token_ids))
    print("[DEBUG] accepted (first pass):", accepted_tokens)

    pred_token_ids = tokenizer.encode(predicted_text, add_special_tokens=False)
    pred_tokens = [tokenizer.decode([tid], skip_special_tokens=False)
                   for tid in pred_token_ids]

    print("[DEBUG] predicted token count:", len(pred_token_ids))

    # Show side-by-side comparison of generated vs predicted for the first
    # min(len(pred), len(gen)) tokens.
    compare_n = min(len(pred_token_ids), len(gen_token_ids))
    pairs = []
    for i in range(compare_n):
        pairs.append((i, gen_token_ids[i], gen_tokens[i], pred_token_ids[i], pred_tokens[i]))

    from pprint import pprint
    print("[DEBUG] first tokens comparison (idx, gen_id, gen_str, pred_id, pred_str):")
    pprint(pairs)


    # The number of tokens accepted in the first pass should cover the entire
    # prediction (and possibly more if the model continued further within the
    # same window).
    predicted_len = len(tokenizer.encode(predicted_text, add_special_tokens=False))
    assert accepted_tokens >= predicted_len, (
        "Speculative pipeline did not accept the full predicted prefix.")
