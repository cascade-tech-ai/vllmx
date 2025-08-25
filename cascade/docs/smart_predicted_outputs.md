# Smart Predicted Outputs

Smart Predicted Outputs is OpenAI’s “predicted outputs” concept with smarter alignment. It is similar to speculative decoding, but the prediction side is trivial (static text), much cheaper to run, and often more accurate on structured text. You attach a predicted continuation; we align at line boundaries and can re‑anchor after divergence so the prediction keeps helping.

## Why

Modern LLM applications often know part of what the model will output next: boilerplate code, templated headers, structured responses, or a continuation the application already computed. If we can hint those tokens to the model and skip work when they match, we get lower latency and cost without changing model weights.

Two needs emerge:
- Fast-path exact or near-exact continuations (accept predicted tokens in bulk).
- Recover alignment after divergences (e.g., a missing stanza, a refactor), resuming useful prediction further down the text.

## What

OpenAI’s API already supports sending a predicted continuation; it will fast‑path exact matches but abandons the rest of the prediction at the first divergence. This project brings that idea to vLLM and extends it so predictions remain useful even after a mismatch.

Concretely, we provide:
- Exact‑match fast path using static‑text speculative decoding to bulk‑accept predicted tokens when they match.
- Resilient, line‑aware alignment that can re‑anchor later in the prediction after local edits (missing stanza, variable rename, helper extraction) and resume proposing.
- Per‑request predictions (text or token IDs) and optional per‑iteration YAML logs for deep diagnostics.

Primary touch points users interact with:
- `SamplingParams.predicted_outputs` to attach the prediction to a request.
- `SpeculativeConfig.method = "static_text"` to enable the static proposer.
- OpenAI server support via the `prediction` field, mapped to the above.

## Implementation in VLLM

At a glance:
1. You attach a predicted continuation to each request.
2. With speculative decoding enabled (`method: "static_text"`), vLLM’s proposer attempts to align the generated context with your prediction at line boundaries.
3. If aligned, it proposes up to `k = num_speculative_tokens` predicted tokens. The target model either accepts the whole prefix in one go or rejects at the first mismatch, advancing by the standard “accepted + 1” rule.
4. On divergence, the proposer abstains until a future context suffix re‑anchors to your prediction, then resumes proposing.

Alignment is line‑based: newline tokens partition both the generated context and your prediction; matching is performed over line tuples using either RapidFuzz LCS (fast path) or `difflib.SequenceMatcher` (fallback). This balances robustness and performance without requiring a draft model.

## How To Use

### Python API (vLLM)

```python
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.spec_decode.predicted_output_params import PredictedOutputParams

llm = LLM(
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    speculative_config={
        "method": "static_text",            # enable static-text proposer
        "num_speculative_tokens": 128,       # lookahead window k
        "disable_mqa_scorer": True,          # typical for non-model proposers
    },
    trust_remote_code=True,
)

sampling = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    predicted_outputs=PredictedOutputParams(
        predicted_text="The tallest mountain in the world is Mount Everest."
    ),
)

out = llm.generate([
    "<|im_start|>user\nWhat is the tallest mountain?<|im_end|>\n<|im_start|>assistant\n"
], [sampling], use_tqdm=False)
print(out[0].outputs[0].text)
```

Notes:
- Either `predicted_text` or `predicted_token_ids` may be set. If only text is provided, vLLM tokenizes it once per request (with CR/LF normalized to `\n` first) and caches the IDs.
- Increase `num_speculative_tokens` to allow longer bursts of acceptance when the prediction matches.
- Works with greedy or sampling; determinism helps showcase acceptance.

### OpenAI-Compatible Server

Start the server with static-text speculation enabled:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model RedHatAI/Llama-3.2-1B-Instruct-FP8 \
  --gpu-memory-utilization 0.8 \
  --served-model-name llama-1b \
  --speculative-config '{"method":"static_text","num_speculative_tokens":64}'
```

Clients can attach a prediction using the `prediction` field:

```json
{
  "model": "llama-1b",
  "messages": [ ... ],
  "prediction": { "type": "content", "content": "... predicted continuation ..." },
  "max_completion_tokens": 128,
  "temperature": 0
}
```

The server maps `prediction.content` to `SamplingParams.predicted_outputs.predicted_text` for the request.

### Tuning Tips

- Lookahead: `num_speculative_tokens` controls the proposal window size `k`. Larger windows yield bigger speedups on perfect matches.
- Newlines: For best line alignment, ensure the prediction uses `\n`. The engine normalizes `\r\n`/`\r` to `\n` before tokenization.
- Token IDs: If you already have a tokenizer, send `predicted_token_ids` to avoid per-request tokenization overhead.
- Optional RapidFuzz: Install `rapidfuzz` to use a faster LCS-based aligner; otherwise the system falls back to `difflib`.

## Smart Prediction (Line-Aware)

Beyond exact-prefix acceptance, we prototype line-based predictors that can re‑anchor after divergences. This helps when the prediction omits or edits blocks (e.g., missing stanza, variable rename, helper extraction).

Prototype components (offline, under `cascade/src/smart_prediction/`):
- Algorithms: naive sequential baseline, greedy suffix, line-diff, and streaming Myers variants.
- Simulator: deterministic loop that mimics speculative decoding’s accept-then-advance rule.
- Examples and tests: poems and small code snippets demonstrating recovery after edits.

These prototypes informed the production line-level alignment used by the `static_text` proposer.

## Implementation Details

### Public API Additions

- `vllm.spec_decode.predicted_output_params.PredictedOutputParams`
  - `predicted_text: Optional[str]`
  - `predicted_token_ids: Optional[list[int]>`
  - `has_prediction()` convenience method.

- `vllm.sampling_params.SamplingParams`
  - New field: `predicted_outputs: Optional[PredictedOutputParams]`.

- OpenAI protocol (`vllm/entrypoints/openai/protocol.py`)
  - New request field: `prediction: { type: "content", content: str }`.
  - Mapped to `SamplingParams.predicted_outputs`.

### v1 Execution Path

- `vllm/v1/engine/processor.py`
  - If `predicted_text` is present, normalize line endings and tokenize once, storing `predicted_token_ids` into the request’s `SamplingParams`.

- `SpeculativeConfig` (`vllm/config.py`)
  - Adds `"static_text"` to `method`. For static-text, v1 reuses the target model config for the “draft” side; no draft model is loaded.

- GPU runner (`vllm/v1/worker/gpu_model_runner.py`)
  - Branches on `method == "static_text"` and delegates draft generation to `StaticTextProposer`.
  - Builds the context as “previous accepted output” + “newly sampled token(s)” before calling the proposer.
  - Optional structured YAML debug logging per iteration (see Debugging).

- Proposer (`vllm/v1/spec_decode/static_text_proposer.py`)
  - Stateless w.r.t. models; pure Python/NumPy on CPU.
  - Per-request state: predicted token list; newline token set; prediction split into line-token tuples; incremental context tracking.
  - Newline detection: scans the tokenizer’s full vocab once to build a global set of token IDs whose decoded form contains `"\n"`.
  - Alignment: finds the cursor line using RapidFuzz LCS if available, else `difflib`. Verifies that the current line prefix matches before proposing.
  - Returns up to `k` predicted tokens (may be empty when unaligned or exhausted).

### v0 Compatibility

- `vllm/spec_decode/static_text_worker.py`
  - Legacy v0 worker delegating line-splitting and alignment to the v1 helpers for a single source of truth.
  - Emits one-hot distributions and pads proposals to the declared `sample_len` (`k`) as Top‑1 proposer requires fixed length.

## Debugging & Observability

When `DEBUG_PREDICTED_OUTPUTS` is enabled in the proposer, each decode iteration writes a structured YAML record:

- File path: `${VLLMX_HOME:-~/.vllmx}/log/<request_id>.yaml`
- Contents include:
  - Entire predicted text split into lines (tokens and text).
  - Per-iteration: new output tokens, newly completed context lines, current-line prefix, proposed tokens/text, and measured duration.

This helps diagnose missed alignments, slow paths, and iteration-by-iteration acceptance. The flag is off by default and intended for development.

## Limits & Notes

- The proposer only uses per-request predictions; there is no global prediction.
- Alignment is line-based; heavily reformatted predictions with different newline structure may reduce effectiveness.
- Wrong proposals are naturally bounded by the acceptance logic (reject-at-first-mismatch + advance by one); they cost iteration time but do not corrupt outputs.
- For best results, keep temperature low when you expect long exact matches.

## Tests & Examples

- `cascade/tests/test_predicted_outputs.py`: end-to-end test verifying that a perfect predicted prefix is accepted in the first pass.
- `cascade/tests/test_smart_prediction.py` and `cascade/tests/test_smart_prediction_snake_game.py`: offline simulation tests covering missing stanza and code-edit scenarios; also includes streaming Myers variants.

## Quick Reference

- Enable speculative decoding with static text: set `speculative_config = {"method": "static_text", "num_speculative_tokens": K}`.
- Provide per-request prediction via `SamplingParams.predicted_outputs` (text or token IDs) or the OpenAI `prediction` field.
- Install `rapidfuzz` to speed up alignment; otherwise the code falls back to `difflib`.
- Use the optional YAML debug logging for deep dives.

---

Last updated: 2025-08-25
