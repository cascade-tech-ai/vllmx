# Predicted Outputs in vLLM **v1**

The current *predicted outputs* implementation (see `StaticTextWorker`) relies
on **speculative decoding** – a feature that is not yet available in the new
`vllm.v1` pipeline.  Migrating the functionality therefore requires a
different integration strategy.

This document records a quick exploration of v1’s architecture followed by
three concrete approaches that could bring (parts of) the predicted‐output
benefits to v1 without implementing speculative decoding from scratch.

---

## 1  Architectural notes on the v1 execution pipeline

```
┌──────────────────────────────────────────────────────────┐
│  AsyncLLM / API server                                  │  ← FastAPI layer
└──────────────────────────────────────────────────────────┘
              │   (async Request)
              ▼
┌──────────────────────────────────────────────────────────┐
│  vllm.v1.engine.LLMEngine                               │
│  ├─ Processor           – converts user-level request    │
│  │                        → EngineCoreRequest            │
│  ├─ EngineCoreClient    – (optionally multi-proc)        │
│  │   └─ EngineCore      – heavy GPU work                │
│  │       └─ Executor    – model forward loop            │
│  └─ OutputProcessor     – EngineCoreOutput → Response   │
└──────────────────────────────────────────────────────────┘
```

Relevant observations:

* **No speculative hooks** – the *Processor → EngineCore → Executor* chain
  performs a classical **prefill + decode** loop.  Cache hits (prefix caching)
  are already supported but *all* tokens still pass through the model once.

* **SamplingParams** is still the user-visible knob for request level options.
  v1 re-uses the v0 dataclass but validates that certain fields are *unset*
  (e.g. `best_of`).  Importantly, **extra fields are ignored**, so we can add
  a new `predicted_outputs` attribute exactly as we did for v0 without
  breaking existing code.

* **Sequence data lives on the engine side** – Once a request has been
  processed, the per-sequence state is stored in `EngineCoreRequest` /
  `EngineCoreOutput` objects.  These are serialised when we run EngineCore in
  a separate process.

* **OutputProcessor constructs `RequestOutput` objects** – This is where we
  could retrofit acceptance of *pre-validated* tokens without invoking the
  model.


---

## 2  Integration ideas (without speculative decoding)

### Proposal A – *Prefix-cache* the prediction

1.  Concatenate `predicted_token_ids` **in-front of the prompt**.
2.  Mark the entire prediction as *cached* so the first prefill step skips
    the expensive forward pass (exactly what prefix caching already does).
3.  Start decoding **after** the prediction.

Pros
* Zero changes to the deep execution loop – we leverage existing prefix cache.
* Deterministic – the model will *re-compute* the logits for the next token.

Cons
* The predicted tokens are still *materialised* into KV-cache which eats GPU
  memory (though quickly evicted as we decode further).
* We must ensure the tokens are **identical** to what the model itself would
  have produced; otherwise logits for the first *new* token will be wrong.

Implementation touch-points
*   Extend `Processor._build_tokenized_prompt` so that – if
    `sampling_params.predicted_outputs` is present – the prediction is
    prepended and the resulting `num_cached_tokens` is updated.


### Proposal B – *Output short-circuit* in the decode loop

1.  During **each** decode step compare the currently generated prefix with
    the user prediction (logic identical to `StaticTextWorker`).
2.  If the entire prefix matches, directly **append** up to *k* predicted
    tokens to the output buffers **without** calling `Executor.forward()`.
3.  Resume normal decoding once either
    *   the prediction is exhausted, or
    *   the model diverges.

Pros
* Achieves the same computational savings as speculative decoding for the
  *happy path* (exact match) while keeping the algorithm entirely on the CPU
  side – no model fallback path required.

Cons
* Requires intrusive changes inside `LLMEngine._run_decode_iteration` (or the
  `Executor`) to allow *skipping* a forward pass but still update KV-cache /
  sequence stats correctly.
* No probabilistic acceptance test – we assume the prediction is *always*
  correct; wrong predictions would silently corrupt the output.

Implementation touch-points
*   Add a lightweight *PredictionVerifier* class that replicates the line-/​
    prefix-matching logic.
*   Insert an early-return branch in the decode loop that calls the verifier
    and, on success, patches the `SequenceData` of the request.


### Proposal C – *Allowed-token filter* until divergence

1.  Convert `predicted_token_ids` into a sliding window of **allowed token
    IDs** (`SamplingParams.allowed_token_ids`).
2.  After each accepted token advance the window so that **only the next
    predicted token** has probability 1, all others –∞.
3.  Once the prediction is wrong, clear `allowed_token_ids` and continue with
    normal sampling.

Pros
* No core changes; uses existing logits-processor infrastructure (already
  supported in v1).
* Guarantees that the model stays on the predicted path while it is still
  valid.

Cons
* **No compute savings** – the model still evaluates the full vocabulary, we
  only mask logits afterwards.  The technique is therefore useful mainly for
  *correctness* (forced decoding) rather than speed.

Implementation touch-points
*   Provide a helper wrapper that updates the per-sequence
    `allowed_token_ids` field in `SamplingParams` on every step.


---

## 3  Recommended next steps

1. **Prototype A** – the prefix-caching route should be implementable in a few
   lines inside `Processor` and will immediately lower the cost of generating
   *long* fixed headers (e.g. code boilerplate).
2. Evaluate memory impact (cached KV size ~ *prediction_len × d_model*).
3. If additional speed-ups are needed, explore **Proposal B**; its complexity
   is higher but the potential gain approaches speculative decoding
   performance on perfect predictions.

---

*Last updated: 2025-07-05*
