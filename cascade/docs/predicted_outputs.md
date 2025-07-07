### HUMAN SECTION ###

This is the feature document for the feature we would like to implement for vllm support in Gidas.  I will document the 
feature requirements here at the top, and you can use the rest of this document for todo items, notes on progress, etc.

# Feature

We would like to implement predicted outputs in vllm, like they exist in the openai api.  In the openai api, you give
a string of text that you predict to be the output.  While openai only supports exact matches of predicted outputs with
the output, we would like to allow for partial matches using an algorithm similar to diff.  We intend to implement this
using the speculative decoding system as a way to propose future tokens to the llm.  If there is an easier way architecturally
to achieve these goals we are open to it.

# Questions

Before you implement this feature, we will have to answer some questions about exactly how it is going to work.

- How will the proposer work?  I think this is the easiest one, since the ngram_worker.py already works in a fairly similar
  way.
- How many tokens can be proposed at once and what controls that?  Most speculative decoders are only proposing a handful of
  tokens at a time, but we want to propose hundreds, or even thousands?  I think a sweet spot would be somewhere between 128
  and 2048 but it will depend on the use case.
- The speculative proposal system uses workers that derive from ProposerWorkerBase.  How do we create one of these and add it
  to the vllm engine?
- Once the static_text_worker is implemented, it seems like these workers are singletons within the system so they can't
  really hold per-request predicted outputs.  Is there any metadata we can attach somehow?  The worker api takes an
  ExecuteModelRequest, is there some way to set predicted outputs in a table in the worker, and then look them up based
  on information in the ExecuteModelRequest?  Or is there some more elegant way to do it?
    - FOR NOW, if there isn't an easy solution here, we can simply set the predicted text globally in the worker upon creation
      and save the issue of how to make it per-request later.

# Implementation

For now, we would like to avoid modifying the existing vllm code and try to make this work entirely externally.  We should put
our source code in joev/src (next to joev/docs).  We should do all of our work within the joev-predicted-outputs branch.

### AGENT SECTION ###

Use this section for notes, todo lists, etc.  I'll write some of your todo's, but feel free to write your own as well.

# TODO

- Lets start by trying to answer some of the questions above, and come up with an implementation plan for how we will implement
  this feature.

### Notes – 2025-06-26

Below are my first-pass answers to the open questions and a concrete implementation plan.  Please review and let me know what you would like to adjust before I start wiring things into the serving stack.

## 1  How will the proposer work?

We can implement a *StaticTextProposer* that subclasses `NonLLMProposerWorkerBase` because it does **not** need an internal model – it only needs to hand back a prespecified sequence of tokens (the *predicted output*) together with dummy probabilities (we can use a one-hot distribution so that the scorer receives proper shapes).

High-level workflow per forward pass
1. Look at `execute_model_req.num_lookahead_slots` – that is the *k* used by the target model for speculation.
2. For each `SequenceGroup` in the batch:
   •   Read the *predicted tokens* that still remain for this sequence.
   •   Return up to *k* of them (could be fewer if we are at the end of the prediction).
3. Pack the proposals into a regular `SamplerOutput` object so that downstream helpers (`sampler_output_to_torch`, etc.) keep working.

Because the worker does not have KV-cache or parameters it can run entirely on CPU with negligible overhead; we simply create a tensor on the engine device at the end so shapes/devices line up.

## 2  How many tokens can be proposed at once and what controls that?

`num_lookahead_slots` (a.k.a. *k*) is already exposed on `ExecuteModelRequest`.  The scheduler sets it based on `max_proposal_length` in the vLLM config.  We can therefore honour whatever value the user sets.  In practice:

• If the user passes `predicted_text` that is longer than *k*, we will propose *k* tokens this step and the remainder on the next step.
• If the user needs to propose *hundreds/thousands* of tokens we simply make sure the global `max_speculative_len` (default 32 in vLLM) is bumped to e.g. 2048 via the usual `--max-model-len` CLI arg or programmatic `VllmConfig`.

## 3  How do we create one of these workers and add it to the engine?

vLLM exposes the `draft_model` argument on `EngineArgs` / `VllmConfig` to specify the *proposer* side of speculative decoding.  Under the hood the engine instantiates a *worker class* referenced by the draft model’s *qualname* (see `resolve_obj_by_qualname`).

We can therefore register our class under a dotted path (e.g. `joev.src.predicted.static_text_worker.StaticTextWorker`) and pass that string as the `draft_model` parameter.

Example:

```python
engine = LLMEngine(
    model="meta-llama/Llama-3-8B-Instruct-hf",
    draft_model="joev.src.predicted.static_text_worker.StaticTextWorker",
    # … regular EngineArgs …
)
```

## 4  Where do we store the per-request predicted text?

Short term (for the prototype) we will:
1. Extend `SamplingParams` with an optional `predicted_text` field.
2. In `LLM.generate` we pick this field up and attach it to each `SequenceGroupMetadata` via its `.user_data` dict (this dict already exists for arbitrary metadata; it is shipped all the way into the worker).
3. Inside `StaticTextProposer` we read `user_data["predicted_tokens"]` for each sequence id.

This avoids any modifications to core vLLM interfaces and keeps the data local to the request.

Longer term we can upstream a first-class field on `ExecuteModelRequest`, but the above will get us moving quickly.

## Implementation Plan (MVP)

1. **Tokenisation helper**  
   `joev/src/predicted/tokenise.py` – wrapper around the engine’s `TokenizerGroup` that encodes a python string into an `torch.Tensor` of token ids.

2. **StaticTextWorker** (`NonLLMProposerWorkerBase`)  
   • Constructor accepts `predicted_text: Optional[str]` (used only for the global fallback).  
   • `sampler_output(...)` returns the proposal tokens & probs as described in section 1 above.

3. **API surface**  
   • Extend `SamplingParams` (via `joev/src/predicted/patch_sampling_params.py`) adding the optional field.  
   • Monkey-patch `LLM.generate` at import-time so that the predicted text is copied into `user_data`.

4. **Small integration test**  
   Add `tests/test_predicted_outputs.py` which
   1. Spins up a local `LLM` with a tiny model (e.g. `hf-internal-testing/tiny-random-LlamaConfig`).
   2. Sends a request with `predicted_text=" nice to meet you"` following some prompt.
   3. Asserts that the response is returned in a single forward-pass (we can inspect `num_model_calls` on the engine metrics).

5. **Documentation**  
   Update this markdown file once code lands + expose a short snippet in `docs/predicted_outputs.md`.

## Stretch Goals / Outstanding items

• Smarter partial-diff: after a mismatch stop proposing the rest of the prediction but keep *unconsumed remainder* for the next round so we still benefit if the model catches up later.  (Straightforward to add once MVP is merged.)
• Support JSON mode / logit-bias interplay.
• Memory budgeting per request for very long predictions (we may skip proposals when downstream memory is tight).

## NEW QUESTION – tracking progress in the predicted output

**Q:** We don’t propose the entire predicted text at once.  How do we always know *where* we are so that every speculative step begins at the correct offset?

**A:** Use the model’s own `output_token_ids` as the source-of-truth and recompute the offset on-the-fly – no extra state required.

Algorithm (executed inside `StaticTextProposer` each call):
1. Obtain the already-generated output tokens for the sequence:
   ```python
   output_tokens = seq_group.seq_data[seq_id].get_output_len()
   ```
2. Compare this list with the full `predicted_tokens` list and compute the length of the **longest common prefix** (LCP).
3. The next proposal window starts at that LCP offset:
   ```python
   start = lcp_len
   end   = start + k    # k == execute_model_req.num_lookahead_slots
   tokens_to_propose = predicted_tokens[start:end]
   ```
4. If `lcp_len` already equals the length of `predicted_tokens` we are done – nothing more to propose for this request.

Why this works:
• The accepted portion of the prediction is *exactly* the part that already shows up in `output_token_ids`, so LCP gives us the current cursor.  
• When speculative tokens are *rejected*, they never become part of `output_token_ids`, so the LCP remains unchanged and we will re-propose the correct next token in the following iteration.  
• If the model diverges permanently, LCP stops short of the prediction and the proposer will return an empty list, gracefully falling back to normal decoding.

Computational cost is O(k) per step per sequence (where k ≤ max proposal length), which is negligible compared to the model forward passes.

---

Let me know your thoughts – otherwise I will start scaffolding the `StaticTextWorker` and wiring the `SamplingParams` extension.

### Notes – 2025-06-27 (implementation session)

Progress
• `StaticTextWorker` integrated; selectable via `speculative_config.method="static_text"` (no core patches).  
• Worker now pads proposals to **k** tokens (Top-1 proposer requires fixed-length); uses EOS token for padding.  
• End-to-end test in `joev/tests/test_predicted_outputs.py` verifies:  
  – full predicted prefix accepted in the first pass  
  – prints debug summary (token counts, output text, acceptance stats, latency).  
• Sandbox fixes: file-store fallback for torch.distributed; optional GPU-skip hack via `sitecustomize.py`.

### Notes – 2025-06-28 (upstreaming cleanup)

The prototype patches have now been folded into **first-class features**:

1.  `vllm.sampling_params.SamplingParams` received a single field
    `predicted_outputs` that carries an instance of the new helper struct
    `vllm.spec_decode.PredictedOutputParams` (contains either
    `predicted_text` _or_ pre-tokenised `predicted_token_ids`).  This replaces
    the early `extra_args["predicted_text"]` hack.
2.  The temporary monkey-patch in `vllm.spec_decode.predicted_patch` was
    deleted – no runtime monkey-patching is required any more.
3.  `StaticTextWorker` consumes the new field directly and performs
    tokenisation on the worker side when necessary.
4.  Tests were updated to construct `SamplingParams` with the new parameter.

With this refactor in place we can now surface the functionality through the
OpenAI-compatible chat-completion endpoint (next task).

Next steps
1. Replace padding with a masked-prob approach once per-sequence *k* is allowed upstream.  
2. Explore using `LLMEngine.step()` for fine-grained latency measurement.  
3. Add benchmark comparing static-text spec-decode vs. vanilla decoding.  
4. Decide public API for per-request predicted text (extend `SamplingParams` vs. explicit field).

<sub>Updated by AI assistant during pairing session.</sub>

### Human Notes 

Things appear to be working, but we aren't getting the correct debug output.  The first token is generated just by the prefill (before even
calling the proposer).  In the first proposal, we propose 8 tokens ("The tallest mountain in the world is Mount Everest") and then it doesn't
accept the proposed period (30), instead it does comma (28).  We then keep re-proposing token 30 every time.   I think the current 
implementation of 'longest common prefix' is no bueno.

The test itself is also incorrect, it is reporting this as a failure when in fact it is working as intended.

(vllm_local) alvion@perro:~/projects/gidas_vllm/vllm$ cat ~/temp/static_text_worker_1751030417173.log
2025-06-27 09:20:17,182 ERROR joev.src.predicted.static_text_worker: **************** CREATED STATIC TEXT WORKER *******************************
2025-06-27 09:20:20,097 ERROR joev.src.predicted.static_text_worker: **************** INIT DEVICE *******************************
2025-06-27 09:20:24,180 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504]
2025-06-27 09:20:24,182 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 9 tokens: [31469, 6740, 281, 260, 905, 314, 5509, 38921, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,373 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28]
2025-06-27 09:20:25,375 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,445 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807]
2025-06-27 09:20:25,446 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,509 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281]
2025-06-27 09:20:25,510 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,578 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260]
2025-06-27 09:20:25,580 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,643 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509]
2025-06-27 09:20:25,643 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,700 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509, 281]
2025-06-27 09:20:25,701 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,765 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509, 281, 18337]
2025-06-27 09:20:25,766 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,821 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509, 281, 18337, 284]
2025-06-27 09:20:25,823 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,869 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509, 281, 18337, 284, 599]
2025-06-27 09:20:25,870 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2025-06-27 09:20:25,926 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Output Tokens: [504, 31469, 6740, 281, 260, 905, 314, 5509, 38921, 28, 3807, 281, 260, 34509, 281, 18337, 284, 599, 282]
2025-06-27 09:20:25,927 ERROR joev.src.predicted.static_text_worker: [StaticTextWorker] Proposing 1 tokens: [30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]