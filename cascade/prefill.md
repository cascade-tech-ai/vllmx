# Alternate Prefill and Generation in vLLM

## User Request

The user wants to alternate between prefill and generation within the low level API. vLLM currently only supports a single prefill at the beginning of the request. They want to insert additional prompt fragments between generations, similar to a *madlibs* scenario. Prefill should be fast, avoiding releasing and re-allocating KV cache or other request state. The API should look roughly:

```python
engine = AsyncLLMEngine.from_engine_args(engine_args)
sampling_params = SamplingParams(temperature=0.8, max_tokens=10)
prompt = "Once upon a time"
request_id = "req1"
await engine.add_request(request_id, prompt, sampling_params)
outputs = await engine.step()
await engine.prefill(" there was a knight named ")
outputs = await engine.step()
```

After a prefill operation the logits should be available so that the caller may re-sample them later. If needed, AsyncLLMEngine may expose `resample()` to sample from stored logits.

## Architectural Notes

vLLM is organized around `LLMEngine` for synchronous use and `AsyncLLMEngine` for online serving. Requests are represented by `SequenceGroup` objects holding one or more `Sequence` instances. Each `Sequence` contains `SequenceData` with prompt and output token ids. Sequences begin in the `PREFILL` stage and move to `DECODE` once their prompt tokens are computed.

Scheduling is handled by the `Scheduler`. On each `step()` the scheduler selects sequence groups for execution. Prefill requests and decode requests are currently disjoint: a request starts with a prefill and then remains in decode mode until completion. The scheduler allocates KV cache blocks when needed and frees them on completion or preemption.

The request lifecycle today looks like:

1. `add_request` tokenizes the prompt and creates a `SequenceGroup` in `PREFILL` stage.
2. `step()` schedules the new request for prefill; the model executes and appends one token.
3. Subsequent calls to `step()` decode one token at a time until the request finishes.

There is no built‑in method to append additional prompt tokens to a running request. Doing so via a new request would release KV cache and repeat block lookup, which the user wants to avoid.

## Challenges

- **State updates** – After generation starts the sequence stage is `DECODE`. Adding new prompt tokens should mark the sequence back to `PREFILL` and increase its length without clearing existing KV cache.
- **Scheduler interaction** – The scheduler expects prefills only at the beginning. Mixed prefill/ decode within one request may require additional logic so that the scheduler can schedule the extra prefill tokens while other requests might be decoding.
- **Logits reuse** – Prefill currently returns a sampled token. Exposing logits for resampling requires storing the last logits per sequence and an API to sample them on demand.
- **API design** – Need to decide whether the engine exposes `prefill(request_id, text)` or modifies `add_request`. We also need a `resample(request_id)` method if logits are stored.

## Implementation Plan

1. **Extend Sequence/SequenceGroup** – Provide a method like `append_prompt_tokens` that appends token ids to `SequenceData`, resets `num_computed_tokens` for those additions, and sets the stage back to `PREFILL`.
2. **LLMEngine.prefill** – Add a method that tokenizes the provided text and invokes the above sequence update for the given request id. The request remains in the scheduler queues with its KV cache intact.
3. **Scheduler modifications** – Ensure that when a running request enters `PREFILL` again it is treated like a prefill job for a number of tokens equal to the new prompt length. This may involve updating `Sequence.get_num_new_tokens()` and queue placement.
4. **AsyncLLMEngine.prefill** – Expose the asynchronous wrapper that schedules the update from the event loop.
5. **Logit resampling** – Store the last model logits in the sequence state and implement `resample(request_id)` that draws a sample using the original sampling parameters. This allows the caller to regenerate from the same logits after an inserted prefill.

### Design Decisions

- **API shape** – Adding explicit `prefill(request_id, prompt)` keeps backwards compatibility and makes the intent clear.
- **Scheduler complexity** – Reusing the existing prefill mechanism by resetting the sequence stage avoids large refactoring. The scheduler will treat the request as if it just started a new prompt chunk.
- **Logits storage** – Keeping the logits within the request state (possibly attached to `Sequence`) allows resampling without extra cache management.

Further work will require touching multiple components—tokenization, sequence data handling, scheduler rules, and the async engine—to enable efficient alternating prefill and generation without releasing KV cache.
