# Preserving and Continuing Sequence State in vLLM

This document examines how vLLM internally represents request state—including the
token sequences, embeddings, and KV cache mappings—via the `Sequence`, `SequenceData`,
and `SequenceGroup` classes, and explores options for preserving that state across
invocations to the `LLMEngine` in order to continue from a previous request without
re-tokenizing or rebuilding the KV cache.

## 1. Core Sequence Abstractions

vLLM encodes each request into one or more `Sequence` objects, which wrap a
`SequenceData` instance containing prompt and output token IDs, embeddings,
and internal counters (e.g., number of computed tokens, cumulative logprobs).
Multiple sequences (e.g., for speculative decoding drafts) are grouped into a
`SequenceGroup`, which tracks shared sampling parameters, request metrics,
and scheduling state.

```python
from vllm.sequence import Sequence, SequenceData, SequenceGroup, SequenceStatus
```
【F:vllm/sequence.py†L570-L650】【F:vllm/sequence.py†L820-L880】

These classes expose methods to:
- Append generated tokens (`Sequence.append_token_id` / `SequenceData.append_token_id`).
- Fork or deep-copy a sequence (`Sequence.fork`).
- Serialize incremental deltas (`SequenceData.get_delta_and_reset`).
- Reset state for recomputation (`Sequence.reset_state_for_recompute`).

## 2. Engine Lifecycle and Request Pooling

The `LLMEngine` manages a pool of active `SequenceGroup`s.  Each new generate/chat
invocation ultimately calls `engine.add_request(...)`, which constructs fresh
`Sequence` / `SequenceGroup` instances from the given prompt and adds them to the
scheduler.

```python
def add_request(
    self,
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    tokenization_kwargs: Optional[dict] = None,
) -> None:
    processed = self.input_preprocessor.preprocess(
        prompt, tokenization_kwargs=tokenization_kwargs)
    self._add_processed_request(
        request_id=request_id,
        processed_inputs=processed,
        params=params,
        arrival_time=arrival_time,
        ...
    )
```
【F:vllm/engine/llm_engine.py†L680-L745】

When all sequences in a group finish (via EOS or stop criteria), the engine releases
their KV cache blocks back to the block manager, potentially evicting them if under
memory pressure.

## 3. Prefix Caching vs. Full Continuation

vLLM’s existing *prefix caching* lets users avoid re-prefilling shared prefixes across
distinct requests by caching and reusing previously computed KV cache blocks:

```markdown
Automatic prefix caching caches the KV state for repeated prompt prefixes;
use `--enable-prefix-caching` to activate.
```
【F:docs/features/automatic_prefix_caching.md†L1-L40】

However, prefix caching still requires re-tokenizing the entire prompt and only
improves the prefill stage rather than preserving a full interactive state that
can seamlessly resume both prompt and output context.

## 4. Continuing Multi-Turn Conversations

At the Python API level, the `LLM.chat()` method retokenizes all prior messages
on each call:

```python
conversation, mm_data = parse_chat_messages(...)
prompt = TokensPrompt(tokenizer.encode(apply_hf_chat_template(...)))
return self.generate([prompt], sampling_params=[...])
```
【F:vllm/entrypoints/llm.py†L728-L780】

There is no built-in way to call `chat()` repeatedly on the same engine instance
without re-tokenization.  While prefix caching can speed up the repeated pipeline,
it does **not** avoid looking up and patching the block table for the full prompt
every turn.

## 5. Proposed Continuation API Sketch

To truly preserve and resume a sequence—maintaining its KV cache blocks, token
embeddings, and scheduling state—a small engine API extension could be added:

```python
class LLMEngine:
    def save_sequence_group(self, request_id: str) -> SequenceGroup:
        """Snapshot internal state to resume later."""
    def load_sequence_group(
        self, seq_group: SequenceGroup, new_request_id: str
    ) -> None:
        """Register a previously saved SequenceGroup under a fresh ID."""
    def continue_generation(
        self, request_id: str, sampling_params: SamplingParams
    ) -> List[int]:
        """Resume decoding on an existing SequenceGroup."""

# Client-side usage:
engine = LLMEngine(...)
# Initial prompt run:
engine.add_request("r1", "Hello, how are you?", SamplingParams(max_tokens=10))
outputs = engine.run()       # or step()/run

# Snapshot state:
seq_group = engine.save_sequence_group("r1")

# Later resume with a follow-up:
engine.load_sequence_group(seq_group, "r2")
engine.continue_generation("r2", SamplingParams(max_tokens=5))
new_tokens = engine.step()
```

This API would:
- Prevent eviction of the old KV cache blocks (as the `SequenceGroup` object remains live).
- Bypass re-tokenization if `SequenceData.prompt_token_ids` and embeds are reused.
- Eliminate the need for block table lookups for already-loaded blocks.

## 6. Next Steps

- **Feasibility:** The engine already retains `SequenceGroup` state until freed.
- **Implementation:** Expose simple `save_`/`load_` hooks in `vllm/engine/llm_engine.py`,
  leveraging deep-copy (`Sequence.fork`) or serialization of `SequenceData`.
- **Documentation:** Add examples to `joev/sequence_continuation.md` and update API docs.

By enriching the `LLMEngine` with a continuation API, we can support true
stateful LLM sessions—perfect for multi-turn chat, tutoring systems, and
interactive pipelines—without redoing expensive prefill or block cache work.