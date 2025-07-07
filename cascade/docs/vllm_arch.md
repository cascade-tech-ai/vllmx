# vLLM Inference Architecture Deep Dive

This document provides a focused overview of vLLM's inference pipeline, covering:
- Request entrypoints and engine lifecycle
- Scheduling strategies for interleaving prefill and token generation
- Batching multiple requests into a single model pass
- KV cache management via paged attention
- Attention API design and FlashInfer integration

> **Note:** We restrict discussion to the core inference components (LLM engine, scheduler, model runner, attention backends) and omit web/API layers.

## 1. Entrypoints

vLLM offers two main entrypoints for inference:

- **Offline Python interface:** the `LLM` class in `vllm/entrypoints/llm.py` wraps the core engine for batch inference.
- **Online async server:** the `AsyncLLMEngine` powers the OpenAI-compatible API server.

See the high-level architecture diagram and entrypoints in the design docs:
```markdown
docs/design/arch_overview.md:
  # Architecture Overview
  ## Entrypoints
  ![Entrypoints Diagram](../assets/design/arch_overview/entrypoints.excalidraw.png)
```
【F:docs/design/arch_overview.md†L1-L13】【F:docs/design/arch_overview.md†L21-L35】

## 2. Core LLM Engine

The heart of vLLM is the `LLMEngine` (and its async wrapper), responsible for tokenization, scheduling, model execution, and output processing:

```markdown
## LLM Engine
- Input Processing: tokenization
- Scheduling: deciding prefill vs decode workloads
- Model Execution: distributed inference across GPUs
- Output Processing: decoding token IDs to text
```
【F:docs/design/arch_overview.md†L79-L102】

### Engine Step Loop

Each call to `LLMEngine.step()` runs one decoding iteration in three phases:

```python
def step(self) -> List[RequestOutput]:
    """Performs one decoding iteration and returns newly generated results.

    Details:
    - Step 1: Schedule sequences and KV cache blocks (swap/copy).
    - Step 2: Invoke the model executor for a batched forward pass.
    - Step 3: Process model outputs (decode, update state, free finished sequences).
    """
```
【F:vllm/engine/llm_engine.py†L1212-L1235】

## 3. Scheduling & Chunked Prefill

vLLM uses iteration-level scheduling to interleave compute-bound prefill and memory-bound decode operations. By default, pending decodes are prioritized and large prefills are automatically chunked:

```markdown
## Chunked Prefill
- Always enabled in vLLM V1. Decodes are prioritized; prefills that don't fit the token budget are split into chunks.
```
【F:docs/configuration/optimization.md†L23-L61】

Scheduler outputs drive KV cache actions and batched workloads:

```python
SchedulerOutputs(
    scheduled_seq_groups: List[ScheduledSequenceGroup],
    blocks_to_swap_in: List[Tuple[int,int]],
    blocks_to_swap_out: List[Tuple[int,int]],
    blocks_to_copy:    List[Tuple[int,int]],
    num_lookahead_slots: int,
    running_queue_size: int,
    finished_requests_ids: List[str],
)
```
【F:vllm/core/scheduler.py†L184-L209】

## 4. Batching Into a Model Pass

After scheduling, `LLMEngine` packages the work into an `ExecuteModelRequest` and dispatches to the model executor:

```python
execute_model_req = ExecuteModelRequest(
    seq_group_metadata_list=seq_group_metadata_list,
    blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
    blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
    blocks_to_copy=scheduler_outputs.blocks_to_copy,
    num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
    running_queue_size=scheduler_outputs.running_queue_size,
    finished_requests_ids=finished_requests_ids,
    last_sampled_token_ids=last_sampled_token_ids,
)
outputs = self.model_executor.execute_model(execute_model_req)
```
【F:vllm/engine/llm_engine.py†L1335-L1345】【F:vllm/engine/llm_engine.py†L1351-L1354】

## 5. KV Cache Management (Paged Attention)

vLLM groups layers sharing the same KV cache into a `KVCacheGroupSpec` and uses a `BlockTable` to track which cache pages each request needs:

```python
class BlockTable:
    # maps request index -> list of cached block IDs
    self.block_table_cpu: Tensor(cpu pinned)
    self.block_table:     Tensor(GPU)
    self.slot_mapping:    Tensor mapping token -> slot

    def commit(self, num_reqs): ...  # push CPU->GPU
```
【F:vllm/v1/worker/block_table.py†L1-L49】

Cache format is specified per-layer via `AttentionSpec`:

```python
@dataclass
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    use_mla: bool

    @property
    def page_size_bytes(self) -> int: ...
```
【F:vllm/v1/kv_cache_interface.py†L73-L113】

## 6. Attention API & FlashInfer Integration

vLLM defines a two-stage Attention API: an `AttentionMetadataBuilder` prepares paged-KV indices and wrappers, and an `AttentionImpl` executes the kernel (FlashInfer, FlashAttention, FlexAttention, etc.).

### FlashInfer Metadata for Paged Attention

```python
@dataclass
class FlashInferMetadata:
    # Paged KV cache specification:
    paged_kv_indptr: Tensor  # [batch_size+1]
    paged_kv_indices: Tensor # concatenated page IDs
    paged_kv_last_page_len: Tensor # last-page sizes per request
    qo_indptr: Tensor        # query-output indptr
    num_decodes: int; num_prefills: int
    use_cascade: bool
    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper]
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper]
    cascade_wrapper: Optional[MultiLevelCascadeAttentionWrapper]
```
【F:vllm/v1/attention/backends/flashinfer.py†L151-L202】

The builder partitions requests into prefill vs decode and plans their wrappers:

```python
def _plan(self, attn_metadata: FlashInferMetadata):
    # (1) Cascade attention setup if shared prefix > 0
    # (2) Plan prefill wrapper on trailing pages
    # (3) Plan decode wrapper on leading single-token pages
```
【F:vllm/v1/attention/backends/flashinfer.py†L330-L415】

Finally, metadata is constructed and passed to the implementation kernel:

```python
attn_metadata = FlashInferMetadata(
    num_actual_tokens=num_actual_tokens,
    qo_indptr=qo_indptr,
    paged_kv_indptr=paged_kv_indptr,
    paged_kv_indices=paged_kv_indices,
    paged_kv_last_page_len=paged_kv_last_page_len,
    num_decodes=self._num_decodes,
    num_prefills=self._num_prefills,
    use_cascade=use_cascade,
    shared_kv_*=...,  # shared prefix pages
)
self._plan(attn_metadata)
```
【F:vllm/v1/attention/backends/flashinfer.py†L473-L500】

### Attention Implementation

The `FlashInferImpl` (or other backend) consumes the metadata and runs the fused paged-attention kernel:

```python
def forward(
   self,
   layer: torch.nn.Module,
   query: Tensor,
   key: Tensor,
   value: Tensor,
   kv_cache: Tensor,
   attn_metadata: FlashInferMetadata,
   output: Tensor,
) -> Tensor:
    # key/value are reshaped using attn_metadata.slot_mapping
    # paged-kv wrappers invoked internally
```
【F:vllm/v1/attention/backends/flashinfer.py†L404-L460】