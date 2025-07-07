"""Static-text proposer for speculative decoding.

This worker returns user-supplied *predicted* tokens so that the target model
can *accept* already known continuations in a single forward pass.  The
predicted sequence is provided **per request** via
``SamplingParams.predicted_outputs``.  No worker-level global prediction is
supported anymore.

The class is selected by setting::

    speculative_config = {
        "method": "static_text",
        "num_speculative_tokens": 128,
    }

When *static_text* is chosen, :pymod:`vllm.spec_decode.spec_decode_worker`
instantiates this worker directly.
"""

from __future__ import annotations

import logging
import weakref
from typing import List, Optional, Set, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Re-use utilities from the v1 implementation so we keep only *one* source of
# truth for the alignment logic.
# ---------------------------------------------------------------------------

from vllm.v1.spec_decode.static_text_proposer import (
    _split_by_newline_tokens,  # noqa: F401 – re-exported helper
    _align_cursor_lines,       # noqa: F401 – re-exported helper
)

import difflib  # retained for public API compatibility (may be unused)

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.spec_decode.interfaces import SpeculativeProposals
from vllm.spec_decode.proposer_worker_base import NonLLMProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer

# ---------------------------------------------------------------------------
# Optional high-performance diff library *rapidfuzz*.
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.distance import LCSseq as _RF_LCS  # type: ignore

    _RAPIDFUZZ_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – executed only when missing
    _RAPIDFUZZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging – capture only warnings / errors via the default vLLM handler.
# ---------------------------------------------------------------------------

logger = init_logger(__name__)
if _RAPIDFUZZ_AVAILABLE:
    # Keep the worker largely silent to avoid flooding the logs.
    logger.setLevel(logging.ERROR)
else:
    # Elevate log level so that the single warning about the missing optional
    # dependency is surfaced to the operator.
    logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Internal helpers – line-level token processing & diff algorithms
# ---------------------------------------------------------------------------

# Local helper functions for newline detection remain, but the heavy-lifting
# (line splitting & alignment) is imported from the v1 module above.


# ---------------------------------------------------------------------------
# Helper dummy model (vLLM expects every worker to expose one)
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    def forward(self, *args, **kwargs):  # noqa: D401 – dummy
        raise RuntimeError("StaticTextWorker has no forward pass")


# ---------------------------------------------------------------------------
# Worker implementation
# ---------------------------------------------------------------------------


class StaticTextWorker(NonLLMProposerWorkerBase):
    """Proposer that emits caller-supplied *static* token sequences."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        local_rank: int = 0,
        device_type: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(vllm_config)

        self.local_rank = local_rank
        self.device_type = device_type

        self.device: Optional[torch.device] = None

        self._tokenizer = None  # lazily initialised for debug pretty-print

        self._proposer: Optional[Top1Proposer] = None

    # ------------------------------------------------------------------
    # WorkerBase overrides
    # ------------------------------------------------------------------

    def init_device(self) -> None:  # noqa: D401 – interface method
        self.device = torch.device(f"{self.device_type}:{self.local_rank}")

        self._proposer = Top1Proposer(
            weakref.proxy(self),
            device=self.device,
            vocab_size=self.vocab_size,
        )

        # No additional preparation required for static-text proposals – all
        # prediction information is supplied *per request* via
        # SamplingParams.predicted_outputs.

    def load_model(self) -> None:  # noqa: D401 – interface method
        pass  # no model to load

    def get_model(self) -> nn.Module:  # noqa: D401 – interface method
        return _DummyModel()

    # ------------------------------------------------------------------
    # Speculative-decoding hooks
    # ------------------------------------------------------------------

    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],  # noqa: D401 – interface param
    ) -> Tuple[Optional[List[Optional[SamplerOutput]]], bool]:
        # ------------------------------------------------------------------
        # Validate the request – StaticTextWorker supports only the simple
        # execution path (no KV-cache manipulations).
        # ------------------------------------------------------------------

        self._raise_if_unsupported(execute_model_req)

        token_id_tensors: List[Optional[torch.Tensor]] = []
        token_prob_tensors: List[Optional[torch.Tensor]] = []
        has_any_proposal = False

        pad_token_id = getattr(
            self.vllm_config.model_config,  # type: ignore[attr-defined]
            "eos_token_id",
            0,
        )

        # ------------------------------------------------------------------ process each sequence group separately
        for seq_group_metadata in execute_model_req.seq_group_metadata_list:
            seq_data = next(iter(seq_group_metadata.seq_data.values()))

            sampling_params = seq_group_metadata.sampling_params

            # ------------------------------------------------------------------
            # 1.  Retrieve *predicted* tokens for this sequence.
            # ------------------------------------------------------------------

            pred_ids: Optional[List[int]] = None

            if (sampling_params is not None and
                    sampling_params.predicted_outputs is not None):
                po = sampling_params.predicted_outputs
                if po.predicted_token_ids is not None:
                    pred_ids = list(po.predicted_token_ids)
                elif po.predicted_text is not None:
                    # Lazily create a tokenizer for the worker.
                    if self._tokenizer is None:
                        from transformers import AutoTokenizer  # local import

                        model_name_or_path = self.vllm_config.model_config.model  # type: ignore[attr-defined]
                        self._tokenizer = AutoTokenizer.from_pretrained(
                            model_name_or_path,
                            use_fast=True,
                        )

                    pred_ids = self._tokenizer.encode(
                        po.predicted_text,
                        add_special_tokens=False,
                    )

            # No prediction available → no proposals.
            if not pred_ids:
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            # ------------------------------------------------------------------
            # 2.  Obtain the *state* dictionary (lives on SamplingParams).
            # ------------------------------------------------------------------

            if sampling_params is None:
                # Should never happen but keep the worker robust.
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            extra: Dict[str, Any]
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            extra = sampling_params.extra_args  # type: ignore[assignment]

            state = extra.setdefault("_static_text_state", {})  # type: Dict[str, Any]

            # ------------------------------------------------------------------
            # 3.  Prepare static data (prediction split into *lines*).
            # ------------------------------------------------------------------

            if "predicted_line_tuples" not in state:
                # Compute newline detection set.
                newline_set: Set[int] = set()
                if self._tokenizer is None:
                    from transformers import AutoTokenizer  # local import

                    model_name_or_path = self.vllm_config.model_config.model  # type: ignore[attr-defined]
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        model_name_or_path,
                        use_fast=True,
                    )

                seen: Set[int] = set()
                for tok in pred_ids:
                    if tok in seen:
                        continue
                    seen.add(tok)
                    if "\n" in self._tokenizer.decode([tok]):
                        newline_set.add(tok)

                # Fallback: if no newline has been detected yet, force-add the
                # tokenizer's standalone "\n" token (common for GPT-style
                # tokenisers).  The encode call may yield multiple tokens – we
                # add the first one.
                if not newline_set:
                    try:
                        nl_enc = self._tokenizer.encode("\n", add_special_tokens=False)
                        if nl_enc:
                            newline_set.add(nl_enc[0])
                    except Exception:  # pragma: no cover – very unlikely
                        pass

                state["newline_set"] = newline_set

                # Pre-compute prediction split into immutable *line* tuples.
                predicted_line_tuples = _split_by_newline_tokens(pred_ids, newline_set)
                state["predicted_line_tuples"] = predicted_line_tuples

                # Map line index → token start index for quick back-conversion.
                line_starts: List[int] = [0]
                for idx, tok in enumerate(pred_ids):
                    if tok in newline_set:
                        line_starts.append(idx + 1)
                state["line_starts"] = line_starts

            newline_set: Set[int] = state["newline_set"]
            predicted_line_tuples: List[Tuple[int, ...]] = state["predicted_line_tuples"]
            line_starts: List[int] = state["line_starts"]

            # ------------------------------------------------------------------
            # 4.  Update *context* tracking buffers.
            # ------------------------------------------------------------------

            ctx_processed: int = state.get("ctx_processed", 0)
            ctx_line_tuples: List[Tuple[int, ...]] = state.setdefault("ctx_line_tuples", [])
            current_line_tokens: List[int] = state.setdefault("current_line_tokens", [])

            output_tokens = list(seq_data.get_output_token_ids())

            # If the sequence was *reset* (e.g. new prompt) we clear state.
            if len(output_tokens) < ctx_processed:
                ctx_processed = 0
                ctx_line_tuples.clear()
                current_line_tokens.clear()

            for tok in output_tokens[ctx_processed:]:
                current_line_tokens.append(tok)
                if tok in newline_set:
                    ctx_line_tuples.append(tuple(current_line_tokens))
                    current_line_tokens.clear()

            ctx_processed = len(output_tokens)
            state["ctx_processed"] = ctx_processed

            last_token_is_nl = (
                len(output_tokens) > 0 and output_tokens[-1] in newline_set
            )

            if last_token_is_nl:
                completed_line_tuples = ctx_line_tuples + [tuple()]
                current_line_prefix: List[int] = []
            else:
                completed_line_tuples = ctx_line_tuples
                current_line_prefix = list(current_line_tokens)

            # ------------------------------------------------------------------
            # 5.  Align completed lines within the prediction.
            # ------------------------------------------------------------------

            line_cursor = _align_cursor_lines(predicted_line_tuples, completed_line_tuples)

            if (
                line_cursor is None or
                line_cursor >= len(predicted_line_tuples)
            ):
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            # ------------------------------------------------------------------
            # 6.  Verify *current* line matches the prediction prefix.
            # ------------------------------------------------------------------

            predicted_line_start_idx = line_starts[line_cursor]

            predicted_line_tokens: List[int] = []
            idx = predicted_line_start_idx
            while (
                idx < len(pred_ids) and
                pred_ids[idx] not in newline_set
            ):
                predicted_line_tokens.append(pred_ids[idx])
                idx += 1

            if current_line_prefix != predicted_line_tokens[: len(current_line_prefix)]:
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            # ------------------------------------------------------------------
            # 7.  Produce speculative *next* tokens.
            # ------------------------------------------------------------------

            token_cursor = predicted_line_start_idx + len(current_line_prefix)

            if token_cursor >= len(pred_ids):
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            end = min(token_cursor + sample_len, len(pred_ids))
            proposed_token_ids = pred_ids[token_cursor:end]

            if not proposed_token_ids:
                token_id_tensors.append(None)
                token_prob_tensors.append(None)
                continue

            # ------------------------------------------------------------------
            # 8.  Convert to tensors expected by vLLM core.
            # ------------------------------------------------------------------

            announced_k = sample_len  # the worker must return exactly sample_len proposals

            proposal_ids_tensor = torch.full(
                (announced_k,),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )

            true_ids_tensor = torch.tensor(
                proposed_token_ids,
                dtype=torch.long,
                device=self.device,
            )

            proposal_ids_tensor[: true_ids_tensor.numel()] = true_ids_tensor

            probs_tensor = torch.zeros(
                (announced_k, self.vocab_size),
                dtype=torch.float32,
                device=self.device,
            )

            if true_ids_tensor.numel():
                probs_tensor[: true_ids_tensor.numel()] = torch.nn.functional.one_hot(
                    true_ids_tensor,
                    num_classes=self.vocab_size,
                ).to(torch.float32)

            probs_tensor[true_ids_tensor.numel():, pad_token_id] = 1.0

            token_id_tensors.append(proposal_ids_tensor)
            token_prob_tensors.append(probs_tensor)
            has_any_proposal = True

        # ------------------------------------------------------------------
        # 9.  Package results for the runtime – same order as input list.
        # ------------------------------------------------------------------

        if not has_any_proposal:
            return None, False

        outputs: List[Optional[SamplerOutput]] = []
        for ids_tensor, probs_tensor in zip(token_id_tensors, token_prob_tensors):
            if ids_tensor is None:
                outputs.append(None)
            else:
                outputs.append(
                    SamplerOutput(
                        outputs=None,
                        sampled_token_probs=probs_tensor,
                        logprobs=torch.zeros_like(probs_tensor),
                        sampled_token_ids=ids_tensor,
                    )
                )

        # Second element in tuple indicates whether the tensors are already
        # transposed – ours are *not*.
        return outputs, False

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        assert self._proposer is not None
        return self._proposer.get_spec_proposals(execute_model_req, seq_ids_with_bonus_token_in_last_step)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_if_unsupported(execute_model_req: ExecuteModelRequest) -> None:
        if any([
            execute_model_req.blocks_to_swap_in,
            execute_model_req.blocks_to_swap_out,
            execute_model_req.blocks_to_copy,
        ]):
            raise NotImplementedError("StaticTextWorker does not support cache swap operations")

        if any(len(sg.seq_data) != 1 for sg in execute_model_req.seq_group_metadata_list):
            raise NotImplementedError("StaticTextWorker expects exactly one sequence per group (no beam search)")
