"""Speculative *static-text* proposer (v1).

This file is now the **single source of truth** for the static-text
speculative-decoding logic used by vLLM.  The legacy v0
``vllm.spec_decode.static_text_worker.StaticTextWorker`` delegates to the
helpers defined here so that future maintenance only needs to touch this
module.

Highlights
----------
1. Robust *line-level* alignment of the **generated context** with the
   **user-supplied prediction**.  Two interchangeable algorithms are provided:

   • rapidfuzz (fast, optional dependency)
   • difflib   (C-accelerated stdlib fallback)

2. Incremental per-request state to avoid re-processing the entire context.

3. No reliance on *global* predictions configured at worker creation – the
   only source for predicted text/tokens is the
   ``SamplingParams.predicted_outputs`` field passed alongside each request.

The implementation is purposely **pure Python / NumPy** so that it can run on
the CPU worker process without touching GPU memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import difflib
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger

# ---------------------------------------------------------------------------
# YAML helper – ensure *inner* token lists are emitted in **flow** style so
# that each list appears on a single line: ``[1, 2, 3]``.  We do this by
# defining a lightweight ``FlowList`` subtype with a custom representer.
# ---------------------------------------------------------------------------

try:
    import yaml  # type: ignore

    class FlowList(list):
        """Marker subclass – dumped using YAML *flow* style."""

    def _represent_flow_seq(dumper: yaml.Dumper, data: list):  # type: ignore[name-defined]
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True)

    yaml.SafeDumper.add_representer(FlowList, _represent_flow_seq)  # type: ignore[attr-defined]

except ModuleNotFoundError:  # pragma: no cover – YAML optional for production

    class FlowList(list):  # type: ignore[empty-body]
        pass

# ---------------------------------------------------------------------------
# Development / debugging switches
# ---------------------------------------------------------------------------

# When set to True the StaticTextProposer will cooperate with the
# gpu_model_runner to emit a *structured* YAML debug log (see
# `cascade/docs/debug_logging.txt`).  The drafter only gathers pieces of
# information that are *easy* for it to compute – the GPU runner owns the
# actual file-write so that the log entry can include the overall iteration
# latency.

DEBUG_PREDICTED_OUTPUTS = True  # noqa: N816 – global constant by design

# ---------------------------------------------------------------------------
# Optional high-performance diff library *rapidfuzz*.
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.distance import LCSseq as _RF_LCS  # type: ignore

    _RAPIDFUZZ_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – executed only when missing
    _RAPIDFUZZ_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging configuration.
# ---------------------------------------------------------------------------

logger = init_logger(__name__)
if _RAPIDFUZZ_AVAILABLE:
    logger.setLevel("ERROR")  # stay quiet in production
else:
    logger.setLevel("WARNING")
    logger.warning(
        "Optional dependency 'rapidfuzz' not found – falling back to "
        "difflib.SequenceMatcher for static-text alignment.  Install rapidfuzz "
        "for optimal speculative-decoding performance.")

class _ReqState:
    """Mutable per-request state stored inside the proposer."""

    __slots__ = (
        "predicted_tokens",
        "newline_set",
        "predicted_line_tuples",
        "line_starts",
        # incremental context tracking
        "ctx_processed",
        "ctx_line_tuples",
        "current_line_tokens",
        # optional debug fields
        "debug_state",
        "_debug_prev_completed_len",
    )

    def __init__(self, predicted_tokens: List[int], newline_set: Set[int]):
        self.predicted_tokens: List[int] = predicted_tokens
        self.newline_set: Set[int] = newline_set

        # Pre-compute line tuples & mapping.
        self.predicted_line_tuples: List[Tuple[int, ...]] = _split_by_newline_tokens(
            predicted_tokens, newline_set)

        self.line_starts: List[int] = [0]
        for idx, tok in enumerate(predicted_tokens):
            if tok in newline_set:
                self.line_starts.append(idx + 1)

        # Context-tracking buffers.
        self.ctx_processed = 0
        self.ctx_line_tuples: List[Tuple[int, ...]] = []
        self.current_line_tokens: List[int] = []

        # ------------------------------------------------------------------
        # Debug helpers – initialised lazily on first access when debugging is
        # enabled.  The structure follows the design in
        # `cascade/docs/debug_logging.txt`.
        # ------------------------------------------------------------------

        self.debug_state: dict | None = None
        self._debug_prev_completed_len: int = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _split_by_newline_tokens(tokens: List[int], newline_set: Set[int]) -> List[Tuple[int, ...]]:
    """Return immutable *lines* where each element is a **tuple of tokens**.

    A token whose decoded string contains a literal ``"\n"`` is considered a
    *newline marker*.  The marker **itself** is stored within the *preceding*
    line so that the semantics mirror the original text – multiple consecutive
    newline tokens therefore yield *empty* lines between them.
    """

    lines: List[List[int]] = [[]]
    for tok in tokens:
        lines[-1].append(tok)
        if tok in newline_set:
            lines.append([])

    return [tuple(line) for line in lines]


# ---------------------------------------------------------------------------
# Alignment helpers – two interchangeable implementations -------------------
# ---------------------------------------------------------------------------


def _rf_align_cursor_lines(a: List[Tuple[int, ...]], b: List[Tuple[int, ...]]) -> Optional[int]:
    """Align completed context *b* inside prediction *a* using rapidfuzz.

    Returns the *line cursor* in *a* immediately after the last matched line
    or ``None`` if the entire *b* sequence cannot be aligned as a subsequence
    of *a*.
    """

    if not _RAPIDFUZZ_AVAILABLE:
        return None

    m = len(b)
    if m == 0:
        return 0

    ops = _RF_LCS.editops(a, b)

    x = 0  # position in a
    y = 0  # position in b

    for op in ops:
        # walk the *snake* (equal region) before this edit op
        while x < op.src_pos and y < op.dest_pos and y < m:
            x += 1
            y += 1

        if y >= m:
            break

        if op.tag == "delete":
            x += 1
        elif op.tag == "insert":
            y += 1
        else:  # replace
            x += 1
            y += 1

        if y >= m:
            break

    # trailing snake after last edit
    while y < m and x < len(a) and a[x] == b[y]:
        x += 1
        y += 1

    if y == m and x > 0 and a[x - 1] == b[-1]:
        return x

    return None


def _sm_align_cursor_lines(a: List[Tuple[int, ...]], b: List[Tuple[int, ...]]) -> Optional[int]:
    """Align completed context *b* inside prediction *a* using difflib."""

    m = len(b)
    if m == 0:
        return 0

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    x = 0
    y = 0
    best_x: Optional[int] = None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            block_len = i2 - i1
            take = min(block_len, m - y)
            x += take
            y += take
            if y == m:
                best_x = x  # candidate cursor – may be refined later
            if take < block_len:
                break
        elif tag == "delete":
            x += i2 - i1
        elif tag == "insert":
            y += j2 - j1
            if y >= m:
                return best_x
        else:  # replace
            x += i2 - i1
            y += j2 - j1
            if y >= m:
                return best_x

    if y == m and x > 0 and a[x - 1] == b[-1]:
        last_line = b[-1]
        candidate = x
        for i in range(x, len(a)):
            if a[i] == last_line:
                candidate = i + 1
        return candidate

    return best_x


# The exported alignment helper – chosen at import time.

_align_cursor_lines = _rf_align_cursor_lines if _RAPIDFUZZ_AVAILABLE else _sm_align_cursor_lines


# ---------------------------------------------------------------------------
# Proposer implementation
# ---------------------------------------------------------------------------


class StaticTextProposer:

    def __init__(self, vllm_config: VllmConfig):
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.vllm_config = vllm_config

        # Tokeniser and model-wide newline set (both initialised lazily the
        # first time we need them).
        self._tokenizer = None
        self._global_newline_set: Set[int] | None = None

        # req_id → _ReqState
        self._state: Dict[str, _ReqState] = {}

        logger.info("Initialized StaticTextProposer")

    # ------------------------------------------------------------------ public API

    def propose(
        self,
        req_id: str,
        context_token_ids: np.ndarray,
        predicted_token_ids: Optional[List[int]],
        debug_iteration: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Return next speculative tokens for *req_id* if prediction matches."""

        if not predicted_token_ids:
            return None

        # Initialise request state on first call.
        if req_id not in self._state:
            newline_set: Set[int] = self._detect_newline_tokens(predicted_token_ids)
            self._state[req_id] = _ReqState(list(predicted_token_ids), newline_set)

        st = self._state[req_id]

        ctx_tokens = context_token_ids.tolist()

        # Fast path: reset when context shrinks (new prompt).
        if len(ctx_tokens) < st.ctx_processed:
            st.ctx_processed = 0
            st.ctx_line_tuples.clear()
            st.current_line_tokens.clear()

        # Update incremental context buffers.
        for tok in ctx_tokens[st.ctx_processed:]:
            st.current_line_tokens.append(tok)

            # The newline set is now *comprehensive* because it is generated
            # once at startup by exhaustively scanning the full vocabulary in
            # `_detect_newline_tokens`.  We therefore avoid the previous,
            # expensive per-token decode that attempted to discover newline
            # tokens on-the-fly.
            if tok in st.newline_set:
                st.ctx_line_tuples.append(tuple(st.current_line_tokens))
                st.current_line_tokens.clear()

        st.ctx_processed = len(ctx_tokens)

        last_is_nl = ctx_tokens and ctx_tokens[-1] in st.newline_set

        if last_is_nl:
            completed = st.ctx_line_tuples + [tuple()]
            current_prefix: List[int] = []
        else:
            completed = st.ctx_line_tuples
            current_prefix = list(st.current_line_tokens)

        predicted_lines_text = [self._tokenizer.decode(l) for l in st.predicted_line_tuples]
        completed_lines_text = [self._tokenizer.decode(l) for l in completed]


        # ------------------------------------------------------------------
        # Attempt alignment between *completed* context lines and the global
        # prediction. Instead of returning early on failure we record the
        # mismatch and continue so that debug logging still happens – this
        # is crucial when investigating why alignment fails in production.
        # ------------------------------------------------------------------

        success = True

        line_cursor = _align_cursor_lines(st.predicted_line_tuples, completed)
        if line_cursor is None or line_cursor >= len(st.predicted_line_tuples):
            success = False

        if success:
            pred_line_start = st.line_starts[line_cursor]

            # Build predicted line tokens for the *current* line.
            pred_line_tokens: List[int] = []
            idx = pred_line_start
            while idx < len(st.predicted_tokens) and st.predicted_tokens[idx] not in st.newline_set:
                pred_line_tokens.append(st.predicted_tokens[idx])
                idx += 1

            if current_prefix != pred_line_tokens[:len(current_prefix)]:
                success = False

        if success:
            token_cursor = pred_line_start + len(current_prefix)
            if token_cursor >= len(st.predicted_tokens):
                success = False

        if success:
            end = min(token_cursor + self.k, len(st.predicted_tokens))
            arr = np.array(st.predicted_tokens[token_cursor:end], dtype=np.int32)
        else:
            arr = np.empty(0, dtype=np.int32)

        logger.info(f"Context:\n" + "".join(self._tokenizer.decode(t) for t in st.ctx_line_tuples))
        logger.info(f"Predicted {len(arr)} tokens: {self._tokenizer.decode(arr)}")

        # ------------------------------------------------------------------
        # Optional debug collection – augment the provided *mutable* dict so
        # that the caller (gpu_model_runner) can add timing information and
        # write the combined structure out to disk.
        # ------------------------------------------------------------------

        if DEBUG_PREDICTED_OUTPUTS and debug_iteration is not None:
            # Lazily initialise the per-request debug_state once we have a
            # tokenizer to decode the prediction.
            if st.debug_state is None:
                st.debug_state = {
                    "predicted_lines_tokens": [FlowList(t) for t in st.predicted_line_tuples],
                    "predicted_lines_text": predicted_lines_text,
                    "iterations": [],
                }

            # Newly completed *context* lines since the previous iteration.
            prev_len = st._debug_prev_completed_len
            new_completed = completed[prev_len:]
            st._debug_prev_completed_len = len(completed)

            debug_iteration["new_context_lines_tokens"] = [FlowList(t) for t in new_completed]
            debug_iteration["new_context_lines_text"] = [
                self._tokenizer.decode(t) for t in new_completed
            ]

            # Current *in-progress* prefix (may be empty when context ends with
            # a newline).
            debug_iteration["current_prefix_tokens"] = FlowList(current_prefix)
            debug_iteration["current_prefix_text"] = self._tokenizer.decode(
                current_prefix) if current_prefix else ""

            # Proposer output – will be overwritten by caller *after* the
            # duration is measured so we only set a default here.
            if arr.size > 0:
                debug_iteration.setdefault("predicted_tokens", FlowList(arr.tolist()))
                debug_iteration.setdefault("predicted_text",
                                            self._tokenizer.decode(arr))
            else:
                debug_iteration.setdefault("predicted_tokens", FlowList())
                debug_iteration.setdefault("predicted_text", "")

            # Finally append to the iterations list – the caller may still
            # mutate duration / predicted_{tokens,text} but the reference is
            # shared so we are safe.
            st.debug_state["iterations"].append(debug_iteration)

        return arr if arr.size > 0 else None

    # ------------------------------------------------------------------ internal helpers

    def _detect_newline_tokens(self, tokens: List[int] | None = None) -> Set[int]:
        """Return *global* set of token ids whose decoded form contains "\n".

        The scan over the entire vocabulary is performed **once** per
        StaticTextProposer instance to maximise runtime performance during
        generation.  Subsequent calls simply return the cached set.
        """

        if self._global_newline_set is not None:
            return self._global_newline_set

        # Lazily load tokenizer.
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # local import

            model_name_or_path = self.vllm_config.model_config.model  # type: ignore[attr-defined]
            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        tok = self._tokenizer

        newline_set: Set[int] = set()

        try:
            vocab_size = tok.vocab_size  # type: ignore[attr-defined]
        except AttributeError:
            vocab_size = len(tok.get_vocab())  # type: ignore[arg-type]

        # Iterate through the full vocabulary once and cache the result.
        # Decoding in a tight Python loop is acceptable because this runs at
        # initialisation time only.
        for tid in range(vocab_size):
            try:
                if "\n" in tok.decode([tid]):
                    newline_set.add(tid)
            except Exception:  # pragma: no cover – ignore decode failures
                continue

        # Ensure at least the canonical encoded newline token is present even
        # if the scan failed.
        if not newline_set:
            try:
                enc = tok.encode("\n", add_special_tokens=False)
                if enc:
                    newline_set.add(enc[0])
            except Exception:  # pragma: no cover
                pass

        self._global_newline_set = newline_set
        return newline_set

    # ------------------------------------------------------------------ draft model API compatibility

    def load_model(self, *args, **kwargs):  # noqa: D401 – interface stub
        # StaticTextProposer is not a model – nothing to load.
        pass
