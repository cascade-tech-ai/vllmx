"""Speculative *static-text* proposer (v1).

Single source of truth for static-text speculative-decoding logic used by
vLLM. The legacy v0 worker delegates to helpers defined here.

Highlights
----------
1. Robust line-level alignment of the generated context with the
   user‑supplied prediction using RapidFuzz (optional) or difflib.
2. Incremental per-request state to avoid re-processing the entire context.
3. Per-request predictions from ``SamplingParams.predicted_outputs`` only.

Implementation uses pure Python/NumPy on CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import difflib
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger

import time

# ---------------------------------------------------------------------------
# Logging configuration.
# ---------------------------------------------------------------------------

import os
VERBOSE = os.getenv('PREDICTED_OUTPUTS_VERBOSE') == '1'

logger = init_logger(__name__)

if VERBOSE:
    logger.setLevel("INFO")
else:
    logger.setLevel("WARNING")

# THE PLAN:
# - Make tests to just test static_text_proposer directly for fast iteration
# - Write tests:
#   - Propose nothing, match nothing
#   - Propose exact, match every token
#   - Propose partial with a few variations.  They should compare two multi-line bits of text
#     with some overlap.  Adding/removing lines.
#
# - Whether tests pass or not, we need to rewrite this:
#   - Keep track of current position, and current prediction as offset in prediction.  Starts at 0
#   - When we get new context, if we have a current prediction (>=0), and new text matches the next
#     text, then just continue (predict next N tokens)
#   - We still need to accumulate context into lines
#   - When prediction doesn't match, we set current prediction to -1 and switch to doing
#     alignment with every newline (not every time)

# ---------------------------------------------------------------------------
# Optional high-performance diff library *rapidfuzz*.
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.distance import LCSseq as _RF_LCS  # type: ignore

    _RAPIDFUZZ_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – executed only when missing
    logger.warning(
        "Optional dependency 'rapidfuzz' not found – falling back to "
        "difflib.SequenceMatcher for static-text alignment.  Install rapidfuzz "
        "for optimal speculative-decoding performance.")
    _RAPIDFUZZ_AVAILABLE = False

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

        # (no additional debug state)


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
    ) -> Optional[np.ndarray]:
        """Return next speculative tokens for *req_id* if prediction matches."""

        if not predicted_token_ids:
            return None

        # Lazily load tokenizer.
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # local import

            model_name_or_path = self.vllm_config.model_config.model  # type: ignore[attr-defined]
            self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

        # print token ids and text for context_token_ids and predicted_token_ids
        logger.info(f"Context token ids: {context_token_ids.tolist()}")
        logger.info(f"Context text: {self._tokenizer.decode(context_token_ids)}")
        if predicted_token_ids:
            logger.info(f"Predicted token ids: {predicted_token_ids}")
            logger.info(f"Predicted text: {self._tokenizer.decode(predicted_token_ids)}")
        else:
            logger.info("No predicted token ids")

        propose_start = time.time()

        # Initialise request state on first call.
        if req_id not in self._state:
            start = time.time()
            newline_set: Set[int] = self._detect_newline_tokens(predicted_token_ids)
            self._state[req_id] = _ReqState(list(predicted_token_ids), newline_set)
            logger.info(f"  Created _ReqState: {(time.time() - start) * 1000:0.2f}ms")

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

        # ------------------------------------------------------------------
        # Attempt alignment between *completed* context lines and the global
        # prediction. Instead of returning early on failure we record the
        # mismatch and continue so that debug logging still happens – this
        # is crucial when investigating why alignment fails in production.
        # ------------------------------------------------------------------

        start = time.time()
        line_cursor = _align_cursor_lines(st.predicted_line_tuples, completed)
        logger.info(f"  _align_cursor_lines: {(time.time() - start) * 1000:0.2f}ms")

        if line_cursor and line_cursor < len(st.predicted_line_tuples):
            pred_line_start = st.line_starts[line_cursor]

            # Build predicted line tokens for the *current* line.
            pred_line_tokens: List[int] = []
            idx = pred_line_start
            while idx < len(st.predicted_tokens) and st.predicted_tokens[idx] not in st.newline_set:
                pred_line_tokens.append(st.predicted_tokens[idx])
                idx += 1

            if current_prefix != pred_line_tokens[:len(current_prefix)]:
                success = False

            token_cursor = pred_line_start + len(current_prefix)
            if token_cursor >= len(st.predicted_tokens):
                success = False

            end = min(token_cursor + self.k, len(st.predicted_tokens))
            arr = np.array(st.predicted_tokens[token_cursor:end], dtype=np.int32)
        else:
            arr = np.empty(0, dtype=np.int32)

        logger.info(f"Context:\n" + "".join(self._tokenizer.decode(t) for t in st.ctx_line_tuples))
        logger.info(f"Predicted {len(arr)} tokens: {self._tokenizer.decode(arr)}")

        logger.info(f"  Proposed {arr.size} tokens in {(time.time() - propose_start) * 1000:0.2f}ms")

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

        start = time.time()

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

        logger.info(f"  _detect_newline_tokens: {(time.time() - start) * 1000:.2f}")

        self._global_newline_set = newline_set
        return newline_set

    # ------------------------------------------------------------------ draft model API compatibility

    def load_model(self, *args, **kwargs):  # noqa: D401 – interface stub
        # StaticTextProposer is not a model – nothing to load.
        pass
