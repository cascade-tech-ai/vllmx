"""Speculative static‑text proposer (v1).

Single source of truth for static‑text speculative‑decoding logic used by
vLLM. The legacy v0 worker delegates to helpers defined here.

Highlights
----------
1. Robust line‑level alignment of the generated context with the
   user‑supplied prediction using RapidFuzz (optional) or difflib.
2. Incremental per‑request state to avoid re‑processing the entire context.
3. Per‑request predictions from ``SamplingParams.predicted_outputs`` only.

Algorithm (cursor + sparse alignment)
-------------------------------------
We maintain a per‑request integer cursor into the predicted token sequence.

- cursor >= 0: we are aligned; cursor is the index in ``predicted_tokens``
  that corresponds to the next token to generate.
- cursor == -1: we are lost; we must realign before proposing again.

On each call with the full output context tokens so far:
1) Fast‑path compare: if cursor >= 0, compare only the newly added context
   tokens to the next tokens in the prediction. If they match, advance the
   cursor by the number of new tokens and immediately propose up to ``k``
   next predicted tokens. If they do not match, set cursor = -1.
2) Sparse alignment: while cursor == -1, attempt line‑level alignment only
   when a new completed line appears in the context (i.e., after a newline).
   We align completed context lines against predicted lines. If alignment
   succeeds and the current partial line (if any) matches the prefix of the
   corresponding predicted line, we set the cursor to the predicted token
   index at the end of the current prefix and begin proposing again. If the
   alignment fails, we suppress further alignment attempts until another line
   completes.

Implementation uses pure Python/NumPy on CPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING

import difflib
import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

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
        # line ending mode: 'lf' or 'crlf'
        "line_ending_mode",
        # cursor-based fast path
        "pred_cursor",
        "align_blocked_until_completed_lines",
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

        # Cursor state
        self.pred_cursor: int = 0  # >=0 aligned at this token index; -1 lost
        self.align_blocked_until_completed_lines: int = -1

        # Line endings — initialized by caller (proposer) after decoding text
        self.line_ending_mode: str = "lf"

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
            # Use cached tokenizer from config (per‑process singleton).
            model_cfg = self.vllm_config.model_config  # type: ignore[attr-defined]
            self._tokenizer = cached_tokenizer_from_config(model_cfg)

        # Verbose header per call
        if VERBOSE:
            logger.info(f"[req={req_id}] propose() k={self.k}")

        propose_start = time.time()

        # Initialise request state on first call.
        if req_id not in self._state:
            start = time.time()
            newline_set: Set[int] = self._detect_newline_tokens(predicted_token_ids)
            self._state[req_id] = _ReqState(list(predicted_token_ids), newline_set)
            if VERBOSE:
                logger.info(
                    f"  [state:new] init in {(time.time() - start) * 1000:0.2f}ms")
                logger.info(
                    f"  [state:new] prediction tokens: {predicted_token_ids}")
                try:
                    pred_text = self._tokenizer.decode(predicted_token_ids)
                    logger.info(
                        f"  [state:new] prediction text: {pred_text}")
                except Exception:
                    pred_text = None
                    logger.info("  [state:new] prediction text: <decode error>")
                # Detect initial line ending mode from prediction text (if any)
                try:
                    if pred_text is None:
                        pred_text = self._tokenizer.decode(predicted_token_ids)
                    mode = self._detect_line_ending_mode_from_text(pred_text)
                except Exception:
                    mode = "lf"
                self._state[req_id].line_ending_mode = mode
                if VERBOSE:
                    logger.info(f"  [le] init mode={mode}")

        st = self._state[req_id]

        ctx_tokens = context_token_ids.tolist()

        # Capture previous processed length and the new segment of tokens.
        prev_ctx_processed = st.ctx_processed
        new_segment: List[int] = ctx_tokens[prev_ctx_processed:]
        if VERBOSE:
            logger.info(
                f"  [ctx:new] tokens: {new_segment if new_segment else '[]'}")
            try:
                logger.info(
                    f"  [ctx:new] text: {self._tokenizer.decode(new_segment)}")
            except Exception:
                logger.info("  [ctx:new] text: <decode error>")

        # Fast path: reset when context shrinks (new prompt).
        if len(ctx_tokens) < st.ctx_processed:
            st.ctx_processed = 0
            st.ctx_line_tuples.clear()
            st.current_line_tokens.clear()
            st.pred_cursor = 0
            st.align_blocked_until_completed_lines = -1

        # Update incremental context buffers.
        for tok in ctx_tokens[st.ctx_processed:]:
            st.current_line_tokens.append(tok)

            if tok in st.newline_set:
                st.ctx_line_tuples.append(tuple(st.current_line_tokens))
                st.current_line_tokens.clear()

        st.ctx_processed = len(ctx_tokens)

        last_is_nl = bool(ctx_tokens) and (ctx_tokens[-1] in st.newline_set)

        completed = st.ctx_line_tuples
        current_prefix: List[int] = [] if last_is_nl else list(
            st.current_line_tokens)

        # ------------------------------------------------------------------
        # Attempt alignment between *completed* context lines and the global
        # prediction. Instead of returning early on failure we record the
        # mismatch and continue so that debug logging still happens – this
        # is crucial when investigating why alignment fails in production.
        # ------------------------------------------------------------------

        # Line-ending switching: if new segment contains any newline, detect
        # the style in the latest context and switch modes if needed.
        if new_segment and any(t in st.newline_set for t in new_segment):
            # Decode a tail window for robust detection
            tail_tok = ctx_tokens[max(0, len(ctx_tokens) - 128):]
            try:
                tail_text = self._tokenizer.decode(tail_tok)
                # Find style of the last newline occurrence in the tail
                last_nl = tail_text.rfind("\n")
                if last_nl != -1:
                    new_mode = "crlf" if last_nl > 0 and tail_text[last_nl - 1] == "\r" else "lf"
                else:
                    new_mode = None
            except Exception:
                new_mode = None

            if new_mode and new_mode != st.line_ending_mode:
                # Switch: convert prediction text and re-tokenize
                try:
                    cur_pred_text = self._tokenizer.decode(st.predicted_tokens)
                except Exception:
                    cur_pred_text = None
                if cur_pred_text is not None:
                    switched_text = self._convert_line_endings(cur_pred_text,
                                                              new_mode)
                    new_pred_tokens = self._tokenizer.encode(
                        switched_text, add_special_tokens=False)
                    if VERBOSE:
                        logger.info(
                            f"  [le] switch {st.line_ending_mode} -> {new_mode}; retokenized {len(st.predicted_tokens)} -> {len(new_pred_tokens)} tokens")
                    st.predicted_tokens = list(new_pred_tokens)
                    self._reindex_prediction(st)
                    st.line_ending_mode = new_mode
                    # Force re-alignment to be safe after retokenization
                    st.pred_cursor = -1
                    st.align_blocked_until_completed_lines = len(
                        st.ctx_line_tuples) - 1

        # 1) Fast-path cursor advance if still aligned
        arr: Optional[np.ndarray]
        arr = None

        if st.pred_cursor >= 0:
            # Compare only the new segment against the prediction from cursor.
            n_new = len(new_segment)
            ok = True
            if n_new > 0:
                # Ensure we don't run past the prediction.
                end_idx = st.pred_cursor + n_new
                if end_idx > len(st.predicted_tokens):
                    ok = False
                else:
                    expected = st.predicted_tokens[st.pred_cursor:end_idx]
                    ok = expected == new_segment
            # Log cursor preview and match status
            if VERBOSE:
                preview_end = min(st.pred_cursor + self.k,
                                  len(st.predicted_tokens))
                preview = st.predicted_tokens[st.pred_cursor:preview_end]
                logger.info(
                    f"  [cursor] pos={st.pred_cursor} next_k_ids={preview}")
                try:
                    logger.info(
                        f"  [cursor] next_k_text: {self._tokenizer.decode(preview)}")
                except Exception:
                    logger.info("  [cursor] next_k_text: <decode error>")
                if n_new > 0:
                    if ok:
                        logger.info("  [cursor] match=new tokens match prediction")
                    else:
                        try:
                            got_txt = self._tokenizer.decode(new_segment)
                            exp_txt = self._tokenizer.decode(
                                st.predicted_tokens[st.pred_cursor:st.pred_cursor + n_new])
                        except Exception:
                            got_txt = exp_txt = "<decode error>"
                        logger.info(
                            f"  [cursor] mismatch got_ids={new_segment} got_text={got_txt}"
                        )
                        logger.info(
                            f"           expected_ids={st.predicted_tokens[st.pred_cursor:st.pred_cursor + n_new]} expected_text={exp_txt}"
                        )
            if ok:
                st.pred_cursor += len(new_segment)
            else:
                st.pred_cursor = -1
                # Allow an immediate alignment attempt on this call.
                st.align_blocked_until_completed_lines = len(
                    st.ctx_line_tuples) - 1

        # 2) If lost, try sparse alignment only when a new line completes
        if st.pred_cursor < 0:
            can_align = len(st.ctx_line_tuples) > st.align_blocked_until_completed_lines
            if can_align:
                if VERBOSE and st.ctx_line_tuples:
                    last_line = st.ctx_line_tuples[-1]
                    try:
                        last_line_text = self._tokenizer.decode(list(last_line))
                    except Exception:
                        last_line_text = "<decode error>"
                    logger.info(
                        f"  [align] last completed line text: {last_line_text}")
                start = time.time()
                line_cursor = _align_cursor_lines(st.predicted_line_tuples,
                                                  completed)
                logger.info(
                    f"  _align_cursor_lines: {(time.time() - start) * 1000:0.2f}ms")

                if line_cursor is not None and line_cursor < len(
                        st.predicted_line_tuples):
                    pred_line_start = st.line_starts[line_cursor]

                    # Build predicted line tokens for the current line.
                    pred_line_tokens: List[int] = []
                    idx = pred_line_start
                    while (idx < len(st.predicted_tokens)
                           and st.predicted_tokens[idx] not in st.newline_set):
                        pred_line_tokens.append(st.predicted_tokens[idx])
                        idx += 1

                    # Verify current partial line matches prefix (string-wise)
                    try:
                        cur_txt = self._tokenizer.decode(list(current_prefix))
                        pred_line_txt = self._tokenizer.decode(pred_line_tokens)
                    except Exception:
                        cur_txt = pred_line_txt = "<decode error>"
                    if pred_line_txt.startswith(cur_txt):
                        # Advance cursor by the number of tokens in the
                        # current prefix (token-level position)
                        st.pred_cursor = pred_line_start + len(current_prefix)
                        if VERBOSE:
                            logger.info(
                                "  [align] aligned; partial-line prefix matches predicted line start"
                            )
                    else:
                        # Block further attempts until another newline
                        st.align_blocked_until_completed_lines = len(
                            st.ctx_line_tuples)
                        if VERBOSE:
                            logger.info(
                                "  [align] aligned; partial-line prefix does NOT match predicted line start"
                            )
                else:
                    # Alignment failed – block until another completed line
                    st.align_blocked_until_completed_lines = len(
                        st.ctx_line_tuples)
                    if VERBOSE:
                        logger.info("  [align] no valid alignment")

        # If aligned, produce next k tokens
        if st.pred_cursor >= 0:
            end = min(st.pred_cursor + self.k, len(st.predicted_tokens))
            if end > st.pred_cursor:
                arr = np.array(st.predicted_tokens[st.pred_cursor:end],
                               dtype=np.int32)
            else:
                arr = None

        if VERBOSE:
            if arr is not None and arr.size > 0:
                try:
                    pred_txt = self._tokenizer.decode(arr.tolist())
                except Exception:
                    pred_txt = "<decode error>"
                logger.info(
                    f"  [predict] ids={arr.tolist()} text={pred_txt}")
            else:
                logger.info("  [predict] (no match, no prediction)")
            logger.info(
                f"  [done] proposed={0 if arr is None else arr.size} tokens in {(time.time() - propose_start) * 1000:0.2f}ms"
            )

        return arr if (arr is not None and arr.size > 0) else None

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

    # --------------------------- line ending helpers ---------------------------

    @staticmethod
    def _detect_line_ending_mode_from_text(text: str) -> str:
        """Return 'crlf' if text contains any "\r\n", otherwise 'lf'."""
        return "crlf" if "\r\n" in text else "lf"

    @staticmethod
    def _convert_line_endings(text: str, to_mode: str) -> str:
        """Convert all line endings in text to LF or CRLF.

        First normalizes to LF by replacing CRLF and lone CR with LF, then
        converts to the target mode.
        """
        # Normalize to LF
        norm = text.replace("\r\n", "\n").replace("\r", "\n")
        if to_mode == "lf":
            return norm
        if to_mode == "crlf":
            return norm.replace("\n", "\r\n")
        return norm

    def _reindex_prediction(self, st: _ReqState) -> None:
        """Recompute derived structures from st.predicted_tokens."""
        st.predicted_line_tuples = _split_by_newline_tokens(
            st.predicted_tokens, st.newline_set)
        st.line_starts = [0]
        for idx, tok in enumerate(st.predicted_tokens):
            if tok in st.newline_set:
                st.line_starts.append(idx + 1)

    # ------------------------------------------------------------------ draft model API compatibility

    def load_model(self, *args, **kwargs):  # noqa: D401 – interface stub
        # StaticTextProposer is not a model – nothing to load.
        pass

    # --------------------------- lifecycle hooks ------------------------------

    def finish_requests(self, req_ids: list[str] | set[str] | tuple[str, ...]):
        """Drop per-request state for finished/aborted requests."""
        for rid in req_ids:
            self._state.pop(rid, None)
