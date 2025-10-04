"""Speculative static‑text proposer (v1).

Static‑text speculative‑decoding logic used by vLLM.

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

Implementation uses pure Python on CPU.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple, Set, Sequence, TYPE_CHECKING

import difflib

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

VERBOSE = os.getenv('VLLM_PREDICTED_OUTPUTS_VERBOSE') == '1'

logger = init_logger(__name__)
logger.setLevel("INFO" if VERBOSE else "WARNING")

# Optional high-performance diff library rapidfuzz.

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
        "predicted_line_tuples",
        # incremental context tracking
        "ctx_processed",
        "ctx_line_tuples",
        "current_line_tokens",
        # line ending mode: 'lf' or 'crlf'
        "line_ending_mode",
        # cursor-based fast path
        "pred_cursor",
        "failed_alignment_this_line",
        # optional debug fields
    )

    def __init__(self, predicted_tokens: List[int], newline_set: Set[int]):
        self.predicted_tokens: List[int] = predicted_tokens

        # Pre-compute line tuples & mapping.
        self.predicted_line_tuples: List[Tuple[int, ...]] = []

        # Context-tracking buffers.
        self.ctx_processed = 0
        self.ctx_line_tuples: List[Tuple[int, ...]] = []
        self.current_line_tokens: List[int] = []

        # Cursor state
        self.pred_cursor: int = 0  # >=0 aligned at this token index; -1 lost
        self.failed_alignment_this_line: bool = False

        # Line endings — initialized by caller (proposer) after decoding text
        self.line_ending_mode: str = "lf"

        # (no additional debug state)

        self._reindex_prediction(newline_set)

    def _reindex_prediction(self, newline_set: Set[int]) -> None:
        """Recompute derived structures from predicted tokens."""
        self.predicted_line_tuples = _split_by_newline_tokens(
            self.predicted_tokens, newline_set)


def _split_by_newline_tokens(tokens: Sequence[int],
                             newline_set: Set[int]) -> List[Tuple[int, ...]]:
    """Return immutable *lines* where each element is a tuple of tokens."""

    lines: List[List[int]] = [[]]
    for tok in tokens:
        token = int(tok)
        lines[-1].append(token)
        if token in newline_set:
            lines.append([])

    return [tuple(line) for line in lines]


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


class StaticTextProposer:

    def __init__(self, vllm_config: VllmConfig):
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.vllm_config = vllm_config

        # Tokeniser and model-wide newline set (both initialised lazily the
        # first time we need them).
        self._tokenizer = None
        self._newline_set: Set[int] | None = None

        # req_id → _ReqState
        self._state: Dict[str, _ReqState] = {}

        logger.info("Initialized StaticTextProposer")

    def generate_drafts(
        self,
        input_batch: "InputBatch",
        requests: Dict[str, "CachedRequestState"],
        sampled_token_ids: List[List[int]],
    ) -> List[List[int]]:
        """Return speculative drafts for each request in the active batch."""

        if not sampled_token_ids:
            return []

        draft_token_ids: List[List[int]] = []
        req_ids = input_batch.req_ids

        for idx, sampled_ids in enumerate(sampled_token_ids):
            if idx >= len(req_ids):
                draft_token_ids.append([])
                continue

            if not sampled_ids:
                draft_token_ids.append([])
                continue

            req_id = req_ids[idx]
            req_state = requests.get(req_id)
            if req_state is None:
                draft_token_ids.append([])
                continue

            pred_params = req_state.sampling_params.predicted_outputs
            if pred_params is None or not pred_params.has_prediction():
                draft_token_ids.append([])
                continue

            pred_tokens = pred_params.predicted_token_ids
            if pred_tokens is None:
                logger.debug("Request %s missing pre-tokenised prediction", req_id)
                draft_token_ids.append([])
                continue

            context_ids = req_state.output_token_ids or sampled_ids

            proposed = self.propose(
                req_id=req_id,
                context_token_ids=context_ids,
                predicted_token_ids=list(pred_tokens),
            )

            draft_token_ids.append([] if proposed is None else proposed)

        return draft_token_ids

    def propose(
        self,
        req_id: str,
        context_token_ids: Sequence[int],
        predicted_token_ids: Optional[List[int]],
    ) -> Optional[List[int]]:
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

        newline_set = self._get_newline_set()

        # Initialise request state on first call.
        if req_id not in self._state:
            start = time.time()
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
                    mode = "crlf" if "\r\n" in pred_text else "lf"

                except Exception:
                    mode = "lf"
                self._state[req_id].line_ending_mode = mode
                if VERBOSE:
                    logger.info(f"  [le] init mode={mode}")

        st = self._state[req_id]

        ctx_tokens = list(context_token_ids)

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

        # Update incremental context buffers.
        for tok in ctx_tokens[st.ctx_processed:]:
            st.current_line_tokens.append(tok)

            if tok in newline_set:
                st.ctx_line_tuples.append(tuple(st.current_line_tokens))
                st.current_line_tokens.clear()
                st.failed_alignment_this_line = False
                self._check_switch_line_endings(st, ctx_tokens, new_segment)

        st.ctx_processed = len(ctx_tokens)

        last_is_nl = bool(ctx_tokens) and (ctx_tokens[-1] in newline_set)

        completed = st.ctx_line_tuples
        current_prefix: List[int] = [] if last_is_nl else list(
            st.current_line_tokens)

        proposal: Optional[List[int]] = None

        # 1) Fast-path cursor advance if still aligned
        if st.pred_cursor >= 0 and new_segment:
            end_idx = st.pred_cursor + len(new_segment)
            if (end_idx <= len(st.predicted_tokens)
                    and st.predicted_tokens[st.pred_cursor:end_idx] == new_segment):
                st.pred_cursor = end_idx
            else:
                st.pred_cursor = -1

        # 2) If lost, try sparse alignment only when allowed
        if st.pred_cursor < 0 and st.ctx_line_tuples and not st.failed_alignment_this_line:
            if VERBOSE:
                last_line = st.ctx_line_tuples[-1]
                try:
                    last_line_text = self._tokenizer.decode(list(last_line))
                except Exception:
                    last_line_text = "<decode error>"
                logger.info(
                    f"  [align] last completed line text: {last_line_text}")
            start = time.time()
            line_cursor = _align_cursor_lines(st.predicted_line_tuples, completed)
            logger.info(
                f"  _align_cursor_lines: {(time.time() - start) * 1000:0.2f}ms")

            if line_cursor is None or line_cursor >= len(st.predicted_line_tuples):
                st.failed_alignment_this_line = True
            else:
                pred_line_tuple = st.predicted_line_tuples[line_cursor]
                predicted_line_body = [tok for tok in pred_line_tuple
                                       if tok not in newline_set]
                prefix_len = len(current_prefix)
                if (prefix_len <= len(predicted_line_body)
                        and predicted_line_body[:prefix_len] == current_prefix):
                    pred_line_start = sum(
                        len(line)
                        for line in st.predicted_line_tuples[:line_cursor])
                    st.pred_cursor = pred_line_start + prefix_len
                else:
                    st.failed_alignment_this_line = True

        # If aligned, produce next k tokens
        if st.pred_cursor >= 0:
            end = min(st.pred_cursor + self.k, len(st.predicted_tokens))
            if end > st.pred_cursor:
                proposal = st.predicted_tokens[st.pred_cursor:end]

        if VERBOSE:
            if proposal:
                try:
                    pred_txt = self._tokenizer.decode(proposal)
                except Exception:
                    pred_txt = "<decode error>"
                logger.info(
                    f"  [predict] ids={proposal} text={pred_txt}")
            else:
                logger.info("  [predict] (no match, no prediction)")
            logger.info(
                f"  [done] proposed={0 if not proposal else len(proposal)} tokens in {(time.time() - propose_start) * 1000:0.2f}ms"
            )

        return proposal if proposal else None


    def _get_newline_set(self) -> Set[int]:
        """Return *global* set of token ids whose decoded form contains "\n"."""

        if self._newline_set is not None:
            return self._newline_set

        start = time.time()
        tok = self._tokenizer
        newline_set: Set[int] = set()

        # for tid in tok.get_vocab().values():
        #     token_id = int(tid)
        #     if "\n" in tok.decode([token_id]):
        #         newline_set.add(token_id)

        try:
            vocab_size = tok.vocab_size  # type: ignore[attrdefined]
        except AttributeError:
            vocab_size = len(tok.get_vocab())  # type: ignore[argtype]

        for tid in range(vocab_size):
            if "\n" in tok.decode([tid]):
                newline_set.add(tid)

        if not newline_set:
            logger.error("No newline tokens found in tokenizer.")
            enc = tok.encode("\n", add_special_tokens=False)
            newline_set.add(enc[0])

        logger.info(f"  _detect_newline_tokens: {(time.time() - start) * 1000:.2f}")

        self._newline_set = newline_set
        return newline_set

    def _check_switch_line_endings(self, st: _ReqState, ctx_tokens: List[int],
                                   new_segment: List[int]) -> None:
        """Detect line-ending changes in the context and retokenize if needed."""

        if not new_segment or self._newline_set is None:
            return

        if not any(tok in self._newline_set for tok in new_segment):
            return

        tail_text = self._tokenizer.decode(ctx_tokens[-3:])

        last_nl = tail_text.rfind("\n")
        if last_nl == -1:
            return

        new_mode = "crlf" if (last_nl > 0 and tail_text[last_nl - 1] == "\r") else "lf"
        if new_mode == st.line_ending_mode:
            return

        # convert prediction text to new line endings
        cur_pred_text = self._tokenizer.decode(st.predicted_tokens)
        normalized = cur_pred_text.replace("\r\n", "\n").replace("\r", "\n")
        switched_text = (normalized if new_mode == "lf" else
                         normalized.replace("\n", "\r\n"))
        new_pred_tokens = self._tokenizer.encode(
            switched_text, add_special_tokens=False)
        st.predicted_tokens = list(new_pred_tokens)
        st._reindex_prediction(self._newline_set)
        st.line_ending_mode = new_mode

        # reset cursor
        st.pred_cursor = -1
        st.failed_alignment_this_line = False

        if VERBOSE:
            logger.info(
                f"  [le] switch {st.line_ending_mode} -> {new_mode}; "
                f"retokenized {len(st.predicted_tokens)} -> {len(new_pred_tokens)} tokens")

    def load_model(self, *args, **kwargs):  # noqa: D401 – interface stub
        # StaticTextProposer is not a model – nothing to load.
        pass

    def finish_requests(self, req_ids: list[str] | set[str] | tuple[str, ...]):
        """Drop per-request state for finished/aborted requests."""
        for rid in req_ids:
            if VERBOSE:
                logger.info(f"clearing finished request: {rid}")
            self._state.pop(rid, None)
