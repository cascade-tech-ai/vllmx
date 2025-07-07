"""Line-based predictor using C-accelerated ``difflib.SequenceMatcher``.

Purpose
-------
Provide an *LCS-based* alignment without relying on *rapidfuzz* while keeping
performance high.  Python's standard library ships a highly optimised C
implementation of the Ratcliff–Obershelp pattern matcher in
``difflib.SequenceMatcher`` which effectively computes matching blocks – good
enough to derive the cursor for our speculative predictor.

Algorithm is identical to :pyclass:`~smart_prediction.algorithms.fuzz_lines.RapidFuzzLinePredictor`
except that the *edit operations* are obtained from ``SequenceMatcher``.
"""

from __future__ import annotations

import difflib
from typing import List, Tuple, Optional, Set

from .base import PredictorBase

# Re-use helper from fuzz_lines.py -------------------------------------------------------


def _split_by_newline_tokens(tokens: List[int], newline_set: Set[int]) -> List[Tuple[int, ...]]:
    """Split *tokens* into immutable line tuples using *newline_set* markers."""

    lines: List[List[int]] = [[]]
    for tok in tokens:
        lines[-1].append(tok)
        if tok in newline_set:
            lines.append([])

    return [tuple(line) for line in lines]


# Internal alignment helper -------------------------------------------------------------


def _sm_align_cursor_lines(a: List[Tuple[int, ...]], b: List[Tuple[int, ...]]) -> Optional[int]:
    """Return *x* (lines consumed in *a*) that aligns entire *b* via SequenceMatcher.

    The function mirrors the cursor logic used by the rapidfuzz variant: we
    replay the opcodes until the *entire* *b* sequence has been consumed.  We
    additionally verify that the *last* consumed line actually matched (i.e.
    came from an 'equal' opcode) so that we do not stop at an insertion.
    """

    m = len(b)
    if m == 0:
        return 0

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    x = 0
    y = 0
    best_x: Optional[int] = None

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # length of equal block
            block_len = i2 - i1
            # how many of those equal elements are still part of *b*?
            take = min(block_len, m - y)

            x += take
            y += take

            if y == m:
                best_x = x  # candidate – keep searching for rightmost

            # If we consumed only part of the block (because *b* ended), we
            # break here – remaining part irrelevant.
            if take < block_len:
                break

        elif tag == "delete":
            x += i2 - i1
        elif tag == "insert":
            y += j2 - j1
            if y >= m:
                # cannot finish on insertion; ignore candidate
                return best_x
        else:  # replace
            x += i2 - i1
            y += j2 - j1
            if y >= m:
                return best_x

    # Attempt to use the final x if still aligned.
    if y == m and x > 0 and a[x - 1] == b[-1]:
        # Prefer the *rightmost* occurrence of the last matched line in *a*.
        last_line = b[-1]
        candidate = x
        for i in range(x, len(a)):
            if a[i] == last_line:
                candidate = i + 1
        return candidate

    return best_x


# Predictor -----------------------------------------------------------------------------


class SequenceMatcherLinePredictor(PredictorBase):
    """LCS-based predictor without rapidfuzz – uses difflib.SequenceMatcher."""

    def __init__(
        self,
        predicted_tokens: List[int],
        *,
        newline_token: int,
        tokenizer=None,
    ) -> None:

        super().__init__(predicted_tokens)

        # Discover newline tokens.
        newline_set: Set[int] = {newline_token}
        if tokenizer is not None:
            seen: Set[int] = set()
            for tok in predicted_tokens:
                if tok in seen or tok in newline_set:
                    continue
                seen.add(tok)
                if "\n" in tokenizer.decode([tok]):
                    newline_set.add(tok)

        self._newline_set = newline_set

        # Precompute line tuples and line→token index mapping.
        self._predicted_line_tuples = _split_by_newline_tokens(predicted_tokens, newline_set)

        self._line_starts: List[int] = [0]
        for idx, tok in enumerate(predicted_tokens):
            if tok in newline_set:
                self._line_starts.append(idx + 1)

        # No per-request mutable state kept on *self* – the incremental
        # parsing buffers live in the ``state`` dictionary passed to
        # :pymeth:`propose`.

    # ------------------------------------------------------------------ PredictorBase
    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        if not context_tokens:
            return []

        # 1. Incrementally update context state (stored in *state*).
        ctx_processed: int = state.get("ctx_processed", 0)
        ctx_line_tuples: List[Tuple[int, ...]] = state.setdefault("ctx_line_tuples", [])
        current_line_tokens: List[int] = state.setdefault("current_line_tokens", [])

        if len(context_tokens) < ctx_processed:
            # Context reset => clear buffers.
            ctx_processed = 0
            ctx_line_tuples.clear()
            current_line_tokens.clear()

        for tok in context_tokens[ctx_processed:]:
            current_line_tokens.append(tok)
            if tok in self._newline_set:
                ctx_line_tuples.append(tuple(current_line_tokens))
                current_line_tokens.clear()

        ctx_processed = len(context_tokens)
        state["ctx_processed"] = ctx_processed

        last_token_is_nl = (
            len(context_tokens) > 0 and context_tokens[-1] in self._newline_set
        )

        if last_token_is_nl:
            completed_line_tuples = ctx_line_tuples + [tuple()]
            current_line_prefix: List[int] = []
        else:
            completed_line_tuples = ctx_line_tuples
            current_line_prefix = list(current_line_tokens)

        # 2. Align via SequenceMatcher.
        line_cursor = _sm_align_cursor_lines(self._predicted_line_tuples, completed_line_tuples)
        if line_cursor is None or line_cursor >= len(self._predicted_line_tuples):
            return []

        # 3. current line prefix check.
        predicted_line_start_idx = self._line_starts[line_cursor]

        predicted_line_tokens: List[int] = []
        idx = predicted_line_start_idx
        while idx < len(self.predicted_tokens) and self.predicted_tokens[idx] not in self._newline_set:
            predicted_line_tokens.append(self.predicted_tokens[idx])
            idx += 1

        if current_line_prefix != predicted_line_tokens[: len(current_line_prefix)]:
            return []

        token_cursor = predicted_line_start_idx + len(current_line_prefix)
        if token_cursor >= len(self.predicted_tokens):
            return []

        end = min(token_cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[token_cursor:end]
