"""Streaming predictor that aligns **lines** using Myers shortest-edit script.

This variant performs Myers diff on the *sequence of lines* rather than on
individual tokens.  It therefore enjoys the *robust re-anchoring* of Myers
while remaining insensitive to changes *within* a line (e.g. variable
renames) – an attractive middle-ground between the pure token-level Myers and
the heuristic `LineDiffPredictor`.

Design choices
--------------
*  The alignment is recomputed **from scratch** at every `propose()` call – the
   input sizes in the prototype are tiny, so the O((N+M)·D) cost is
   negligible.
*  We operate on lists of *strings* (each string = one line) produced by the
   helper `_tokens_to_lines()` which mirrors the implementation inside
   `LineDiffPredictor`.
*  After finding the cursor (number of *predicted* lines already aligned) we
   translate it back to a *token index* via the pre-computed `_line_starts` so
   that we can return the next `max_lookahead` tokens.
"""

from __future__ import annotations

from typing import List, Dict

from .base import PredictorBase


# ---------------------------------------------------------------------------
# Predictor implementation
# ---------------------------------------------------------------------------


class MyersLinePredictor(PredictorBase):
    """Line-based predictor using Myers diff for alignment."""

    def __init__(self, predicted_tokens: List[int], newline_token: int):  # noqa: D401 simple
        super().__init__(predicted_tokens)

        self._newline_token = newline_token

        # Convert *prediction* into line list once so that we do not repeat
        # work on every call.
        self._predicted_lines: List[str] = self._tokens_to_lines(predicted_tokens)

        # Map *line index* -> *token index* for quick conversion back.
        self._line_starts: List[int] = [0]
        for idx, tok in enumerate(predicted_tokens):
            if tok == newline_token:
                self._line_starts.append(idx + 1)

    # ------------------------------------------------------------------
    # PredictorBase
    # ------------------------------------------------------------------
    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        if not context_tokens:
            return []

        # Fast-path: context longer than prediction ⇒ nothing to predict.
        if len(context_tokens) >= len(self.predicted_tokens):
            return []

        context_lines = self._tokens_to_lines(context_tokens)

        line_cursor = _myers_align_cursor_lines(self._predicted_lines, context_lines)
        if line_cursor is None:
            return []  # Cannot align – abstain.

        if line_cursor >= len(self._predicted_lines):
            return []

        token_cursor = self._line_starts[line_cursor]
        end = min(token_cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[token_cursor:end]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tokens_to_lines(self, tokens: List[int]) -> List[str]:
        """Convert a token list into list[str] where each string is a *line*."""
        lines: List[List[str]] = [[]]
        for tok in tokens:
            if tok == self._newline_token:
                lines.append([])
            else:
                lines[-1].append(str(tok))
        return [" ".join(parts) for parts in lines]


# ---------------------------------------------------------------------------
# Internal – Myers on line sequences
# ---------------------------------------------------------------------------


def _myers_align_cursor_lines(a: List[str], b: List[str]) -> int | None:
    """Return *x* (lines consumed in *a*) that aligns full *b* via Myers.

    Parameters
    ----------
    a
        List of **predicted lines**.
    b
        List of **already generated lines** (context).
    """

    n = len(a)
    m = len(b)

    if m == 0:
        return 0

    max_d = n + m
    v: Dict[int, int] = {0: 0}

    for d in range(0, max_d + 1):
        best_x_full_y = None

        for k in range(-d, d + 1, 2):
            if k == -d:
                x_start = v.get(k + 1, 0)  # insertion (down)
            elif k == d:
                x_start = v.get(k - 1, 0) + 1  # deletion (right)
            else:
                prev_del = v.get(k - 1, 0) + 1  # deletion
                prev_ins = v.get(k + 1, 0)      # insertion
                x_start = prev_del if prev_del > prev_ins else prev_ins

            y_start = x_start - k

            # Extend snake while lines match exactly.
            x = x_start
            y = y_start
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1

            v[k] = x

            if y >= m:
                if best_x_full_y is None or x > best_x_full_y:
                    best_x_full_y = x

        if best_x_full_y is not None:
            # Verify last line matched, not insertion.
            last_a_idx = best_x_full_y - 1
            last_b_idx = m - 1
            if 0 <= last_a_idx < n and a[last_a_idx] == b[last_b_idx]:
                return best_x_full_y
            # else: continue searching with larger d

    return None
