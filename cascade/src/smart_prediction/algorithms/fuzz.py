"""Predictor based on *rapidfuzz* Levenshtein edit distance.

The algorithm parallels :pyclass:`MyersStreamingPredictor` but replaces the
pure-Python Myers diff with a **C++/Rust accelerated** computation provided by
``rapidfuzz``.

Given the *prediction* sequence **A** and the already generated *context*
tokens **B′** the predictor:

1.  Computes a *minimal* edit script converting A → B′ via
    ``Levenshtein.editops``.
2.  Replays the script until the *entire* context is consumed.  The resulting
    X-coordinate (cursor) tells us how many tokens of A have already been
    matched/ skipped.
3.  Returns up to ``max_lookahead`` tokens from that cursor as speculation.

If alignment is impossible (e.g. B′ ends with a token that never occurs next
in A) the predictor abstains and returns an empty list.
"""

from __future__ import annotations

from typing import List, Optional

# ---------------------------------------------------------------------------
# Select diff algorithm (Levenshtein vs LCS) – default = LCSseq for lines.
# ---------------------------------------------------------------------------

import os

from rapidfuzz.distance import Levenshtein, LCSseq

_DIFF_IMPL = LCSseq  # better cursor placement for our predictors

# Allow override via environment variable: SMART_DIFF_ALGO=lev or lcs
_override = os.getenv("SMART_DIFF_ALGO")
if _override is not None:
    if _override.lower().startswith("lev"):
        _DIFF_IMPL = Levenshtein
    elif _override.lower().startswith("lcs"):
        _DIFF_IMPL = LCSseq

from .base import PredictorBase

# ---------------------------------------------------------------------------
# Internal helper – cursor calculation
# ---------------------------------------------------------------------------


def _rf_align_cursor(a: List[int], b: List[int]) -> Optional[int]:
    """Return *x* position in *a* after aligning *b* (prefix) or *None*.

    The function is a token-level equivalent of the helper used by
    ``myers_predict.py`` but leverages the accelerated *rapidfuzz* backend.
    """

    m = len(b)
    if m == 0:
        return 0

    # Obtain a minimal edit script converting A → B′.
    ops = _DIFF_IMPL.editops(a, b)

    x = 0
    y = 0

    for op in ops:
        # Advance across equal region (snake).
        while x < op.src_pos and y < op.dest_pos and y < m:
            x += 1
            y += 1

        if y >= m:
            break

        # Apply the edit operation itself.
        tag = op.tag  # 'insert', 'delete', 'replace'
        if tag == "delete":
            x += 1
        elif tag == "insert":
            y += 1
        else:  # replace
            x += 1
            y += 1

        if y >= m:
            break

    # Consume trailing snake if ops ended early.
    while y < m and x < len(a) and a[x] == b[y]:
        x += 1
        y += 1

    if y == m and x > 0 and a[x - 1] == b[-1]:
        return x

    return None  # alignment impossible under our rules


# ---------------------------------------------------------------------------
# Predictor implementation
# ---------------------------------------------------------------------------


class RapidFuzzPredictor(PredictorBase):
    """Stateless predictor using *rapidfuzz* alignment each call."""

    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:  # noqa: D401 simple
        # 1. No context yet → no speculation.
        if not context_tokens:
            return []

        # 2. Prediction already fully consumed.
        if len(context_tokens) >= len(self.predicted_tokens):
            return []

        cursor = _rf_align_cursor(self.predicted_tokens, context_tokens)
        if cursor is None or cursor >= len(self.predicted_tokens):
            return []

        end = min(cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[cursor:end]
