"""Predictor that realigns by *token-suffix* rather than entire lines.

The algorithm looks at the last *N* tokens of the already generated context
(`context_tokens`) and tries to find **any** suffix (down to length 1) that
occurs inside the *predicted* token sequence.  It picks the *longest* such
suffix and then proposes the following tokens.

This heuristic is considerably more robust than the simple `LineDiffPredictor`
for situations where **individual tokens** within a line change (e.g. variable
renames) because it does not insist on matching the entire line verbatim.

Complexity:  O(K·M) in the worst case where K = search window (≤ `max_suffix`)
and M = length of the prediction.  With the small inputs used in the unit
tests this is negligible (< 0.1 ms per call).
"""

from __future__ import annotations

from typing import List

from .base import PredictorBase

# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class GreedySuffixPredictor(PredictorBase):
    """Realign by longest *token-level* suffix found in the prediction."""

    def __init__(self, predicted_tokens: List[int], max_suffix: int = 32):
        super().__init__(predicted_tokens)
        self._max_suffix = max_suffix

    # ------------------------------------------------------------------
    # PredictorBase API
    # ------------------------------------------------------------------
    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        # Nothing left to predict.
        if len(context_tokens) >= len(self.predicted_tokens):
            return []

        # Determine the longest suffix (≤ _max_suffix) of `context_tokens`
        # that appears inside `predicted_tokens`.
        suffix_len = min(len(context_tokens), self._max_suffix)
        best_pos = None  # start idx inside predicted_tokens
        best_len = 0

        while suffix_len > 0 and best_pos is None:
            suffix = context_tokens[-suffix_len:]
            pos = _find_subseq(self.predicted_tokens, suffix)
            if pos is not None:
                best_pos = pos
                best_len = suffix_len
                break  # we search from longest → shortest, so first hit is best
            suffix_len -= 1

        if best_pos is None:
            return []  # cannot align – abstain

        token_cursor = best_pos + best_len
        if token_cursor >= len(self.predicted_tokens):
            return []

        end = min(token_cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[token_cursor:end]


# ---------------------------------------------------------------------------
# Utility helper (duplicated to avoid circular import)
# ---------------------------------------------------------------------------


def _find_subseq(haystack: List[int], needle: List[int]) -> int | None:
    """Return the *start index* of *needle* inside *haystack* or *None*."""

    if len(needle) == 0 or len(needle) > len(haystack):
        return None

    first = needle[0]
    max_start = len(haystack) - len(needle)
    for idx in range(0, max_start + 1):
        if haystack[idx] != first:
            continue
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None
