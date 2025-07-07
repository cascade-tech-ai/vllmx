"""Streaming predictor using Myers diff algorithm on token sequences.

This predictor treats both *prediction* and *already generated context* as
plain **token sequences** (no special line handling) and employs the classical
*Myers O((N+M)·D)* shortest-edit-script algorithm to *align* the two
sequences.  Given the current `context_tokens` emitted by the language model
it computes the *minimal* edit path from the beginning of the *prediction*
(`predicted_tokens`) to the point **at which the context ends**.  The X-
coordinate of that path inside the edit graph tells us how many tokens of the
prediction have been *consumed* so far.  The predictor can therefore propose
the *next* tokens that follow this alignment.

The implementation recomputes the alignment **from scratch** every time
`propose()` is called.  Although this is not asymptotically optimal, the test
inputs used in the prototype are tiny (≪ 10 k tokens) so the additional cost
is negligible (< 1 ms on a laptop).  A future optimisation could keep the *V*
array across calls and extend it incrementally, yielding an *online Myers*
algorithm – for now correctness and simplicity are preferred.
"""

from __future__ import annotations

from typing import List, Dict

from .base import PredictorBase

# ---------------------------------------------------------------------------
# Predictor implementation
# ---------------------------------------------------------------------------


class MyersStreamingPredictor(PredictorBase):
    """Predictor that aligns tokens via Myers shortest-edit-script diff."""

    # We do *not* maintain internal state between calls – the alignment is
    # re-computed each round.  This keeps the code small and avoids having to
    # implement incremental frontier updates for now.

    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        # If we do not have any context yet we cannot align reliably.
        if not context_tokens:
            return []

        # All tokens already generated – nothing left to predict.
        if len(context_tokens) >= len(self.predicted_tokens):
            return []

        cursor = _myers_align_cursor(self.predicted_tokens, context_tokens)
        if cursor is None:
            return []  # Could not find a path that consumes the entire context.

        if cursor >= len(self.predicted_tokens):
            return []  # Prediction exhausted – no continuation available.

        end = min(cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[cursor:end]


# ---------------------------------------------------------------------------
# Internal helpers – Myers diff
# ---------------------------------------------------------------------------


def _myers_align_cursor(a: List[int], b: List[int]) -> int | None:
    """Return *x* coordinate after aligning *b* (context) against *a*.

    Parameters
    ----------
    a
        The *predicted* token sequence (full, immutable).
    b
        The *already generated* tokens produced by the LM so far (grows over
        time).  We wish to know how many tokens of *a* have been *matched/
        skipped* once *b* has been consumed, under the *shortest-edit-script*
        criterion.  The returned integer therefore serves as the **cursor**
        into *a* from which to propose further tokens.

    Returns
    -------
    int | None
        The position in *a* after the alignment **or** `None` if no alignment
        is possible (e.g. because *b* contains tokens that never appear in
        *a*).
    """

    n = len(a)
    m = len(b)

    # Trivial cases – acceleration for small inputs.
    if m == 0:
        return 0  # nothing consumed yet

    # The Myers algorithm keeps a frontier vector V that maps *diagonal index*
    # `k = x - y` to the *furthest* x reached along that diagonal for a given
    # `D = #edits`.
    #
    # We store V as a dict to avoid converting k to list indices with offset.

    max_d = n + m
    v: Dict[int, int] = {0: 0}

    # We track, for each edit distance *d*, the *best* x that achieves y == m.
    # We postpone the early return until we have processed **all** k on the
    # current *d*-layer so that we can choose the maximum x among candidates.

    for d in range(0, max_d + 1):
        best_x_for_full_y = None  # candidate cursor for this d if any

        for k in range(-d, d + 1, 2):
            if k == -d:
                x_start = v.get(k + 1, 0)  # insertion into *a*
            elif k == d:
                x_start = v.get(k - 1, 0) + 1  # deletion from *a*
            else:
                # Prefer the branch that yields the **larger** x because that
                # corresponds to the furthest reach along the edit graph (see
                # Myers' paper, section 2.2).
                prev_k_minus = v.get(k - 1, 0) + 1
                prev_k_plus = v.get(k + 1, 0)
                if prev_k_minus > prev_k_plus:
                    x_start = prev_k_minus  # deletion (move right)
                else:
                    x_start = prev_k_plus  # insertion (move down)

            y_start = x_start - k

            # Extend *snake* (diagonal) while tokens match.
            x = x_start
            y = y_start
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1

            v[k] = x  # Update frontier for this k at edit distance d.

            # If this diagonal reached the bottom of the context (y == m),
            # remember the *farthest* x seen; we will consider it after the
            # inner loop finishes so that all diagonals are explored for the
            # current edit distance `d`.
            if y >= m:
                if best_x_for_full_y is None or x > best_x_for_full_y:
                    best_x_for_full_y = x

        # After exploring all k for this *d*, decide whether we found a valid
        # alignment for the entire context.  We deliberately take the *max x*
        # so that we advance as far as possible along the prediction while
        # still incurring the minimal number of edits.

        if best_x_for_full_y is not None:
            if m == 0:
                return best_x_for_full_y

            last_a_idx = best_x_for_full_y - 1
            last_b_idx = m - 1

            if 0 <= last_a_idx < n and a[last_a_idx] == b[last_b_idx]:
                return best_x_for_full_y
            # Otherwise abstain (context ended on insertion) – continue to
            # next edit distance hoping for alternative path.

    # Should not reach here – unless *b* contains tokens not present in *a*
    # at all, in which case there is no finite edit script.
    return None
