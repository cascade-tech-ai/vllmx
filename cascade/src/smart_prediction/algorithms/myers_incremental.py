"""Incremental predictor based on the *streaming* Myers implementation.

This variant keeps a single :class:`joev.src.streaming_myers.StreamingMyers`
instance alive for the whole simulation.

Workflow per `propose()` call:

1.  Determine **new** portion of `context_tokens` (everything since the last
    invocation) and *feed* it into the streaming Myers object.  This updates
    the search frontiers **incrementally** – no work from previous calls is
    repeated.

2.  After the forward update we inspect the *current frontier* `V_D` (for the
    optimal edit distance *D*) to find the cell that sits at
    ``y = len(context_tokens)``.  Its X-coordinate `x*` tells us how many
    tokens of *A* (the *prediction*) have been matched/ skipped so far.

3.  Every token ``A[x*]`` that lies *to the right* of that cell would be a
    **deletion** if the diff ended here – which is exactly the continuation
    we want to predict.  We therefore return the next
    ``min(max_lookahead, |A| - x*)`` tokens from *A*.

Compared to the existing `myers_streaming.py` predictor this avoids a complete
O((N+M)·D) recomputation for every call and therefore scales to much longer
documents.
"""

from __future__ import annotations

from typing import List

from joev.src.streaming_myers import StreamingMyers

# We re-use the robust cursor-alignment helper from the existing stateless
# predictor instead of re-implementing the delicate corner-cases again.
from .myers_streaming import _myers_align_cursor

from .base import PredictorBase


class MyersIncrementalPredictor(PredictorBase):
    """Incremental, stateful Myers predictor."""

    def __init__(self, predicted_tokens: List[int]):
        super().__init__(predicted_tokens)

        # Streaming Myers instance initialised with *A* (= predicted tokens).
        self._sm = StreamingMyers(predicted_tokens)

        # Remember how many context tokens have already been forwarded so we
        # can feed only the *delta* next time.
        self._consumed: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        # 0. Nothing to align → make no guess.
        if not context_tokens:
            return []

        # 1. Feed *new* tokens (if any) into the incremental diff.
        if len(context_tokens) > self._consumed:
            new_chunk = context_tokens[self._consumed :]
            self._sm.feed(new_chunk)
            self._consumed = len(context_tokens)

        # 2. Determine current cursor position x* inside A.

        # ------------------------------------------------------------------
        # 2. Compute cursor *robustly* via the proven helper (still cheap –
        #    input sizes for smart-prediction are small).
        # ------------------------------------------------------------------

        cursor = _myers_align_cursor(self.predicted_tokens, context_tokens)
        if cursor is None or cursor >= len(self.predicted_tokens):
            return []

        # 3. Return up to max_lookahead tokens.
        end = min(cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[cursor:end]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_x(self) -> int | None:
        """Return *x* coordinate where the optimal path ends for current B′."""

        # The last frontier corresponds to the current optimal edit distance.
        m = len(self._sm._b)

        best_x = None

        # Scan *all* stored frontiers starting from d = 0.  The earliest d
        # that reaches y == m corresponds to the minimal-edit alignment for
        # the *current* prefix.  We still pick the furthest x among diagonals
        # of that d so that we progress as much as possible.

        for V in self._sm._trace:  # type: ignore[attr-defined]
            candidate_x = None
            for k, x in V.items():
                if x - k == m:
                    # Ensure the *last* move onto (x, m) was a diagonal
                    # equality so that we stay on a snake (prevents the
                    # mis-alignment where the optimal path ends with an
                    # insertion).
                    if x == 0 or m == 0:
                        continue  # cannot test equality – skip

                    a_tok = self.predicted_tokens[x - 1]
                    b_tok = self._sm._b[m - 1]  # type: ignore[attr-defined]

                    if a_tok != b_tok:
                        continue  # last step was insertion ⇒ ignore

                    if candidate_x is None or x > candidate_x:
                        candidate_x = x

            if candidate_x is not None:
                best_x = candidate_x
                break  # minimal d with diagonal ending

        return best_x
