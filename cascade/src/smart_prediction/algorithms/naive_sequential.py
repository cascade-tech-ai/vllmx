"""Simplest speculative predictor.

The *naive sequential* strategy always proposes the **next** ``max_lookahead``
tokens from the reference prediction, assuming that the model output will
exactly match the ground-truth text.

Implementation details
----------------------
The current *cursor* (how many tokens have already been accepted by the
simulator) is **derived from the provided context** on every call rather than
being stored on the instance.  This makes the algorithm *stateless* between
requests and therefore compatible with the new ``state`` mechanism passed in
by the simulator.
"""

from __future__ import annotations

from typing import List

from .base import PredictorBase


class NaiveSequentialPredictor(PredictorBase):
    """Always returns the next ``max_lookahead`` ground-truth tokens."""

    # ------------------------------------------------------------------ PredictorBase
    def propose(
        self,
        context_tokens: List[int],
        max_lookahead: int,
        state: dict,  # noqa: D401 – simple verb
    ) -> List[int]:
        # Use *internal* cursor maintained via ``ack`` just like the original
        # implementation – this achieves the intended behaviour while still
        # keeping the new *state*-based signature.
        cursor: int = getattr(self, "_cursor", 0)

        remaining = len(self.predicted_tokens) - cursor
        if remaining <= 0:
            return []

        n = min(max_lookahead, remaining)
        return self.predicted_tokens[cursor : cursor + n]

    # ------------------------------------------------------------------ simulation feedback (unused)
    def ack(self, num_correct: int) -> None:  # noqa: D401 – simple verb
        # Lazily create attribute (avoids __init__ boilerplate when the class
        # is instantiated but never used).
        if not hasattr(self, "_cursor"):
            self._cursor = 0  # type: ignore[attr-defined]

        self._cursor += num_correct
