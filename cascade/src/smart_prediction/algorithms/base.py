from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class PredictorBase(ABC):
    """Abstract base class for *smart prediction* algorithms.

    A predictor receives
    1. The *full* sequence of tokens that it expects the language model to
       generate (`predicted_tokens`).
    2. The *already generated* tokens so far (`context_tokens`).  These are a
       prefix of the ground-truth output in the simulation.

    The task of the predictor is to *propose* up to `max_lookahead` tokens that
    it believes come **next** given the current context.

    If it is unsure it should return an *empty* list, letting the model advance
    one token by itself (the simulator will consume `+1` ground-truth token in
    any case).
    """

    def __init__(self, predicted_tokens: List[int]):
        self._predicted_tokens = predicted_tokens

    @abstractmethod
    def propose(
        self,
        context_tokens: List[int],
        max_lookahead: int,
        state: dict,
    ) -> List[int]:
        """Return a list of token-ids – **may be empty**.

        Parameters
        ----------
        context_tokens
            Tokens that have already been *generated* by the model so far.
        max_lookahead
            Maximum number of tokens that may be speculatively proposed.
        state
            Mutable dictionary that is preserved **within** a single request
            and reset for every *new* request.  Predictors can use this to
            store incremental parsing results instead of keeping mutable state
            on ``self``.  Predictors that do not require any state can safely
            ignore the argument (it will be an empty ``dict``).

        The returned list *must not* be longer than ``max_lookahead``.
        """

    # ------------------------------------------------------------------
    # Optional – simulation feedback
    # ------------------------------------------------------------------
    def ack(self, num_correct: int) -> None:  # noqa: D401 – simple verb
        """Notification from the simulator – *num_correct* tokens were accepted.

        Predictors that keep internal state (e.g. `NaiveSequentialPredictor`)
        can override this method to update their cursor.  The default
        implementation is a no-op so that stateless predictors do not need to
        worry about it.
        """

        # Default: do nothing.
        return None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def predicted_tokens(self) -> List[int]:
        return self._predicted_tokens
