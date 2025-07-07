"""Container for *predicted output* settings used by speculative decoding.

This helper struct allows callers to provide a *prediction* of what the model
will generate, enabling the speculative-decoding pipeline to skip work when
the prediction matches.  Either the human-readable *text* or the already
tokenised *token IDs* can be supplied.

The class lives in :pymod:`vllm.spec_decode` so it can be used without pulling
in heavy model-execution modules during type-checking.
"""

from __future__ import annotations

from typing import Optional

import msgspec


class PredictedOutputParams(msgspec.Struct, omit_defaults=True):  # type: ignore[call-arg]
    """Parameters describing the caller-supplied *predicted output*.

    Exactly one of the two representations should be provided:

    predicted_text
        Raw text in *string* form.  Convenient for API clients.  Will be
        tokenised by vLLM before the request is executed.

    predicted_token_ids
        Pre-tokenised sequence corresponding to *predicted_text*.  Supplying
        IDs avoids redundant work when the client already has a tokenizer.
    """

    predicted_text: Optional[str] = None
    predicted_token_ids: Optional[list[int]] = None

    def has_prediction(self) -> bool:  # noqa: D401 – simple util
        return bool(self.predicted_text or self.predicted_token_ids)
