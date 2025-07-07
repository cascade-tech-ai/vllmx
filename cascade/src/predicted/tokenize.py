"""Utility helper for *predicted outputs* tests.

At the time of writing the *predicted outputs* integration is still in flux.
For the purpose of unit tests we expose a very small wrapper around the
prototype `BasicTokenizer` so that callers can convert arbitrary text into
token-ids without depending on external model files.
"""

from __future__ import annotations

from typing import List

from joev.src.smart_prediction.tokenizer import BasicTokenizer

# Singleton instance – trivial for the whitespace tokeniser.
_TOKENIZER = BasicTokenizer()


def encode_predicted_text(text: str) -> List[int]:
    """Return *token ids* for *text* using the prototype tokenizer."""

    return _TOKENIZER.encode(text)
