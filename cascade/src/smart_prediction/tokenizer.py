"""Thin wrapper around a HuggingFace *AutoTokenizer* for the simulation.

The previous *BasicTokenizer* implementation was a whitespace-split fallback
used for CI environments without access to the real model files.  We now
assume that the proper tokenizer is available locally and expose **exactly**
the two helpers required by the simulator:

    • ``encode(text) -> List[int]``
    • ``decode(token_ids) -> str``

The class intentionally keeps the original name *BasicTokenizer* so that no
changes are needed elsewhere in the codebase.
"""

from __future__ import annotations

from typing import List

from transformers import AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


class BasicTokenizer:  # noqa: D101 (kept name for backward compatibility)
    """HuggingFace tokenizer wrapper exposing `encode` / `decode`."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        # Load tokenizer from local cache; we deliberately disallow network IO
        # because the simulation should run fully offline.
        self._tok = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True
        )

        # Provide *eos_token* attribute for predictors that may rely on it.
        self.eos_token = self._tok.eos_token

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:  # noqa: D401 simple
        """Return list of token IDs without adding special tokens."""

        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:  # noqa: D401 simple
        """Inverse of :py:meth:`encode`.  Keeps whitespace exactly."""

        return self._tok.decode(token_ids, clean_up_tokenization_spaces=False)
