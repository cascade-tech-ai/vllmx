"""Line-level *rapidfuzz* predictor – design document.

Goal
====
Provide a variant of :pymod:`smart_prediction.algorithms.fuzz` that performs the
alignment on **whole lines** instead of individual tokens.  Matching by line
improves the structural coherence of speculative inserts in code or prose
because partial-token edits (which can split identifiers or indentation) are no
longer considered.

The new predictor will therefore treat *both* the ground-truth *prediction* and
the already typed *context* as **sequences of strings** (one string == one
full line, _without_ its trailing "\n").  A minimal edit script on those
sequences is then computed with ``rapidfuzz.distance.Levenshtein.editops`` –
exactly the same approach the current token-level predictor takes, just with
different atomic elements.


Algorithm (per `propose()` call)
--------------------------------
Inputs:
  • `predicted_text`   – full text the model is expected to emit (from the
                         evaluation harness; internally we store the
                         corresponding *tokens* but we can convert to text).
  • `context_text`     – text that is already present in the editor buffer at
                         the moment of the call.  _Includes_ the current
                         **incomplete** line (cursor position).

Steps:

1. Split the strings at "\n" _without_ keeping the newline:

       prediction_lines  = predicted_text.split("\n")
       context_lines_raw = context_text.split("\n")

2. Separate the *current* line from the *completed* prefix:

       current_line_text = context_lines_raw[-1]          # may be ""
       context_lines     = context_lines_raw[:-1]          # list[str]

   The lines in `context_lines` are **guaranteed** to be complete because the
   original text contained a "\n" after each of them.

3. Compute a *minimal* edit script converting *prediction_lines* →
   *context_lines* using rapidfuzz:

       ops = Levenshtein.editops(prediction_lines, context_lines)

   Replay the script exactly like `fuzz._rf_align_cursor` does to find the
   X-coordinate (line index) `cursor_line` _after_ the entire `context_lines`
   have been matched.  If alignment is impossible return `[]`.

4. Anchor check for the **incomplete** line.

   The next text of the prediction must start exactly where the user cursor
   currently is.  Concretely, we require

       prediction_lines[cursor_line].startswith(current_line_text)

   If false we abstain (return empty list).

5. Determine the *character* offset within the first unmatched line:

       char_cursor = len("\n".join(prediction_lines[:cursor_line]))
                     + (1 if cursor_line > 0 else 0)  # account for newlines
                     + len(current_line_text)

   Everything **before** `char_cursor` in `predicted_text` has already been
   typed.

6. Convert `char_cursor` → `token_cursor` using the same infrastructure helper
   that maps character positions to token indices (used by existing
   predictors).  Slice `predicted_tokens[token_cursor : token_cursor +
   max_lookahead]` and return that list.


Handling of the current line
----------------------------
• It is *excluded* from the diff: we never allow an edit operation to touch
  characters that the user might still be editing.
• Only a **strict prefix** test is performed – if the user typed "pri" and
  `prediction_lines[cursor_line]` starts with "prin", that is **accepted**;
  but if it starts with "por" we refuse to speculate.


Rapidfuzz on sequences of lines
-------------------------------
`Levenshtein.editops` operates on *any* sequence of hashable objects.  Passing
`List[str]` therefore works out of the box.  No custom hashing needed.


Complexities / Edge cases
-------------------------
1. CRLF – if the editor provides "\r\n" we should normalise to "\n" first.
2. Empty current line – prefix test trivially succeeds; we may predict the rest
   of that line including a closing newline if present in the ground truth.
3. Prediction shorter than context – caught by the alignment step; returns
   empty list.
4. Performance – number of *lines* in prompt is small; rapidfuzz diff on ~200
   elements is sub-millisecond.


Next steps to implement
-----------------------
1. Add helper `_rf_align_cursor_lines(a: List[str], b: List[str]) -> Optional[int]`
   mirroring the token variant but on strings.
2. Draft `RapidFuzzLinePredictor` class implementing `propose()`.
3. Unit tests covering the scenarios above.


"""

# ---------------------------------------------------------------------------
# Actual implementation below the design notes.
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional, Tuple

# Diff implementation selector (Levenshtein vs LCS).

import os

from rapidfuzz.distance import Levenshtein, LCSseq

_DIFF_IMPL = LCSseq
_override = os.getenv("SMART_DIFF_ALGO")
if _override is not None:
    if _override.lower().startswith("lev"):
        _DIFF_IMPL = Levenshtein
    elif _override.lower().startswith("lcs"):
        _DIFF_IMPL = LCSseq

from .base import PredictorBase

# ---------------------------------------------------------------------------
# Internal helpers – token⇄line conversions
# ---------------------------------------------------------------------------


def _split_by_newline_tokens(
    tokens: List[int],
    newline_set: set[int],
) -> List[Tuple[int, ...]]:
    """Return *lines* where **each element is a tuple of tokens**.

    A *newline* is *any token id present in ``newline_set``*.  The newline token
    **itself** is part of the *preceding* line.  Several consecutive newline
    tokens therefore yield *empty* lines between them – exactly mirroring the
    user's specification (one token with ``"\n\n"`` counts as a single line;
    two consecutive tokens with ``"\n"`` each count as two lines).
    """

    lines: List[List[int]] = [[]]
    for tok in tokens:
        lines[-1].append(tok)
        if tok in newline_set:
            lines.append([])

    # Convert to immutable tuples for hashing in rapidfuzz.
    return [tuple(line) for line in lines]


def _rf_align_cursor_lines(
    a: List[Tuple[int, ...]],
    b: List[Tuple[int, ...]],
) -> Optional[int]:
    """Return *x* position in *a* after aligning full *b* via rapidfuzz.

    Parameters
    ----------
    a
        Sequence of **predicted** line tuples.
    b
        Sequence of **context** line tuples (only *completed* lines).
    """

    m = len(b)
    if m == 0:
        return 0

    # Minimal edit script A → B.
    ops = _DIFF_IMPL.editops(a, b)

    x = 0
    y = 0

    for op in ops:
        # Advance across equal region (snake) before this edit.
        while x < op.src_pos and y < op.dest_pos and y < m:
            x += 1
            y += 1

        if y >= m:
            break

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

    # Trailing snake after last edit.
    while y < m and x < len(a) and a[x] == b[y]:
        x += 1
        y += 1

    if y == m and x > 0 and a[x - 1] == b[-1]:
        return x

    return None


# ---------------------------------------------------------------------------
# Predictor implementation
# ---------------------------------------------------------------------------


class RapidFuzzLinePredictor(PredictorBase):
    """Predictor that aligns **lines** via rapidfuzz edit distance."""

    def __init__(
        self,
        predicted_tokens: List[int],
        *,
        newline_token: int,
        tokenizer=None,
    ) -> None:  # noqa: D401 simple
        """Create predictor.

        Parameters
        ----------
        predicted_tokens
            Full sequence we want to predict.
        newline_token
            Token that represents a standalone "\n" (always considered a line
            boundary).  Additional boundary tokens are discovered **via the
            tokenizer** if provided.
        tokenizer
            Optional HF tokenizer; if supplied we inspect *every* token in
            `predicted_tokens` and add those whose decoded form contains a
            newline to the *newline set*.
        """

        super().__init__(predicted_tokens)

        # ------------------------------------------------------------------ detect newline-like tokens
        newline_set: set[int] = {newline_token}

        if tokenizer is not None:
            # Decode each unique token only once.
            seen: set[int] = set()
            for tok in predicted_tokens:
                if tok in seen or tok in newline_set:
                    continue
                seen.add(tok)
                text = tokenizer.decode([tok])
                if "\n" in text:
                    newline_set.add(tok)

        self._newline_set = newline_set

        # Pre-compute *prediction* as line tuples.
        self._predicted_line_tuples: List[Tuple[int, ...]] = _split_by_newline_tokens(
            predicted_tokens, newline_set
        )

        # Map line index → token index for quick back-conversion.
        self._line_starts: List[int] = [0]
        for idx, tok in enumerate(predicted_tokens):
            if tok in self._newline_set:
                self._line_starts.append(idx + 1)

        # All *incremental* context state is now stored in the **state dict**
        # passed to :pymeth:`propose`, therefore the predictor instance itself
        # remains **stateless** between requests.

    # ------------------------------------------------------------------ PredictorBase
    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:  # noqa: D401 simple
        if not context_tokens:
            return []



        # ------------------------------------------------------------------
        # 1.  Incrementally update *context* state (kept in ``state``).
        # ------------------------------------------------------------------
        ctx_processed: int = state.get("ctx_processed", 0)
        ctx_line_tuples: List[Tuple[int, ...]] = state.setdefault("ctx_line_tuples", [])
        current_line_tokens: List[int] = state.setdefault("current_line_tokens", [])

        if len(context_tokens) < ctx_processed:
            # Context was reset – clear incremental buffers and start anew.
            ctx_processed = 0
            ctx_line_tuples.clear()
            current_line_tokens.clear()

        for tok in context_tokens[ctx_processed:]:
            current_line_tokens.append(tok)
            if tok in self._newline_set:
                # Finalise current line (including the newline token itself).
                ctx_line_tuples.append(tuple(current_line_tokens))
                current_line_tokens.clear()

        ctx_processed = len(context_tokens)

        # Persist updated numbers back into the state dict so that the next
        # call continues where we left off.
        state["ctx_processed"] = ctx_processed

        last_token_is_newline = (
            len(context_tokens) > 0 and context_tokens[-1] in self._newline_set
        )

        if last_token_is_newline:
            # Include *empty* trailing line after newline.
            completed_line_tuples = ctx_line_tuples + [tuple()]
            current_line_prefix: List[int] = []
        else:
            completed_line_tuples = ctx_line_tuples
            current_line_prefix = list(current_line_tokens)

        # ------------------------------------------------------------------
        # 2.  Align completed lines via rapidfuzz.
        # ------------------------------------------------------------------
        line_cursor = _rf_align_cursor_lines(
            self._predicted_line_tuples,
            completed_line_tuples,
        )
        if line_cursor is None or line_cursor >= len(self._predicted_line_tuples):
            return []  # Cannot align – abstain.

        # ------------------------------------------------------------------
        # 3.  Verify *current* line is a prefix of the prediction.
        # ------------------------------------------------------------------
        predicted_line_start_idx = self._line_starts[line_cursor]

        # Gather predicted tokens of that line.
        predicted_line_tokens: List[int] = []
        idx = predicted_line_start_idx
        while idx < len(self.predicted_tokens) and self.predicted_tokens[idx] not in self._newline_set:
            predicted_line_tokens.append(self.predicted_tokens[idx])
            idx += 1

        # Prefix check.
        if current_line_prefix != predicted_line_tokens[: len(current_line_prefix)]:
            return []

        # ------------------------------------------------------------------
        # 4.  Compute token cursor (consumed tokens) and return lookahead slice.
        # ------------------------------------------------------------------
        token_cursor = predicted_line_start_idx + len(current_line_prefix)

        if token_cursor >= len(self.predicted_tokens):
            return []

        end = min(token_cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[token_cursor:end]

