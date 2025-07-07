from __future__ import annotations

from typing import List

from .base import PredictorBase



class LineDiffPredictor(PredictorBase):
    """A *line-based* predictor that can recover after local mismatches.

    The algorithm treats the *predicted text* as a sequence of **lines**.  To
    decide where it currently is it compares the *suffix* of the *generated
    context* (also line-split) with every window of equal length inside the
    predicted text and picks the *rightmost* match.  This is equivalent to
    finding the longest suffix of the context that is also a prefix of some
    suffix of the prediction – a cheap form of *anchored diff*.

    Once an alignment is found, we simply propose up to `max_lookahead` tokens
    that follow the matched position.

    The method is intentionally simple (O(N·M) worst-case) but more than fast
    enough for the small prototype inputs in `joev/docs/smart_prediction.md`.
    """

    def __init__(self, predicted_tokens: List[int], newline_token: int):
        super().__init__(predicted_tokens)
        self._newline_token = newline_token

        # Pre-compute *line offsets* so we can quickly map between *line index*
        # and *token index*.
        self._line_starts: List[int] = [0]
        for idx, tok in enumerate(predicted_tokens):
            if tok == newline_token:
                self._line_starts.append(idx + 1)

    # ------------------------------------------------------------------
    # PredictorBase
    # ------------------------------------------------------------------
    def propose(self, context_tokens: List[int], max_lookahead: int, state: dict) -> List[int]:
        # Do **not** bail out merely because the *context* has advanced past
        # the length of the *prediction* – the prediction might have skipped
        # a block (e.g. an omitted stanza) and still contain relevant tokens
        # later on.  We therefore remove the early-exit that compared the two
        # lengths and rely on the alignment logic below to decide whether
        # further predictions are possible.

        # ------------------------------------------------------------------
        # 1. Fast *line-suffix* alignment.
        # ------------------------------------------------------------------
        # We look at the **last line** of the already generated context (i.e.
        # the tokens after the most recent newline).  If that entire line
        # appears anywhere in the predicted text we treat the character after
        # the match as the *current cursor*.

        try:
            last_nl = max(i for i, tok in enumerate(context_tokens) if tok == self._newline_token)  # noqa: E501
            line_suffix = context_tokens[last_nl + 1 :]
        except ValueError:
            # No newline found – take the whole context.
            line_suffix = context_tokens

        # Bail out early if we do not have any context yet.
        if not line_suffix:
            return []

        # Naive search for the *line_suffix* inside the predicted tokens.
        token_cursor = _find_subseq(self.predicted_tokens, line_suffix)
        if token_cursor is None:
            return []  # cannot align – abstain from prediction

        token_cursor += len(line_suffix)  # advance to the position *after* the suffix

        if token_cursor >= len(self.predicted_tokens):
            return []

        # ------------------------------------------------------------------
        # 2. Return up to `max_lookahead` tokens starting at *token_cursor*.
        # ------------------------------------------------------------------
        end = min(token_cursor + max_lookahead, len(self.predicted_tokens))
        return self.predicted_tokens[token_cursor:end]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _find_subseq(haystack: List[int], needle: List[int]) -> int | None:
    """Return the *start index* of *needle* inside *haystack* or *None*.

    The implementation is a straightforward forward scan because our inputs
    are tiny (≫ 1e3 tokens) in the unit tests.
    """

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _tokens_to_lines(self, tokens: List[int]) -> List[str]:
        """Convert a token list into *line strings* (tokens joined by space)."""
        lines: List[List[str]] = [[]]
        for tok in tokens:
            if tok == self._newline_token:
                lines.append([])
            else:
                # Use string representation of token id – we do not have the
                # original text here and only need *equality* for diff.
                lines[-1].append(str(tok))

        # Join tokens within each line to single string so that SequenceMatcher
        # operates on lines rather than tokens.
        return [" ".join(toks) for toks in lines]
