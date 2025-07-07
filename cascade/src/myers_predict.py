"""One-shot predictor with coloured diff visualisation.

The script compares a *prediction/reference* sequence **A** with a *ground-
truth* sequence **B**.  Only a **prefix** of *B* (configurable via
``--ratio``) is revealed to the alignment algorithm; the remainder serves as
hold-out data to evaluate the quality of the continuation that the predictor
outputs.

Conceptual mapping (speculative decoding)
----------------------------------------
* **A** – *prediction*  (1st CLI arg)
  Your *speculative guess* of what the model will produce in entirety.  It is
  immutable and **does not** influence what the model really outputs – it is
  only used for *verification*.

* **B** – *ground-truth* (2nd CLI arg)
  The *actual* tokens streamed by the model.  A prefix **B′** has already
  arrived; the remainder is still unknown at alignment time and becomes the
  *continuation* we want to visualise.

Visual output
-------------
1. **Alignment diff** of A vs. B′
   Highlights how much of the speculative text matches the context the model
   has already emitted (grey = match, blue = only in A, orange = insertion
   from B′).
2. ``BEGIN PREDICTION:`` separator.
3. **Ground-truth continuation** (next *N* tokens from B) coloured by
   predictor accuracy – green if the same token was present at the
   corresponding position in A, red otherwise.

Below that, statistics such as runtime and *#correct / #shown* are printed.  A
diagnostic note is emitted if the Myers alignment fails entirely so that such
cases stand out in logs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List

# ---------------------------------------------------------------------------
# Optional heavy dependency for token-level mode
# ---------------------------------------------------------------------------

try:
    from transformers import AutoTokenizer  # noqa: F401 – optional
except ModuleNotFoundError:  # pragma: no cover – transformers not installed
    AutoTokenizer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers & imports
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Alignment helpers
# ---------------------------------------------------------------------------

# We keep the original pure-Python Myers implementation for reference but use
# the significantly faster C implementation from the Python standard library
# (difflib.SequenceMatcher) for the *cursor* calculation because our usage
# pattern only needs the *position* – not the full edit script.

# Prefer rapidfuzz (Rust/C) when available for the cursor calculation; fall
# back to the stdlib difflib version otherwise (always present).
from difflib import SequenceMatcher

try:
    from rapidfuzz.distance import Levenshtein as _rf_lev  # type: ignore

    _HAVE_RAPIDFUZZ = True
except ModuleNotFoundError:  # pragma: no cover – optional dependency
    _HAVE_RAPIDFUZZ = False

# Keep reference to original implementation for potential debugging – import
# guarded so that linters do not mark it as unused when optimisation is used
# exclusively.
from joev.src.smart_prediction.algorithms.myers_streaming import (
    _myers_align_cursor as _myers_align_cursor_ref,  # noqa: F401
)

from joev.src import myers_diff as _myers


CSI = "\x1b["


def _c(text: str, code: str) -> str:
    """Return *text* wrapped in ANSI colour *code*."""

    return f"{CSI}{code}m{text}{CSI}0m"


# 8-colour palette approximations (readable on dark & light terminals)
GREY = "90"  # bright black ≈ grey
BLUE = "34"
ORANGE = "33"  # yellow works okay-ish as orange surrogate
GREEN = "32"
RED = "31"


def _read_tokens(path: Path, use_tokens: bool, model_name: str) -> List[int | str]:
    """Load *path* and return a list of elements suitable for Myers diff.

    If *use_tokens* is *True* we tokenize with a HuggingFace tokenizer and
    return **token IDs** (``int``).  Otherwise each **line** (including the
    trailing newline) is treated as one element (``str``).  Returning raw
    *ints* for tokens is crucial so that the diff sees *equal* tokens even if
    they decode to the same string with different leading spaces.
    """

    if use_tokens:
        if AutoTokenizer is None:
            sys.exit("[error] transformers missing – cannot tokenize")
        tok = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True, trust_remote_code=True
        )
        text = path.read_text(encoding="utf-8", errors="replace")
        return tok.encode(text, add_special_tokens=False)

    # Fallback: treat each **line** as a diff element (preserve newlines so
    # round-trip is lossless).
    return path.read_text(encoding="utf-8", errors="replace").splitlines(
        keepends=True
    )


# ---------------------------------------------------------------------------
# Fast cursor determination using difflib
# ---------------------------------------------------------------------------


def _align_cursor(seq_a: List[int | str], seq_b_prefix: List[int | str]) -> int | None:
    """Return the *cursor* into *seq_a* after aligning *seq_b_prefix*.

    The implementation leverages ``difflib.SequenceMatcher`` which is backed
    by a C implementation and therefore considerably faster than the pure
    Python Myers routine for typical token-level inputs (×2–×3 on the sample
    files).

    The behaviour mirrors ``_myers_align_cursor``:
    • returns ``None`` if the *prefix* cannot be aligned under an edit script
      that *ends* on a **match or deletion** (i.e. not on an insertion),
    • otherwise returns the X-coordinate – the number of tokens from *A* that
      have been *consumed / matched* by the prefix.
    """

    m = len(seq_b_prefix)
    if m == 0:
        return 0

    # ------------------------------------------------------------------ 1. rapidfuzz fast-path -------------------------------------------------
    if _HAVE_RAPIDFUZZ:
        # rapidfuzz returns a **minimal** edit script as a list of EditOp
        # objects.  We replay the script to track how far we advance in *A*
        # (x) while consuming *all* of B′ (y == m).
        try:
            ops = _rf_lev.editops(seq_a, seq_b_prefix)
        except Exception:  # pragma: no cover – unexpected failure, fall back
            ops = None

        if ops is not None:
            x = 0
            y = 0
            # ops are sorted; iterate through them maintaining the cursor.
            for op in ops:
                # First, advance across the equal *snake* until we reach the
                # coordinates of the next edit op.
                while x < op.src_pos and y < op.dest_pos and y < m:
                    x += 1
                    y += 1

                if y >= m:
                    break

                # Apply the edit op itself.
                if op.tag == "delete":
                    x += 1
                elif op.tag == "insert":
                    y += 1
                else:  # replace
                    x += 1
                    y += 1

                if y >= m:
                    break

            # After processing edit ops we may still have a trailing snake.
            while y < m and x < len(seq_a) and seq_a[x] == seq_b_prefix[y]:
                x += 1
                y += 1

            if y == m:
                # Alignment only valid if last token matched (not insertion).
                if x > 0 and seq_a[x - 1] == seq_b_prefix[-1]:
                    return x
                # Otherwise treat as failure (ended on insertion).

        # If rapidfuzz is available we trust its minimal edit script; if this
        # ended on an insertion we *know* that **no** equal-or-shorter script
        # can end on a match (it would have to perform at least one extra
        # deletion/insert pair).  Therefore we can *skip* the slower difflib
        # fallback and return failure immediately.

        return None

    # ------------------------------------------------------------------ 2. difflib fallback (no RF) -------------------------------------------
    sm = SequenceMatcher(None, seq_a, seq_b_prefix, autojunk=False)
    for a_idx, b_idx, size in reversed(sm.get_matching_blocks()):
        if size == 0:
            continue  # sentinel
        if b_idx + size == m:
            return a_idx + size

    return None  # alignment failed in both strategies


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def _main(argv: List[str] | None = None) -> None:  # noqa: D401 simple
    parser = argparse.ArgumentParser(
        description="Visualise alignment of *prediction* against *ground truth* and show continuation"
    )
    parser.add_argument("prediction", type=Path, help="prediction / reference document  (A)")
    parser.add_argument("ground_truth", type=Path, help="ground-truth document (B)")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="fraction of B to use as prefix B′ (0‥1, default 0.5)",
    )
    parser.add_argument("--tokens", action="store_true", help="token-level diff via HuggingFace tokenizer")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="HF model name to load the tokenizer from (tokens mode)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=20,
        help="how many continuation tokens to show (default 20)",
    )

    args = parser.parse_args(argv)

    if not (0.0 <= args.ratio <= 1.0):
        sys.exit("[error] --ratio must be between 0 and 1")

    if not args.prediction.exists() or not args.ground_truth.exists():
        sys.exit("[error] input file(s) missing")

    # ----------------------------  load sequences
    seq_a = _read_tokens(args.prediction, args.tokens, args.model)
    seq_b = _read_tokens(args.ground_truth, args.tokens, args.model)

    # ----------------------------  create pretty-printer
    if args.tokens:
        if AutoTokenizer is None:
            sys.exit("[error] transformers missing – cannot decode tokens")

        _tok = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=True
        )

        def fmt(token_id: int | str) -> str:  # type: ignore[override]
            """Return human-readable text for a single token ID."""

            return _tok.decode([int(token_id)], clean_up_tokenization_spaces=False)

    else:

        def fmt(text: str | int) -> str:  # type: ignore[override]
            return str(text)

    # ------------------------------------------------------------------  stage 1: compute alignment cursor (Myers)
    prefix_len = int(len(seq_b) * args.ratio)
    prefix_b = seq_b[:prefix_len]

    # ----------------------------  alignment (fast path via difflib)
    t0 = time.perf_counter()
    raw_cursor = _align_cursor(seq_a, prefix_b)
    t_pred_ms = (time.perf_counter() - t0) * 1e3  # forward-pass only!  (diff below is excluded)

    alignment_failed = raw_cursor is None
    cursor = raw_cursor or 0  # keep downstream logic simple

    # Build diff **only** for the part of A that has been consumed (0‥cursor).
    # If alignment failed we fall back to diffing the *entire* A but trim any
    # trailing deletions so we do not leak the yet-unconsumed future.
    base_a_for_diff = seq_a[:cursor] if cursor > 0 else seq_a

    script_full = _myers.diff(base_a_for_diff, prefix_b)

    # ----- trim tail   (delete-only region from A == future prediction)
    last_non_del = -1
    for idx, (tag, _) in enumerate(script_full):
        if tag != "delete":
            last_non_del = idx

    script = script_full[: last_non_del + 1]

    # ------------------------------------------------------------------  colourise alignment stream
    coloured_stream_parts: List[str] = []
    # ------------------------------------------------------------------  formatting helper
    # In *token* mode decoded fragments already include the required leading
    # whitespace so we concatenate **without** an additional separator to
    # reconstruct the original string exactly.  In line-wise mode we insert a
    # single space for legibility between coloured segments.
    delim = "" if args.tokens else " "

    for tag, tok in script:
        tok_str = fmt(tok)
        if tag == "equal":
            coloured_stream_parts.append(_c(tok_str, GREY))
        elif tag == "delete":
            coloured_stream_parts.append(_c(tok_str, BLUE))  # only in A
        elif tag == "insert":
            coloured_stream_parts.append(_c(tok_str, ORANGE))  # only in B′
        else:  # pragma: no cover – defensive
            coloured_stream_parts.append(tok_str)

    # ------------------------------------------------------------------  stage 2: ground-truth continuation with accuracy colouring

    # ------------------------------------------------------------------ 1. collect predictor *guess*
    # Extract up to `--num-predict` tokens from **A** beginning at the cursor.
    # These tokens represent what the speculative predictor *would* like to
    # see next.  We will *not* print them directly – they are used solely to
    # decide whether each upcoming ground-truth token is coloured green or
    # red.
    predicted_tokens: List[int] = []
    if not alignment_failed and not (cursor == 0 and prefix_len > 0):
        lookahead = min(args.num_predict, len(seq_a) - cursor)
        if lookahead > 0:
            predicted_tokens = seq_a[cursor : cursor + lookahead]

    # ------------------------------------------------------------------ 2. reference continuation (ground truth)
    # Select the same window size from **B** right after the revealed prefix.
    # This is what the model truly emits and therefore what we display.
    max_show = min(args.num_predict, len(seq_b) - prefix_len)
    gt_continuation = seq_b[prefix_len : prefix_len + max_show]

    coloured_pred_parts: List[str] = []
    correct_cnt = 0

    # ------------------------------------------------------------------ 3. colourise GT tokens according to predictor accuracy
    for idx, tok in enumerate(gt_continuation):
        tok_str = fmt(tok)
        if idx < len(predicted_tokens) and tok == predicted_tokens[idx]:
            coloured_pred_parts.append(_c(tok_str, GREEN))
            correct_cnt += 1
        else:
            coloured_pred_parts.append(_c(tok_str, RED))

    # ------------------------------------------------------------------  emit output
    # 1. alignment diff
    print(delim.join(coloured_stream_parts))

    # 2. separator
    print("\n\nBEGIN PREDICTION:\n\n")

    # 3. continuation or failure notice
    if alignment_failed:
        print(_c("failed alignment, no prediction", RED))
    elif coloured_pred_parts:
        print(delim.join(coloured_pred_parts))

    # 4. stats & diagnostics
    print()
    print(f"A tokens          : {len(seq_a)}")
    print(
        f"B tokens          : {len(seq_b)}  (prefix {prefix_len}, suffix {len(seq_b) - prefix_len})"
    )
    print(f"forward time      : {t_pred_ms:.2f} ms")

    if alignment_failed:
        print("continuation tokens : 0 (alignment failed – no prediction)")
    elif gt_continuation:
        print(
            f"continuation tokens : {len(gt_continuation)} (predicted correctly {correct_cnt} / {len(gt_continuation)})"
        )
    else:
        print("continuation tokens : 0 (no continuation available)")

    if alignment_failed:
        print("[debug] Myers alignment failed – context could not be matched against prediction.\n")


if __name__ == "__main__":  # pragma: no cover
    _main()
