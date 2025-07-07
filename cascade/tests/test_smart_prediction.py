import math


from joev.src.smart_prediction.algorithms.naive_sequential import NaiveSequentialPredictor
from joev.src.smart_prediction.algorithms.line_diff import LineDiffPredictor
from joev.src.smart_prediction.algorithms.greedy_suffix import GreedySuffixPredictor
from joev.src.smart_prediction.tokenizer import BasicTokenizer
from joev.src.smart_prediction.examples import (
    POEM_MISSING_MIDDLE_STANZA,
    POEM_THREE_STANZAS,
)

# ---------------------------------------------------------------------------
# Helper utilities – colourful diagnostics
# ---------------------------------------------------------------------------


def _print_coloured_alignment(result, ground_truth_text, tokenizer):  # pragma: no cover
    """Pretty-print *ground_truth_text* colouring *predicted* tokens blue.

    Tokens that **were not** predicted (i.e. produced by the model itself) are
    coloured red so that human readers can quickly judge how well the
    predictor did.  The function is designed for *debugging* only – it is only
    executed when pytest is run with ``-s`` so that *stdout* is visible.
    """

    BLUE = "\033[94m"
    RED = "\033[91m"
    RESET = "\033[0m"

    ground_ids = tokenizer.encode(ground_truth_text)
    coloured = []
    for tok_id, was_predicted in zip(ground_ids, result.predicted_mask):
        txt = tokenizer.decode([tok_id])
        if txt == "\n":
            coloured.append("\n")
            continue

        colour = BLUE if was_predicted else RED
        # Add an explicit marker so that the output remains readable when ANSI
        # colours are not rendered (e.g. in some CI logs).
        marker = "[P]" if was_predicted else "[M]"
        coloured.append(f"{colour}{marker}{txt}{RESET}")

    # Join with spaces, taking care of newlines being separate items.
    out_lines = []
    buffer = []
    for part in coloured:
        if part == "\n":
            out_lines.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(part)
    if buffer:
        out_lines.append(" ".join(buffer))

    print("\n[DEBUG] Predicted tokens in blue, missed in red:\n")
    print("\n".join(out_lines))

from joev.src.smart_prediction.simulator import simulate


def test_naive_perfect_poem():
    # With a *perfect* prediction the naive predictor should achieve *all*
    # correct tokens and *zero* wrong ones.
    result = simulate(
        NaiveSequentialPredictor,
        predicted_text=POEM_THREE_STANZAS,
        ground_truth_text=POEM_THREE_STANZAS,
        lookahead=128,
    )

    # Human-readable diagnostic output (visible with `pytest -s`).
    _print_coloured_alignment(result, POEM_THREE_STANZAS, BasicTokenizer())

    assert result.wrong_predictions == 0
    assert result.correct_predictions > 0
    # Score should be exactly the number of tokens.
    assert math.isclose(result.score, result.correct_predictions)


def test_line_diff_handles_missing_stanza():
    # The *prediction* is missing the middle stanza.  A naive predictor would
    # quickly go off the rails but the *line diff* predictor should be able to
    # resynchronise once the third stanza starts.

    result = simulate(
        LineDiffPredictor,
        predicted_text=POEM_MISSING_MIDDLE_STANZA,
        ground_truth_text=POEM_THREE_STANZAS,
        lookahead=32,
    )

    _print_coloured_alignment(result, POEM_THREE_STANZAS, BasicTokenizer())

    # We expect at least some correct predictions (specifically tokens in the
    # first **and** last stanza) and *fewer* wrong predictions than naive.
    assert result.correct_predictions > 0
    # Empirically the diff predictor should achieve a positive score here.
    assert result.score > 0


# ---------------------------------------------------------------------------
# Additional regression tests proposed in *smart_prediction.md*
# ---------------------------------------------------------------------------


# Small Python algorithm ------------------------------------------------------


PYTHON_BUBBLE_SORT = (
    "def bubble_sort(arr):\n"
    "    n = len(arr)\n"
    "    for i in range(n):\n"
    "        for j in range(0, n - i - 1):\n"
    "            if arr[j] > arr[j + 1]:\n"
    "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
    "    return arr"
)


PYTHON_BUBBLE_SORT_RENAMED = (
    "def bubble_sort(nums):\n"
    "    n = len(nums)\n"
    "    for i in range(n):\n"
    "        for j in range(0, n - i - 1):\n"
    "            if nums[j] > nums[j + 1]:\n"
    "                nums[j], nums[j + 1] = nums[j + 1], nums[j]\n"
    "    return nums"
)


def test_naive_vs_diff_variable_rename():
    """Compare predictors on a variable-rename mutation.

    The *ground truth* is the original bubble-sort implementation, whereas the
    *prediction* uses a different variable name throughout.  We expect the
    naive predictor to go off track almost immediately, yielding a *negative*
    score, while the *line-diff* predictor should eventually re-anchor once it
    encounters an unchanged line and therefore achieve a **positive** score.
    """

    # Naive predictor – should perform poorly (negative score expected).
    res_naive = simulate(
        NaiveSequentialPredictor,
        predicted_text=PYTHON_BUBBLE_SORT_RENAMED,
        ground_truth_text=PYTHON_BUBBLE_SORT,
        lookahead=16,
    )

    _print_coloured_alignment(res_naive, PYTHON_BUBBLE_SORT, BasicTokenizer())

    assert res_naive.score < 0

    # Line-diff predictor – expected to recover and earn a positive score.
    res_diff = simulate(
        GreedySuffixPredictor,
        predicted_text=PYTHON_BUBBLE_SORT_RENAMED,
        ground_truth_text=PYTHON_BUBBLE_SORT,
        lookahead=16,
    )

    # Myers-streaming predictor – should perform at least as well as greedy
    # suffix on variable renames because it uses full-diff alignment.
    from joev.src.smart_prediction.algorithms import MyersStreamingPredictor

    res_myers = simulate(
        MyersStreamingPredictor,
        predicted_text=PYTHON_BUBBLE_SORT_RENAMED,
        ground_truth_text=PYTHON_BUBBLE_SORT,
        lookahead=16,
    )

    _print_coloured_alignment(res_myers, PYTHON_BUBBLE_SORT, BasicTokenizer())

    # Must outperform naive baseline.
    assert res_myers.score > res_naive.score
    # Should not be worse than greedy suffix by more than a small margin.
    assert res_myers.score > (res_diff.score - 5)  # allow small variance

    _print_coloured_alignment(res_diff, PYTHON_BUBBLE_SORT, BasicTokenizer())

    # Basic sanity: diff predictor must outperform naive.
    assert res_diff.score > res_naive.score
    assert res_diff.score > 0
