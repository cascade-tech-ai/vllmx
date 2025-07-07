"""Simple loop that *simulates* speculative decoding with a predictor.

The simulator is *deterministic* because it operates on a *ground-truth*
document rather than a neural network.  The only randomness comes from the
predictor if it chooses to be stochastic (none of the prototypes are).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Type, Optional

from .tokenizer import BasicTokenizer

# ---------------------------------------------------------------------------
# ANSI helpers for optional verbose output
# ---------------------------------------------------------------------------

CSI = "\x1b["


def _c(text: str, code: str) -> str:
    return f"{CSI}{code}m{text}{CSI}0m"


GREY = "90"
GREEN = "32"
RED = "31"
ORANGE = "33"


# ---------------------------------------------------------------------------
# Results dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Container for the outcome of a speculative-decoding simulation.

    Attributes
    ----------
    correct_predictions
        Number of tokens that were *successfully* predicted by the predictor.
    wrong_predictions
        Number of tokens that were *proposed* by the predictor but turned out
        to be wrong (i.e. mismatched the ground-truth stream).
    predicted_mask
        Boolean mask (same length as the ground-truth token stream) indicating
        for every token whether it has been *predicted* (``True``) or whether
        it had to be produced by the target model itself (``False``).
    """

    correct_predictions: int
    wrong_predictions: int
    predicted_mask: list[bool]
    iterations: int
    total_time_ms: float
    avg_tokens_per_iter: float
    avg_time_ms_per_iter: float

    @property
    def score(self) -> float:
        """Composite metric defined in `smart_prediction.md`."""
        return self.correct_predictions - (self.wrong_predictions / 10.0)

    @property
    def coverage(self) -> float:  # noqa: D401 simple
        """Fraction of *ground-truth* tokens that were predicted correctly."""
        if not self.predicted_mask:
            return 0.0
        return self.correct_predictions / len(self.predicted_mask)

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"correct={self.correct_predictions}  wrong={self.wrong_predictions}  "
            f"score={self.score:.2f}"
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def simulate(
    predictor_cls: Type["PredictorBase"],
    predicted_text: str,
    ground_truth_text: str,
    lookahead: int = 20,
    tokenizer: BasicTokenizer | None = None,
    *,
    verbose: bool = False,
    max_iterations: Optional[int] = None,
    alg_name: str = "",
    debug_iteration: Optional[int] = None,
) -> SimulationResult:
    """Run the speculative-decoding simulation and return metrics."""

    # Local import to avoid circulars (predictor depends on simulator helpers).
    from .algorithms.base import PredictorBase  # pylint: disable=import-at-top

    tokenizer = tokenizer or BasicTokenizer()

    ground_tokens: List[int] = tokenizer.encode(ground_truth_text)
    predicted_tokens: List[int] = tokenizer.encode(predicted_text)

    predictor: PredictorBase
    if predictor_cls.__name__ in {"RapidFuzzLinePredictor", "SequenceMatcherLinePredictor"}:
        newline_tok = tokenizer.encode("\n")[0]
        predictor = predictor_cls(predicted_tokens, newline_token=newline_tok, tokenizer=tokenizer)  # type: ignore[arg-type]
    elif predictor_cls.__name__ in {"LineDiffPredictor", "MyersLinePredictor"}:
        newline_tok = tokenizer.encode("\n")[0]
        predictor = predictor_cls(predicted_tokens, newline_tok)  # type: ignore[arg-type]
    else:
        predictor = predictor_cls(predicted_tokens)  # type: ignore[abstract]

    # Simulation state ------------------------------------------------------------------
    ctx: List[int] = []  # tokens *generated* so far
    # Predictor-specific mutable state.  Reset at the beginning of **each**
    # simulation run (i.e. for every new *request* in the serving context).
    # Predictors that do not need any state can simply ignore this argument.
    state: dict = {}
    gt_cursor = 0  # index into `ground_tokens`

    num_correct = 0
    num_wrong = 0

    # Track per-token coverage so that the test suite can visualise which
    # parts of the ground-truth stream were predicted by the proposer.
    predicted_mask: List[bool] = [False] * len(ground_tokens)

    iteration = 0

    propose_time_total_ms = 0.0
    tokens_total = 0

    while gt_cursor < len(ground_tokens):
        if max_iterations is not None and iteration >= max_iterations:
            break

        # ------------------------------------------------------------- obtain proposal
        if gt_cursor == 0:
            # First iteration – always speculate the first *lookahead* tokens
            # from the reference prediction, independent of the algorithm.
            proposal = predicted_tokens[:lookahead]
            propose_ms = 0.0  # Not timed – trivial slice.
        else:
            t0 = time.perf_counter()
            proposal = predictor.propose(ctx, lookahead, state)
            propose_ms = (time.perf_counter() - t0) * 1e3

        # Step 1: compare *proposal* against *ground truth* starting at cursor.
        correct_here = 0
        for i, tok in enumerate(proposal):
            target_idx = gt_cursor + i
            if target_idx < len(ground_tokens) and tok == ground_tokens[target_idx]:
                correct_here += 1
                predicted_mask[target_idx] = True  # mark as correctly predicted
            else:
                break  # stop at first mismatch

        # Redefine *wrong* – the first mismatching token only *breaks* the
        # speculative run (orange) and is **not** penalised.  Only tokens
        # *after* that are counted as wrong (red).
        # All proposal tokens after the correct prefix are wrong (red).
        wrong_here = len(proposal) - correct_here

        # Update overall metrics.
        num_correct += correct_here
        num_wrong += wrong_here

        tokens_total += len(proposal) + 1  # +1 generated token each iter
        propose_time_total_ms += propose_ms

        # Step 2: model *consumes* the accepted tokens + **one** extra token.
        to_consume = correct_here + 1
        # Do not step beyond the available ground-truth tokens.
        remaining = len(ground_tokens) - gt_cursor
        to_consume = min(to_consume, remaining)

        ctx.extend(ground_tokens[gt_cursor : gt_cursor + to_consume])
        gt_cursor += to_consume

        # Invariant: all consumed tokens must equal the corresponding prefix
        # of the ground-truth sequence.
        assert ctx == ground_tokens[:gt_cursor], "ctx diverged from ground truth – bug in simulator logic"

        # Feedback for stateful predictors.
        if hasattr(predictor, "ack"):
            predictor.ack(correct_here)

        # ------------------------------- verbose logging
        if verbose:
            # Start index in ground truth before this step.
            start_idx = gt_cursor - to_consume  # position before consumption

            ctx_col = _c(f"idx {start_idx}", GREY)

            # Proposal coloured by correctness
            coloured_parts: List[str] = []

            # 1. Correctly predicted tokens (green)
            for i in range(correct_here):
                tok_text = tokenizer.decode([proposal[i]])
                coloured_parts.append(_c(tok_text, GREEN))

            # 2. Generated ground-truth token (orange) – always exists unless
            #    we have reached end-of-stream.
            gen_idx = gt_cursor + correct_here
            if gen_idx < len(ground_tokens):
                gen_tok_text = tokenizer.decode([ground_tokens[gen_idx]])
                coloured_parts.append(_c(gen_tok_text, ORANGE))

            # 3. Remaining (wrong) proposal tokens (red)
            for i in range(correct_here, len(proposal)):
                wrong_text = tokenizer.decode([proposal[i]])
                coloured_parts.append(_c(wrong_text, RED))

            prop_col = "".join(coloured_parts)

            # Assemble plain strings for each category.
            correct_text = tokenizer.decode(proposal[:correct_here]) if correct_here else ""

            gen_idx_print = start_idx + correct_here
            generated_text = (
                tokenizer.decode([ground_tokens[gen_idx_print]])
                if gen_idx_print < len(ground_tokens)
                else ""
            )

            wrong_text = tokenizer.decode(proposal[correct_here:]) if correct_here < len(proposal) else ""

            ctx_text_full = tokenizer.decode(ctx)
            ctx_tail = "\n".join(ctx_text_full.splitlines()[-3:])

            print(f"[{alg_name}] iter {iteration:04d}")
            print(" correct   :", _c(correct_text, GREEN))
            print(" generated :", _c(generated_text, ORANGE))
            print(" wrong     :", _c(wrong_text, RED))
            print(f" propose_ms: {propose_ms:.2f} ms")
            print(" context   :", _c(ctx_tail, GREY))
            print()

        # If debug_iteration requested, dump full context once.
        if debug_iteration is not None and iteration == debug_iteration:
            full_ctx_text = tokenizer.decode(ctx)
            print("\n---- DEBUG full context up to iteration", iteration, "----")
            print(_c(full_ctx_text, GREY))
            print("---- END DEBUG ----\n")


        iteration += 1

    avg_tokens = tokens_total / iteration if iteration else 0.0
    avg_time_ms = propose_time_total_ms / iteration if iteration else 0.0

    return SimulationResult(
        correct_predictions=num_correct,
        wrong_predictions=num_wrong,
        predicted_mask=predicted_mask,
        iterations=iteration,
        total_time_ms=propose_time_total_ms,
        avg_tokens_per_iter=avg_tokens,
        avg_time_ms_per_iter=avg_time_ms,
    )
