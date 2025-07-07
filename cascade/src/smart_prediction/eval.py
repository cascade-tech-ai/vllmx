"""Quick benchmarking helper for smart-prediction algorithms.

Usage
-----
From the repository root run e.g.

    python -m joev.src.smart_prediction.eval \
        --ground doc_full.txt \
        --prediction doc_pred.txt \
        --algorithms naive line_diff greedy_suffix

If *--ground* / *--prediction* are omitted the built-in poem example is used
with the *missing-stanza* variant as prediction.

The tool prints a small table with *coverage*, *wrong tokens* and *score* for
each predictor class so you can quickly compare their behaviour on a given
input pair.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
from typing import List, Type

from . import examples  # built-in test snippets
from .algorithms.naive_sequential import NaiveSequentialPredictor
from .algorithms.line_diff import LineDiffPredictor
from .algorithms.greedy_suffix import GreedySuffixPredictor
from .algorithms.fuzz import RapidFuzzPredictor
from .algorithms.fuzz_lines import RapidFuzzLinePredictor
from .algorithms.lcs_lines import SequenceMatcherLinePredictor
from .simulator import SimulationResult, simulate
from .tokenizer import BasicTokenizer

# ---------------------------------------------------------------------------
# Registry of available predictors
# ---------------------------------------------------------------------------


PREDICTOR_REGISTRY: dict[str, Type] = {
    "naive": NaiveSequentialPredictor,
    "line_diff": LineDiffPredictor,
    "greedy_suffix": GreedySuffixPredictor,
    "rapidfuzz": RapidFuzzPredictor,
    "rapidfuzz_lines": RapidFuzzLinePredictor,
    "sm_lines": SequenceMatcherLinePredictor,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def run_one(
    predictor_name: str,
    predicted_text: str,
    ground_truth_text: str,
    *,
    lookahead: int = 32,
    verbose: bool = False,
    iterations: int | None = None,
    debug_iteration: int | None = None,
) -> SimulationResult:
    predictor_cls = PREDICTOR_REGISTRY[predictor_name]
    return simulate(
        predictor_cls,
        predicted_text=predicted_text,
        ground_truth_text=ground_truth_text,
        lookahead=lookahead,
        tokenizer=BasicTokenizer(),
        verbose=verbose,
        max_iterations=iterations,
        alg_name=predictor_name,
        debug_iteration=debug_iteration,
    )


def main(argv: List[str] | None = None) -> None:  # noqa: D401 simple
    parser = argparse.ArgumentParser(description="Evaluate smart-prediction algorithms")
    parser.add_argument("--ground",
                        type=pathlib.Path,
                        help="Path to *ground-truth* text file.")
    parser.add_argument("--prediction",
                        type=pathlib.Path,
                        help="Path to *prediction* text file.")
    parser.add_argument("--algorithms",
                        nargs="+",
                        default=list(PREDICTOR_REGISTRY.keys()),
                        help=f"Predictor keys {list(PREDICTOR_REGISTRY.keys())} (default: all).")
    parser.add_argument("--lookahead",
                        type=int,
                        default=32,
                        help="Maximum speculative tokens per step (default: 32).")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose per-iteration output")

    parser.add_argument("-i", "--iterations", type=int, help="Limit number of iterations to run")

    parser.add_argument("--debug", type=int, help="Iteration index to dump full context")

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load texts – fallback to built-in poem example if not provided.
    # ------------------------------------------------------------------
    if args.ground and args.prediction:
        ground_text = _read_text(args.ground)
        prediction_text = _read_text(args.prediction)
    else:
        ground_text = examples.POEM_THREE_STANZAS
        prediction_text = examples.POEM_MISSING_MIDDLE_STANZA
        print("[info] Using built-in poem example (missing middle stanza).\n")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    rows = []
    for key in args.algorithms:
        if key not in PREDICTOR_REGISTRY:
            raise SystemExit(f"Unknown algorithm: {key!r}")
        res = run_one(
            key,
            prediction_text,
            ground_text,
            lookahead=args.lookahead,
            verbose=args.verbose,
            iterations=args.iterations,
            debug_iteration=args.debug,
        )
        rows.append(
            (
                key,
                res.coverage * 100,
                res.wrong_predictions,
                res.score,
                res.iterations,
                res.avg_tokens_per_iter,
                res.avg_time_ms_per_iter,
                res.total_time_ms,
            )
        )

    # ------------------------------------------------------------------
    # Render simple table
    # ------------------------------------------------------------------
    hdr = (
        f"{'algorithm':<15}  {'coverage %':>10}  {'wrong':>7}  {'score':>8}"
        f"  {'iters':>6}  {'tok/iter':>9}  {'ms/iter':>9}  {'total ms':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for (
        key,
        cov,
        wrong,
        score,
        iters,
        tok_iter,
        ms_iter,
        total_ms,
    ) in rows:
        print(
            f"{key:<15}  {cov:10.1f}  {wrong:7d}  {score:8.2f}  "
            f"{iters:6d}  {tok_iter:9.2f}  {ms_iter:9.2f}  {total_ms:10.1f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
