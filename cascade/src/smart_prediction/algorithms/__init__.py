"""Public re-export of all *predictor* algorithms used by the prototype."""

from __future__ import annotations

# Base class ---------------------------------------------------------------

from .base import PredictorBase  # noqa: F401 (re-export only)

# Concrete predictors ------------------------------------------------------

from .naive_sequential import NaiveSequentialPredictor  # noqa: F401
from .line_diff import LineDiffPredictor  # noqa: F401
from .greedy_suffix import GreedySuffixPredictor  # noqa: F401
from .myers_streaming import MyersStreamingPredictor  # noqa: F401
from .myers_lines import MyersLinePredictor  # noqa: F401
from .myers_incremental import MyersIncrementalPredictor  # noqa: F401
from .fuzz import RapidFuzzPredictor  # noqa: F401

__all__ = [
    "PredictorBase",
    "NaiveSequentialPredictor",
    "LineDiffPredictor",
    "GreedySuffixPredictor",
    "MyersStreamingPredictor",
    "MyersLinePredictor",
    "MyersIncrementalPredictor",
    "RapidFuzzPredictor",
]
