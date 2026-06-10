# -*- coding: utf-8 -*-
"""DEPRECATED shim. The heuristic engine moved to predictors/legacy/.

The overdue/proximity heuristic is the gambler's fallacy and is statistically
invalid for a fair wheel. It is retained ONLY for education (Lab Mode) and is
no longer offered in the main engine dropdown. New code should import from
`predictors.legacy.heuristic_engine`. This shim keeps old imports working.
"""
import warnings

from predictors.legacy.heuristic_engine import HeuristicEngine

warnings.warn(
    "predictors.heuristic_engine is deprecated; the heuristic (gambler's "
    "fallacy) moved to predictors.legacy.heuristic_engine and is Lab-Mode only.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HeuristicEngine"]
