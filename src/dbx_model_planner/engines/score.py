"""Composite scoring for candidate compute nodes."""

from __future__ import annotations

import math
from math import exp

from dbx_model_planner.domain.common import FitLevel

_FIT_SCORES: dict[FitLevel, int] = {
    FitLevel.SAFE: 100,
    FitLevel.BORDERLINE: 50,
    FitLevel.UNLIKELY: 10,
}

_BASE_WEIGHTS = {
    "fit": 40,
    "utilization": 30,
    "speed": 20,
    "cost": 10,
}


def compute_candidate_score(
    fit_level: FitLevel,
    estimated_memory_gb: float | None,
    total_gpu_memory_gb: float,
    estimated_tok_s: float | None = None,
    cost_per_hour: float | None = None,
) -> int:
    """Compute a 0-100 composite score for a candidate.

    Dimensions:
    - Fit score (40% weight): SAFE=100, BORDERLINE=50, UNLIKELY=10
    - Utilization score (30% weight): Sweet spot is 50-80% memory usage.
      100 at 65%, drops off toward 0% and 100% usage.
    - Speed score (20% weight): Normalized tok/s (higher is better, cap at 100)
    - Cost score (10% weight): Lower cost is better (inverse relationship)

    If tok/s or cost is unavailable, redistribute weights proportionally.
    """
    scores: dict[str, float] = {}
    weights: dict[str, int] = {}

    # Fit score (always available)
    scores["fit"] = float(_FIT_SCORES.get(fit_level, 10))
    weights["fit"] = _BASE_WEIGHTS["fit"]

    # Utilization score (always available when we have memory figures)
    if estimated_memory_gb is not None and total_gpu_memory_gb > 0 and estimated_memory_gb > 0:
        usage_pct = (estimated_memory_gb / total_gpu_memory_gb) * 100
        scores["utilization"] = 100.0 * exp(-((usage_pct - 65) / 25) ** 2)
        weights["utilization"] = _BASE_WEIGHTS["utilization"]
    else:
        # No memory info — skip this dimension
        pass

    # Speed score (optional)
    if estimated_tok_s is not None and estimated_tok_s > 0:
        scores["speed"] = min(100.0, math.log2(estimated_tok_s + 1) * 15.0)
        weights["speed"] = _BASE_WEIGHTS["speed"]

    # Cost score (optional)
    if cost_per_hour is not None and cost_per_hour >= 0:
        scores["cost"] = 100.0 / (1.0 + cost_per_hour * 0.5)
        weights["cost"] = _BASE_WEIGHTS["cost"]

    # Weighted average with redistribution
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0

    weighted_sum = sum(scores[k] * weights[k] for k in scores)
    result = weighted_sum / total_weight
    return max(0, min(100, round(result)))
