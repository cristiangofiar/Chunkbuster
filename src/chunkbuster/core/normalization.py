"""Small score normalizers shared by ranking pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt
from typing import Literal

Normalization = Literal["none", "min_max", "z_score"]


def normalize_scores(
    scores: Iterable[float],
    method: Normalization = "none",
) -> tuple[float, ...]:
    """Normalize one score list without changing its order."""
    values = tuple(float(score) for score in scores)
    if not values or method == "none":
        return values
    if method == "min_max":
        low, high = min(values), max(values)
        scale = high - low
        return (
            tuple(0.0 for _ in values)
            if scale == 0
            else tuple((score - low) / scale for score in values)
        )
    if method == "z_score":
        mean = sum(values) / len(values)
        deviation = sqrt(sum((score - mean) ** 2 for score in values) / len(values))
        return (
            tuple(0.0 for _ in values)
            if deviation == 0
            else tuple((score - mean) / deviation for score in values)
        )
    raise ValueError(f"unknown score normalization {method!r}")
