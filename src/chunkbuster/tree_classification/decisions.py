"""Deterministic terminal deciders."""

from __future__ import annotations

from ..core.ranking import Ranking
from .config import (
    DeciderConfig,
    ThresholdDeciderConfig,
    TopKDeciderConfig,
)
from .models import ClassificationDecision, TaxonomyPath


def decide(
    spec: DeciderConfig,
    ranking: Ranking[TaxonomyPath],
) -> ClassificationDecision:
    if isinstance(spec, TopKDeciderConfig):
        selected = ranking.items[: spec.count]
    elif isinstance(spec, ThresholdDeciderConfig):
        selected = tuple(item for item in ranking if item.score >= spec.min_score)
        if spec.count is not None:
            selected = selected[: spec.count]
    else:
        selected = ranking.items[:1]
    return ClassificationDecision(
        spec.name,
        "selected" if selected else "abstained",
        tuple(selected),
    )
