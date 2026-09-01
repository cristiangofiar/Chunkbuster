"""Deterministic and bound terminal deciders."""

from __future__ import annotations

from ..core._async import resolve
from ..core.models import Query
from ..core.ranking import Ranking
from ..errors import InvalidModelOutputError
from .config import (
    DeciderConfig,
    LLMDeciderConfig,
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


async def decide_with_llm(
    spec: LLMDeciderConfig,
    component: object,
    query: Query,
    ranking: Ranking[TaxonomyPath],
) -> ClassificationDecision:
    raw = await resolve(component.decide(query, ranking, count=spec.count))
    if isinstance(raw, str):
        raise InvalidModelOutputError("LLM decider must return an iterable of path IDs")
    try:
        path_ids = tuple(raw)
    except TypeError as exc:
        raise InvalidModelOutputError(
            "LLM decider must return an iterable of path IDs"
        ) from exc
    if not all(isinstance(path_id, str) for path_id in path_ids):
        raise InvalidModelOutputError("LLM decider path IDs must be strings")
    if len(path_ids) != len(set(path_ids)):
        raise InvalidModelOutputError("LLM decider returned duplicate path IDs")
    if len(path_ids) > spec.count:
        raise InvalidModelOutputError(
            f"LLM decider returned {len(path_ids)} paths; maximum is {spec.count}"
        )
    by_id = {item.id: item for item in ranking}
    unknown = set(path_ids) - set(by_id)
    if unknown:
        raise InvalidModelOutputError(
            f"LLM decider returned unknown path IDs: {sorted(unknown)!r}"
        )
    selected = tuple(by_id[path_id] for path_id in path_ids)
    return ClassificationDecision(
        spec.name,
        "selected" if selected else "abstained",
        selected,
    )
