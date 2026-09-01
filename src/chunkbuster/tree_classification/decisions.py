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
from .models import ClassificationDecision, DecisionSelection, TaxonomyPath


def decide(
    spec: DeciderConfig,
    ranking: Ranking[TaxonomyPath],
) -> DecisionSelection:
    if isinstance(spec, TopKDeciderConfig):
        selected = ranking.items[: spec.count]
    elif isinstance(spec, ThresholdDeciderConfig):
        selected = tuple(item for item in ranking if item.score >= spec.min_score)
        if spec.count is not None:
            selected = selected[: spec.count]
    else:
        selected = ranking.items[:1]
    return DecisionSelection(tuple(item.id for item in selected))


async def decide_with_llm(
    spec: LLMDeciderConfig,
    component: object,
    query: Query,
    ranking: Ranking[TaxonomyPath],
) -> DecisionSelection:
    raw = await resolve(component.decide(query, ranking, count=spec.count))
    if not isinstance(raw, DecisionSelection):
        raise InvalidModelOutputError("LLM decider must return DecisionSelection")
    return raw


def materialize_decision(
    spec: DeciderConfig,
    selection: DecisionSelection,
    ranking: Ranking[TaxonomyPath],
) -> ClassificationDecision:
    if isinstance(
        spec,
        (TopKDeciderConfig, ThresholdDeciderConfig, LLMDeciderConfig),
    ):
        limit = spec.count
    else:
        limit = 1
    if limit is not None and len(selection.path_ids) > limit:
        raise InvalidModelOutputError(
            f"decider returned {len(selection.path_ids)} paths; maximum is {limit}"
        )
    by_id = {item.id: item for item in ranking}
    unknown = set(selection.path_ids) - set(by_id)
    if unknown:
        raise InvalidModelOutputError(
            f"decider returned unknown path IDs: {sorted(unknown)!r}"
        )
    selected = tuple(by_id[path_id] for path_id in selection.path_ids)
    return ClassificationDecision(
        name=spec.name,
        status="selected" if selected else "abstained",
        selected=selected,
        reason=selection.reason,
        metadata=selection.metadata,
    )
