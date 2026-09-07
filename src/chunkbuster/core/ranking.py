"""Immutable rankings and the first shared fusion strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any

from ..errors import InvalidModelOutputError
from .normalization import Normalization, normalize_scores


@dataclass(frozen=True, slots=True)
class RankedItem[Item]:
    """One canonically identified item at one ranking stage."""

    id: str
    item: Item
    score: float
    rank: int = 0
    provenance: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("ranked item id must be a non-empty string")
        if not isfinite(float(self.score)):
            raise ValueError("ranked item score must be finite")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Ranking[Item]:
    """An ordered sequence with unique IDs and normalized one-based ranks."""

    items: tuple[RankedItem[Item], ...] = ()
    score_semantics: str = "raw"
    origins: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        items = tuple(self.items)
        ids = tuple(item.id for item in items)
        if len(ids) != len(set(ids)):
            raise ValueError("ranking item ids must be unique")
        object.__setattr__(
            self,
            "items",
            tuple(replace(item, rank=index) for index, item in enumerate(items, 1)),
        )
        object.__setattr__(self, "origins", frozenset(self.origins))

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)

    def to_text(
        self,
        formatter: Callable[[RankedItem[Item]], str] | None = None,
    ) -> str:
        if formatter is None:
            return "\n".join(
                f"- {candidate.id}: {candidate.item}" for candidate in self.items
            )
        return "\n".join(formatter(candidate) for candidate in self.items)

    def top(self, count: int) -> Ranking[Item]:
        if count <= 0:
            raise ValueError("count must be positive")
        return Ranking(self.items[:count], self.score_semantics, self.origins)


def require_ranking(value: Any, *, component: str) -> Ranking[Any]:
    """Reject loose provider output at the component boundary."""
    if not isinstance(value, Ranking):
        raise InvalidModelOutputError(f"{component!r} must return Ranking")
    return value


def require_subset[Item](
    output: Ranking[Item],
    candidates: Ranking[Any],
    *,
    component: str,
) -> Ranking[Item]:
    """Ensure a unary/candidate stage cannot invent identities."""
    unknown = set(output.ids) - set(candidates.ids)
    if unknown:
        raise InvalidModelOutputError(
            f"{component!r} returned identities outside its input: {sorted(unknown)!r}"
        )
    canonical = {item.id: item.item for item in candidates}
    items = tuple(replace(item, item=canonical[item.id]) for item in output)
    return Ranking(items, output.score_semantics, candidates.origins)


def reciprocal_rank_fusion[Item](
    rankings: Mapping[str, Ranking[Item]],
    *,
    top_k: int,
    k: int = 60,
) -> Ranking[Item]:
    """Fuse named rankings with deterministic reciprocal-rank fusion."""
    if len(rankings) < 2:
        raise ValueError("rrf requires at least two rankings")
    if top_k <= 0 or k <= 0:
        raise ValueError("rrf top_k and k must be positive")
    return fuse_rankings(
        rankings,
        method="rrf",
        top_k=top_k,
        k=k,
    )


def fuse_rankings[Item](
    rankings: Mapping[str, Ranking[Item]],
    *,
    method: str,
    top_k: int,
    weights: Mapping[str, float] | None = None,
    normalization: Normalization = "none",
    k: int = 60,
    normalize_missing_as_zero: bool = False,
) -> Ranking[Item]:
    """Fuse rankings by rank, normalized score sum, or score consensus."""
    if len(rankings) < 2:
        raise ValueError("fusion requires at least two rankings")
    if top_k <= 0:
        raise ValueError("fusion top_k must be positive")
    weights = dict(weights or {name: 1.0 for name in rankings})
    if set(weights) != set(rankings):
        raise ValueError("fusion weights must match ranking names")
    if any(not isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("fusion weights must be finite and non-negative")
    if not any(weights.values()):
        raise ValueError("fusion requires at least one positive weight")

    if method in {"rrf", "weighted_rrf"}:
        if normalization != "none":
            raise ValueError("weighted_rrf does not accept score normalization")
        if k <= 0:
            raise ValueError("rrf k must be positive")
        scores = {
            name: {item.id: weights[name] / (k + item.rank) for item in ranking}
            for name, ranking in rankings.items()
        }
    elif method in {"weighted_sum", "comb_mnz"}:
        scores = {}
        known_ids = {item.id for ranking in rankings.values() for item in ranking}
        for name, ranking in rankings.items():
            missing = len(known_ids) - len(ranking) if normalize_missing_as_zero else 0
            raw_scores = tuple(item.score for item in ranking) + (0.0,) * missing
            normalized = normalize_scores(raw_scores, normalization)[: len(ranking)]
            scores[name] = {
                item.id: weights[name] * score
                for item, score in zip(ranking, normalized, strict=True)
            }
    else:
        raise ValueError(f"unknown ranking fusion {method!r}")

    totals: dict[str, float] = {}
    examples: dict[str, RankedItem[Item]] = {}
    seen_order: dict[str, int] = {}
    origins: set[str] = set()
    for name, ranking in rankings.items():
        origins.update(ranking.origins)
        for item in ranking:
            totals[item.id] = totals.get(item.id, 0.0) + scores[name][item.id]
            examples.setdefault(item.id, item)
            seen_order.setdefault(item.id, len(seen_order))
    if method == "comb_mnz":
        memberships = tuple(set(ranking.ids) for ranking in rankings.values())
        for item_id in totals:
            totals[item_id] *= sum(item_id in ids for ids in memberships)

    ordered = sorted(
        totals, key=lambda item_id: (-totals[item_id], seen_order[item_id])
    )
    items = tuple(
        replace(
            examples[item_id],
            score=totals[item_id],
            provenance=examples[item_id].provenance + (method,),
        )
        for item_id in ordered[:top_k]
    )
    return Ranking(items, method, frozenset(origins))
