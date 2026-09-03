"""Immutable rankings and the first shared fusion strategy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any

from ..errors import InvalidModelOutputError


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

    totals: dict[str, float] = {}
    examples: dict[str, RankedItem[Item]] = {}
    seen_order: dict[str, int] = {}
    origins: set[str] = set()
    next_order = 0
    for _source, ranking in rankings.items():
        origins.update(ranking.origins)
        for item in ranking:
            totals[item.id] = totals.get(item.id, 0.0) + 1.0 / (k + item.rank)
            examples.setdefault(item.id, item)
            if item.id not in seen_order:
                seen_order[item.id] = next_order
                next_order += 1

    ordered = sorted(
        totals,
        key=lambda item_id: (-totals[item_id], seen_order[item_id]),
    )
    items = tuple(
        replace(
            examples[item_id],
            score=totals[item_id],
            provenance=examples[item_id].provenance + ("rrf",),
        )
        for item_id in ordered[:top_k]
    )
    return Ranking(items, "rrf", frozenset(origins))
