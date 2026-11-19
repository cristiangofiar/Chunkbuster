"""Injection container for external components."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .models import Query
from .ranking import Ranking


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


async def invoke_router[Item](
    component: Any,
    query: Query,
    rankings: Mapping[str, Ranking[Item]],
    *,
    parameters: Mapping[str, Any],
) -> Any:
    """Give the router a private copy of its settings behind a read-only mapping."""
    return await component.route(
        query,
        _freeze(rankings),
        parameters=MappingProxyType(deepcopy(dict(parameters))),
    )


@dataclass(frozen=True, slots=True)
class ComponentBindings:
    """Async external components, keyed by the config's ``binding`` value."""

    preprocessors: Mapping[str, Any] = field(default_factory=dict)
    retrievers: Mapping[str, Any] = field(default_factory=dict)
    path_scorers: Mapping[str, Any] = field(default_factory=dict)
    deciders: Mapping[str, Any] = field(default_factory=dict)
    routers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "preprocessors",
            "retrievers",
            "path_scorers",
            "deciders",
            "routers",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
