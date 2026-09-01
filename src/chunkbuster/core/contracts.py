"""Injection container for external components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class ComponentBindings:
    """Concrete external objects, keyed by the config's ``binding`` value."""

    preprocessors: Mapping[str, Any] = field(default_factory=dict)
    retrievers: Mapping[str, Any] = field(default_factory=dict)
    path_scorers: Mapping[str, Any] = field(default_factory=dict)
    deciders: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("preprocessors", "retrievers", "path_scorers", "deciders"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
