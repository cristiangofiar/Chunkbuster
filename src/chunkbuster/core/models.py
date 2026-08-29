"""Shared immutable value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a detached, read-only shallow mapping."""
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class Query:
    """One text query with optional caller identity and metadata."""

    text: str
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be a non-empty string")
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise ValueError("query id must be a non-empty string or None")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


def as_query(value: str | Query) -> Query:
    """Normalize the public query shorthand."""
    return Query(value) if isinstance(value, str) else value

