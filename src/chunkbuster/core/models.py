"""Query values and input normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Query:
    """One query; caller-owned metadata is preserved without conversion or copying."""

    text: str
    id: str | None = None
    metadata: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("query text must be a non-empty string")
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise ValueError("query id must be a non-empty string or None")


def as_query(value: str | Query) -> Query:
    """Normalize the public query shorthand."""
    return Query(value) if isinstance(value, str) else value
