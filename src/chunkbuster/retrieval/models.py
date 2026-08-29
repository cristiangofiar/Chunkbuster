"""Immutable retrieval value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from ..core.ranking import Ranking


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("chunk id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("chunk text must be a non-empty string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RetrievalOutput:
    terminal_node: str
    ranking: Ranking[Chunk]

    @property
    def status(self) -> Literal["completed", "empty"]:
        return "completed" if self.ranking else "empty"


@dataclass(frozen=True, slots=True)
class RetrievalPipelineResult:
    query_id: str | None
    outputs: Mapping[str, RetrievalOutput]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
