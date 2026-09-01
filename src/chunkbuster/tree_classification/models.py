"""Domain models for strict-forest classification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal

from ..core.ranking import RankedItem


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class TaxonomyNode:
    id: str
    label: str
    text: str | None = None
    embedding: tuple[float, ...] | None = None
    tokens: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("taxonomy node id must be a non-empty string")
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("taxonomy node label must be a non-empty string")
        text = self.label if self.text is None else self.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("taxonomy node text must be a non-empty string")
        object.__setattr__(self, "text", text)
        if self.embedding is not None:
            vector = tuple(float(value) for value in self.embedding)
            if not vector or not all(isfinite(value) for value in vector):
                raise ValueError("taxonomy node embedding must contain finite values")
            object.__setattr__(self, "embedding", vector)
        if self.tokens is not None:
            object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TaxonomyEdge:
    parent_id: str
    child_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parent_id or not self.child_id:
            raise ValueError("taxonomy edge endpoints must be non-empty strings")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Taxonomy:
    id: str
    nodes: tuple[TaxonomyNode, ...]
    edges: tuple[TaxonomyEdge, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("taxonomy id must be a non-empty string")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TaxonomyPath:
    node_ids: tuple[str, ...]
    nodes: tuple[TaxonomyNode, ...]

    @property
    def id(self) -> str:
        return json.dumps(self.node_ids, ensure_ascii=False, separators=(",", ":"))

    @property
    def root_id(self) -> str:
        return self.node_ids[0]

    @property
    def leaf_id(self) -> str:
        return self.node_ids[-1]

    @property
    def depth(self) -> int:
        return len(self.node_ids) - 1


@dataclass(frozen=True, slots=True)
class DecisionSelection:
    path_ids: tuple[str, ...] = ()
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.path_ids, str):
            raise ValueError("decision path IDs must be a sequence of strings")
        path_ids = tuple(self.path_ids)
        if not all(isinstance(path_id, str) and path_id for path_id in path_ids):
            raise ValueError("decision path IDs must be non-empty strings")
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("decision path IDs must be unique")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("decision reason must be a string or None")
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ClassificationDecision:
    name: str
    status: Literal["selected", "abstained"]
    selected: tuple[RankedItem[TaxonomyPath], ...]
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TreeClassificationResult:
    query_id: str | None
    taxonomy_id: str
    outputs: Mapping[str, ClassificationDecision]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
