"""Pure dense, BM25, node-ranking, and path-scoring functions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log, sqrt
from types import MappingProxyType

from ..core.ranking import RankedItem, Ranking
from ..errors import InvalidModelOutputError, PreprocessingError
from .config import (
    BM25NodeScorerConfig,
    MeanPathScorerConfig,
    WeightedSumPathScorerConfig,
)
from .models import TaxonomyNode, TaxonomyPath
from .taxonomy import TaxonomySnapshot

type NodeScores = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BM25Index:
    document_frequencies: Mapping[str, int]
    term_frequencies: Mapping[str, Mapping[str, int]]
    document_lengths: Mapping[str, int]
    average_document_length: float
    document_count: int


def validate_tokens(value: object, *, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise PreprocessingError(f"{label} must be a sequence of tokens")
    try:
        tokens = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PreprocessingError(f"{label} must be a sequence of tokens") from exc
    if not all(isinstance(token, str) and token for token in tokens):
        raise PreprocessingError(f"{label} must contain non-empty strings")
    return tokens


def validate_vector(
    vector: object,
    *,
    dimensions: int,
    label: str,
) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise PreprocessingError(f"{label} must be a numeric vector") from exc
    if len(values) != dimensions:
        raise PreprocessingError(
            f"{label} has dimension {len(values)}; expected {dimensions}"
        )
    if not all(isfinite(value) for value in values):
        raise PreprocessingError(f"{label} must contain only finite values")
    return values


def _similarity(left: tuple[float, ...], right: tuple[float, ...], kind: str) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if kind == "dot_product":
        return dot
    if kind == "euclidean":
        return -sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise PreprocessingError("cosine similarity requires non-zero vectors")
    return dot / (left_norm * right_norm)


def score_nodes(
    snapshot: TaxonomySnapshot,
    query_embedding: tuple[float, ...],
    *,
    similarity: str,
) -> dict[str, float]:
    return {
        node_id: _similarity(query_embedding, node.embedding or (), similarity)
        for node_id, node in snapshot.nodes_by_id.items()
    }


def build_bm25_index(snapshot: TaxonomySnapshot) -> BM25Index:
    frequencies: Counter[str] = Counter()
    term_frequencies = {}
    lengths = {}
    for node_id, node in snapshot.nodes_by_id.items():
        tokens = node.tokens or ()
        frequencies.update(set(tokens))
        term_frequencies[node_id] = MappingProxyType(dict(Counter(tokens)))
        lengths[node_id] = len(tokens)
    return BM25Index(
        MappingProxyType(dict(frequencies)),
        MappingProxyType(term_frequencies),
        MappingProxyType(lengths),
        sum(lengths.values()) / len(lengths),
        len(lengths),
    )


def score_bm25_nodes(
    snapshot: TaxonomySnapshot,
    index: BM25Index,
    query_tokens: tuple[str, ...],
    spec: BM25NodeScorerConfig,
) -> dict[str, float]:
    if not query_tokens or index.average_document_length == 0:
        return {}
    scores = {}
    for node_id in snapshot.nodes_by_id:
        term_frequencies = index.term_frequencies[node_id]
        length = index.document_lengths[node_id]
        score = 0.0
        for token in dict.fromkeys(query_tokens):
            frequency = term_frequencies.get(token, 0)
            if not frequency:
                continue
            document_frequency = index.document_frequencies[token]
            inverse_document_frequency = log(
                1
                + (index.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + spec.k1 * (
                1 - spec.b + spec.b * length / index.average_document_length
            )
            score += (
                inverse_document_frequency * frequency * (spec.k1 + 1) / denominator
            )
        if score > 0:
            scores[node_id] = score
    return scores


def rank_node_scores(
    snapshot: TaxonomySnapshot,
    scores: NodeScores,
    *,
    source: str,
) -> Ranking[TaxonomyNode]:
    candidates = tuple(
        RankedItem(node_id, snapshot.nodes_by_id[node_id], score, provenance=(source,))
        for node_id, score in scores.items()
    )
    order = {node_id: index for index, node_id in enumerate(snapshot.nodes_by_id)}
    return Ranking(
        tuple(sorted(candidates, key=lambda item: (-item.score, order[item.id]))),
        source,
        frozenset({source}),
    )


def builtin_path_score(
    path: TaxonomyPath,
    node_scores: tuple[float, ...],
    spec: MeanPathScorerConfig | WeightedSumPathScorerConfig,
) -> float:
    if isinstance(spec, MeanPathScorerConfig):
        return sum(node_scores) / len(node_scores)

    score = 0.0
    for term in spec.terms:
        name, weight = next(iter(term.items()))
        if name == "mean":
            value = sum(node_scores) / len(node_scores)
        elif name == "root":
            value = node_scores[0]
        elif name == "leaf":
            value = node_scores[-1]
        elif name == "lowest":
            value = min(node_scores)
        elif name == "highest":
            value = max(node_scores)
        else:
            level = int(name.removeprefix("level_"))
            try:
                value = node_scores[level]
            except IndexError as exc:
                raise PreprocessingError(
                    f"path {path.id} has no {name} after its root"
                ) from exc
        score += weight * value
    return score


def validate_path_score(value: object, *, path: TaxonomyPath) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise InvalidModelOutputError(
            f"path scorer returned a non-numeric score for {path.id}"
        ) from exc
    if not isfinite(score):
        raise InvalidModelOutputError(
            f"path scorer returned a non-finite score for {path.id}"
        )
    return score


def rank_paths(
    snapshot: TaxonomySnapshot,
    path_scores: dict[str, float],
    *,
    source: str,
) -> Ranking[TaxonomyPath]:
    candidates = tuple(
        RankedItem(
            path.id,
            path,
            path_scores[path.id],
            provenance=(source,),
        )
        for path in snapshot.paths
    )
    order = {item.id: index for index, item in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda item: (-item.score, order[item.id]))
    return Ranking(tuple(ranked), "path_score", frozenset({source}))
