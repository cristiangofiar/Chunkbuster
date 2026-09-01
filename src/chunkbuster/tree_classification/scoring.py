"""Pure dense and path scoring for the first vertical."""

from __future__ import annotations

from math import isfinite, sqrt

from ..core.ranking import RankedItem, Ranking
from ..errors import InvalidModelOutputError, PreprocessingError
from .config import MeanPathScorerConfig, WeightedSumPathScorerConfig
from .models import TaxonomyPath
from .taxonomy import TaxonomySnapshot


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
