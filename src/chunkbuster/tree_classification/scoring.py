"""Pure dense and path scoring for the first vertical."""

from __future__ import annotations

from math import isfinite, sqrt

from ..core.ranking import RankedItem, Ranking
from ..errors import PreprocessingError
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


def score_paths(
    snapshot: TaxonomySnapshot,
    query_embedding: tuple[float, ...],
    *,
    similarity: str,
    top_k: int,
    source: str,
) -> Ranking[TaxonomyPath]:
    node_scores = {
        node_id: _similarity(query_embedding, node.embedding or (), similarity)
        for node_id, node in snapshot.nodes_by_id.items()
    }
    candidates = tuple(
        RankedItem(
            path.id,
            path,
            sum(node_scores[node_id] for node_id in path.node_ids)
            / len(path.node_ids),
            provenance=(source,),
        )
        for path in snapshot.paths
    )
    order = {item.id: index for index, item in enumerate(candidates)}
    ranked = sorted(candidates, key=lambda item: (-item.score, order[item.id]))
    return Ranking(tuple(ranked[:top_k]), "similarity", frozenset({source}))
