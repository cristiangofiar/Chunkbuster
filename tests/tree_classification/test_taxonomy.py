from __future__ import annotations

import pytest

from chunkbuster.errors import InvalidTaxonomyError
from chunkbuster.tree_classification.models import Taxonomy, TaxonomyEdge, TaxonomyNode
from chunkbuster.tree_classification.taxonomy import build_snapshot


def _node(node_id: str) -> TaxonomyNode:
    return TaxonomyNode(node_id, node_id)


def test_forest_supports_multiple_roots_and_singleton_subtrees() -> None:
    taxonomy = Taxonomy(
        "forest",
        (
            _node("root_a"),
            _node("leaf_a"),
            _node("root_b"),
            _node("leaf_b"),
            _node("single"),
        ),
        (
            TaxonomyEdge("root_a", "leaf_a"),
            TaxonomyEdge("root_b", "leaf_b"),
        ),
    )

    snapshot = build_snapshot(taxonomy)

    assert snapshot.roots == ("root_a", "root_b", "single")
    assert tuple(path.node_ids for path in snapshot.paths) == (
        ("root_a", "leaf_a"),
        ("root_b", "leaf_b"),
        ("single",),
    )


def test_forest_rejects_a_node_with_multiple_parents() -> None:
    taxonomy = Taxonomy(
        "not-a-forest",
        (_node("root_a"), _node("root_b"), _node("leaf")),
        (
            TaxonomyEdge("root_a", "leaf"),
            TaxonomyEdge("root_b", "leaf"),
        ),
    )

    with pytest.raises(InvalidTaxonomyError, match="more than one parent"):
        build_snapshot(taxonomy)


def test_forest_rejects_cycles() -> None:
    taxonomy = Taxonomy(
        "cyclic",
        (_node("valid_root"), _node("a"), _node("b")),
        (TaxonomyEdge("a", "b"), TaxonomyEdge("b", "a")),
    )

    with pytest.raises(InvalidTaxonomyError, match="cycle"):
        build_snapshot(taxonomy)
