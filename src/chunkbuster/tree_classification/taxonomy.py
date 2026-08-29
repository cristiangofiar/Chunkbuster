"""Strict-forest validation and deterministic path enumeration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..errors import InvalidTaxonomyError
from .models import Taxonomy, TaxonomyNode, TaxonomyPath


@dataclass(frozen=True, slots=True)
class TaxonomySnapshot:
    taxonomy: Taxonomy
    nodes_by_id: Mapping[str, TaxonomyNode]
    roots: tuple[str, ...]
    paths: tuple[TaxonomyPath, ...]


def build_snapshot(taxonomy: Taxonomy) -> TaxonomySnapshot:
    """Validate one or more strict trees and enumerate every root-leaf path."""
    if not taxonomy.nodes:
        raise InvalidTaxonomyError("taxonomy must contain at least one node")
    node_ids = tuple(node.id for node in taxonomy.nodes)
    if len(node_ids) != len(set(node_ids)):
        raise InvalidTaxonomyError("taxonomy node ids must be unique")

    nodes = {node.id: node for node in taxonomy.nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    parent: dict[str, str] = {}
    seen_edges: set[tuple[str, str]] = set()
    for edge in taxonomy.edges:
        pair = (edge.parent_id, edge.child_id)
        if pair in seen_edges:
            raise InvalidTaxonomyError(f"duplicate taxonomy edge {pair!r}")
        seen_edges.add(pair)
        if edge.parent_id not in nodes or edge.child_id not in nodes:
            raise InvalidTaxonomyError(
                f"taxonomy edge has an unknown endpoint: {pair!r}"
            )
        if edge.parent_id == edge.child_id:
            raise InvalidTaxonomyError(f"taxonomy contains self-loop {pair!r}")
        if edge.child_id in parent:
            raise InvalidTaxonomyError(
                f"node {edge.child_id!r} has more than one parent"
            )
        parent[edge.child_id] = edge.parent_id
        children[edge.parent_id].append(edge.child_id)

    roots = tuple(node_id for node_id in node_ids if node_id not in parent)
    if not roots:
        raise InvalidTaxonomyError("taxonomy has no root")

    visiting: set[str] = set()
    visited: set[str] = set()
    path_ids: list[tuple[str, ...]] = []

    def visit(node_id: str, prefix: tuple[str, ...]) -> None:
        if node_id in visiting:
            raise InvalidTaxonomyError("taxonomy contains a cycle")
        if node_id in visited:
            return
        visiting.add(node_id)
        current = prefix + (node_id,)
        if not children[node_id]:
            path_ids.append(current)
        else:
            for child_id in children[node_id]:
                visit(child_id, current)
        visiting.remove(node_id)
        visited.add(node_id)

    for root_id in roots:
        visit(root_id, ())
    if len(visited) != len(nodes):
        unreachable = set(nodes) - visited
        raise InvalidTaxonomyError(
            f"taxonomy contains a cycle or unreachable nodes: {sorted(unreachable)!r}"
        )

    paths = tuple(
        TaxonomyPath(ids, tuple(nodes[node_id] for node_id in ids))
        for ids in path_ids
    )
    return TaxonomySnapshot(
        taxonomy,
        MappingProxyType(nodes),
        roots,
        paths,
    )
