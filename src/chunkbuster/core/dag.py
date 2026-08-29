"""Minimal named-dependency validation using the standard library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType

from ..errors import BuildError


@dataclass(frozen=True, slots=True)
class CompiledDAG:
    order: tuple[str, ...]
    dependencies: Mapping[str, tuple[str, ...]]


def compile_dag(
    dependencies: Mapping[str, tuple[str, ...]],
    outputs: Mapping[str, str],
) -> CompiledDAG:
    """Validate references, cycles, outputs, and unused nodes."""
    deps = {name: tuple(values) for name, values in dependencies.items()}
    if not deps:
        raise BuildError("pipeline must declare at least one executable node")
    if not outputs:
        raise BuildError("pipeline must declare at least one output")
    for name, values in deps.items():
        missing = set(values) - set(deps)
        if missing:
            raise BuildError(
                f"node {name!r} references unknown nodes {sorted(missing)!r}"
            )
        if len(values) != len(set(values)):
            raise BuildError(f"node {name!r} repeats an input")
    missing_outputs = set(outputs.values()) - set(deps)
    if missing_outputs:
        raise BuildError(f"outputs reference unknown nodes {sorted(missing_outputs)!r}")
    try:
        order = tuple(TopologicalSorter(deps).static_order())
    except CycleError as exc:
        raise BuildError("pipeline graph contains a cycle") from exc

    reachable: set[str] = set()

    def visit(name: str) -> None:
        if name in reachable:
            return
        reachable.add(name)
        for dependency in deps[name]:
            visit(dependency)

    for terminal in outputs.values():
        visit(terminal)
    unused = set(deps) - reachable
    if unused:
        raise BuildError(f"pipeline contains unused nodes {sorted(unused)!r}")
    return CompiledDAG(
        tuple(name for name in order if name in reachable),
        MappingProxyType(deps),
    )
